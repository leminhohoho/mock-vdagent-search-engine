"""Postgres access (schema `mockserp`); the only module that writes SQL text.

Vectors travel as text literals ('[0.1,…]') and are cast inside SQL, so nothing depends on
which schema the `vector` type lives in.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg_pool import ConnectionPool, PoolTimeout

# pg_advisory_xact_lock key that serializes concurrent ingests ("mockserp" as a bigint).
INGEST_LOCK = int.from_bytes(b"mockserp", "big")
# undefined_table, invalid_schema_name, undefined_function
_UNDEFINED_OBJECT = {"42P01", "3F000", "42883"}


class StoreError(RuntimeError):
    """The database could not be reached or rejected a statement."""


class NotMigrated(StoreError):
    """The `mockserp` schema or one of its objects does not exist."""


class IndexMismatch(StoreError):
    """SQLSTATE MS001: nothing ingested, or the query embedding is from another model/dim."""


@dataclass(frozen=True)
class IndexMeta:
    version: str
    embedding_model: str
    dim: int
    n_docs: int
    n_chunks: int
    corpus_sha256: str
    built_at: datetime


@dataclass(frozen=True, eq=False)
class DocumentRecord:
    """One page ready to store: computed columns plus its chunks and their embeddings."""

    url: str
    norm_url: str
    host: str
    title: str
    raw_content: str
    published_date: datetime | None
    favicon: str | None
    search_tokens: str
    chunks: list[str]
    embeddings: np.ndarray  # [len(chunks), dim], L2-normalized


@dataclass(frozen=True)
class SearchRow:
    url: str
    title: str
    raw_content: str
    published_date: datetime | None
    favicon: str | None
    score: float
    snippets: list[str]


@dataclass(frozen=True)
class StoredDocument:
    id: int
    url: str
    raw_content: str
    favicon: str | None


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.9g}" for x in np.asarray(vector, dtype=np.float32).tolist()) + "]"


class Store:
    def __init__(self, conninfo: str, max_size: int = 5, timeout: float = 10.0):
        try:
            conninfo_to_dict(conninfo)
        except psycopg.Error as e:
            raise StoreError(f"invalid connection string: {e}") from e
        self._timeout = timeout
        self._pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=max_size,
            # prepare_threshold=None: no server-side prepared statements, so both Supabase
            # pooler modes (session and transaction) work.
            kwargs={"autocommit": True, "prepare_threshold": None},
            open=True,
            name="mockserp",
        )

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        try:
            with self._pool.connection(timeout=self._timeout) as conn:
                yield conn
        except PoolTimeout as e:
            raise StoreError(f"no database connection within {self._timeout:g}s: {e}") from e
        except psycopg.Error as e:
            if e.sqlstate == "MS001":
                raise IndexMismatch(e.diag.message_primary or str(e)) from e
            if e.sqlstate in _UNDEFINED_OBJECT:
                raise NotMigrated(str(e).strip()) from e
            raise StoreError(str(e).strip()) from e

    def meta(self) -> IndexMeta | None:
        with self._connection() as conn:
            row = conn.execute(
                "select version, embedding_model, dim, n_docs, n_chunks, corpus_sha256, built_at"
                " from mockserp.index_meta"
            ).fetchone()
        return IndexMeta(*row) if row else None

    def replace_corpus(self, records: list[DocumentRecord], meta: IndexMeta) -> None:
        """Swap in a new corpus atomically; readers see the old one until the commit."""
        with self._connection() as conn, conn.transaction():
            conn.execute("select pg_advisory_xact_lock(%s)", [INGEST_LOCK])
            conn.execute("delete from mockserp.documents")
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into mockserp.documents (url, norm_url, host, title, raw_content,"
                    " published_date, favicon, search_tokens)"
                    " values (%s, %s, %s, %s, %s, %s, %s, %s) returning id",
                    [
                        (
                            r.url,
                            r.norm_url,
                            r.host,
                            r.title,
                            r.raw_content,
                            r.published_date,
                            r.favicon,
                            r.search_tokens,
                        )
                        for r in records
                    ],
                    returning=True,
                )
                ids = []
                while True:
                    ids.append(cur.fetchone()[0])
                    if not cur.nextset():
                        break
                with cur.copy(
                    "copy mockserp.chunks (doc_id, ord, text, embedding) from stdin"
                ) as copy:
                    for doc_id, r in zip(ids, records, strict=True):
                        for ord_, (text, emb) in enumerate(
                            zip(r.chunks, r.embeddings, strict=True)
                        ):
                            copy.write_row((doc_id, ord_, text, vector_literal(emb)))
            conn.execute(
                "insert into mockserp.index_meta (version, embedding_model, dim, n_docs,"
                " n_chunks, corpus_sha256, built_at) values (%s, %s, %s, %s, %s, %s, %s)"
                " on conflict (singleton) do update set version = excluded.version,"
                " embedding_model = excluded.embedding_model, dim = excluded.dim,"
                " n_docs = excluded.n_docs, n_chunks = excluded.n_chunks,"
                " corpus_sha256 = excluded.corpus_sha256, built_at = excluded.built_at",
                [
                    meta.version,
                    meta.embedding_model,
                    meta.dim,
                    meta.n_docs,
                    meta.n_chunks,
                    meta.corpus_sha256,
                    meta.built_at,
                ],
            )

    def search(
        self,
        query_embedding: np.ndarray,
        embedding_model: str,
        *,
        max_results: int,
        chunks_per_source: int,
        include_domains: list[str],
        exclude_domains: list[str],
        prefer_domains: bool,
        window_start: datetime | None,
        window_end: datetime | None,
        drop_undated: bool,
        phrases: list[str],
        min_score: float | None = None,
    ) -> list[SearchRow]:
        with self._connection() as conn:
            rows = conn.execute(
                "select url, title, raw_content, published_date, favicon, score, snippets"
                " from mockserp.search(query_embedding => %s, embedding_model => %s,"
                " max_results => %s, chunks_per_source => %s, include_domains => %s::text[],"
                " exclude_domains => %s::text[], prefer_domains => %s, window_start => %s,"
                " window_end => %s, drop_undated => %s, phrases => %s::text[],"
                " min_score => %s)",
                [
                    vector_literal(query_embedding),
                    embedding_model,
                    max_results,
                    chunks_per_source,
                    include_domains,
                    exclude_domains,
                    prefer_domains,
                    window_start,
                    window_end,
                    drop_undated,
                    phrases,
                    min_score,
                ],
            ).fetchall()
        return [SearchRow(*row) for row in rows]

    def documents_by_norm_url(self, norm_urls: list[str]) -> dict[str, StoredDocument]:
        with self._connection() as conn:
            rows = conn.execute(
                "select norm_url, id, url, raw_content, favicon from mockserp.documents"
                " where norm_url = any(%s::text[])",
                [norm_urls],
            ).fetchall()
        return {norm_url: StoredDocument(*rest) for norm_url, *rest in rows}

    def extract_chunks(
        self, query_embedding: np.ndarray, embedding_model: str, doc_id: int, k: int
    ) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute(
                "select mockserp.extract_chunks(%s, %s, %s, %s)",
                [vector_literal(query_embedding), embedding_model, doc_id, k],
            ).fetchall()
        return [text for (text,) in rows]
