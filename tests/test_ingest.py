"""`ingest`: corpus files → one transaction replacing documents, chunks and index_meta."""

import re
import threading
import time

import psycopg
import pytest

from mockserp import db
from mockserp.chunker import chunk
from mockserp.corpus import load_corpus
from mockserp.db import StoreError
from mockserp.ingest import build_records, corpus_sha256, ingest

from .conftest import HashEmbedder, para, write_corpus

A1 = {
    "url": "https://www.alpha.example/one",
    "title": "Alpha one",
    "raw_content": "\n".join([para("First alpha paragraph."), para("Second alpha paragraph.")]),
    "published_date": "2026-09-29",
}
A2 = {"url": "https://alpha.example/two", "title": "Alpha two", "raw_content": "Short page."}
B1 = {"url": "https://beta.example/", "title": "Beta", "raw_content": "Beta only."}


def _urls(url: str) -> list[str]:
    with psycopg.connect(url) as conn:
        return [r[0] for r in conn.execute("select url from mockserp.documents order by url")]


@pytest.fixture
def fresh_store(fresh_db_url):
    s = db.Store(fresh_db_url)
    yield s
    s.close()


def test_ingest_loads_every_document_and_records_index_meta(tmp_path, fresh_store, fresh_db_url):
    write_corpus(tmp_path, [A1, A2])
    meta = ingest(tmp_path, fresh_store, HashEmbedder(model="hash-test", dim=64))

    assert fresh_store.meta() == meta
    assert (meta.embedding_model, meta.dim, meta.n_docs) == ("hash-test", 64, 2)
    assert meta.n_chunks == len(chunk(A1["raw_content"])) + 1 == 3
    assert meta.corpus_sha256 == corpus_sha256(tmp_path)
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", meta.version)
    assert meta.version.endswith(meta.corpus_sha256[:8])
    assert _urls(fresh_db_url) == [A2["url"], A1["url"]]


def test_reingest_replaces_previous_corpus_and_meta(tmp_path, fresh_store, fresh_db_url):
    write_corpus(tmp_path / "a", [A1, A2])
    first = ingest(tmp_path / "a", fresh_store, HashEmbedder())
    write_corpus(tmp_path / "b", [B1])
    second = ingest(tmp_path / "b", fresh_store, HashEmbedder(model="other", dim=32))

    assert _urls(fresh_db_url) == [B1["url"]]
    got = fresh_store.meta()
    assert got == second != first
    assert (got.n_docs, got.n_chunks, got.embedding_model, got.dim) == (1, 1, "other", 32)


def test_failed_replace_keeps_previous_corpus(tmp_path, fresh_store, fresh_db_url):
    write_corpus(tmp_path / "a", [A1, A2])
    before = ingest(tmp_path / "a", fresh_store, HashEmbedder())

    write_corpus(tmp_path / "b", [B1])
    records = build_records(load_corpus(tmp_path / "b"), HashEmbedder())
    with pytest.raises(StoreError):  # same norm_url twice violates the unique constraint
        fresh_store.replace_corpus(records + records, before)

    assert _urls(fresh_db_url) == [A2["url"], A1["url"]]
    assert fresh_store.meta() == before


def test_concurrent_ingests_run_one_after_another(tmp_path, fresh_store, fresh_db_url):
    write_corpus(tmp_path, [B1])
    with psycopg.connect(fresh_db_url) as holder:
        holder.execute("select pg_advisory_xact_lock(%s)", [db.INGEST_LOCK])
        done = threading.Event()
        worker = threading.Thread(
            target=lambda: (ingest(tmp_path, fresh_store, HashEmbedder()), done.set())
        )
        worker.start()
        time.sleep(1.0)
        assert not done.is_set() and fresh_store.meta() is None  # waiting for the lock
        holder.commit()
        worker.join(timeout=30)
    assert done.is_set() and fresh_store.meta().n_docs == 1
