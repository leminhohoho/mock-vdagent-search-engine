"""`mockserp ingest`: Markdown corpus → chunks → embeddings → one database transaction."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from .chunker import chunk
from .corpus import Document, load_corpus, tokenize
from .db import DocumentRecord, IndexMeta, Store
from .embedder import Embedder
from .search import normalize_domain


def search_tokens(doc: Document) -> str:
    """Space-padded token string, so a phrase matches as `' ' || phrase || ' '`."""
    return " " + " ".join(tokenize(f"{doc.title}\n{doc.raw_content}")) + " "


def build_records(docs: list[Document], embedder: Embedder) -> list[DocumentRecord]:
    """Chunk every document and embed `title + "\\n" + chunk` for all chunks."""
    chunked = [chunk(doc.raw_content) for doc in docs]
    embeddings = embedder.embed(
        [f"{doc.title}\n{text}" for doc, texts in zip(docs, chunked, strict=True) for text in texts]
    )
    records, start = [], 0
    for doc, texts in zip(docs, chunked, strict=True):
        records.append(
            DocumentRecord(
                url=doc.url,
                norm_url=doc.norm_url,
                host=normalize_domain(doc.url),
                title=doc.title,
                raw_content=doc.raw_content,
                published_date=doc.published_date,
                favicon=doc.favicon,
                search_tokens=search_tokens(doc),
                chunks=texts,
                embeddings=embeddings[start : start + len(texts)],
            )
        )
        start += len(texts)
    return records


def corpus_sha256(corpus_dir: Path) -> str:
    """Hash over the sorted (*.md file name, bytes) pairs that make up the corpus."""
    h = hashlib.sha256()
    for path in sorted(Path(corpus_dir).glob("*.md")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def prepare(corpus_dir: Path, embedder: Embedder) -> tuple[list[DocumentRecord], IndexMeta]:
    """Load, chunk and embed the corpus without touching the database."""
    docs = load_corpus(corpus_dir)
    records = build_records(docs, embedder)
    sha = corpus_sha256(corpus_dir)
    built_at = datetime.now(UTC).replace(microsecond=0)
    meta = IndexMeta(
        version=f"{built_at:%Y%m%dT%H%M%SZ}-{sha[:8]}",
        embedding_model=embedder.model,
        dim=int(records[0].embeddings.shape[1]),
        n_docs=len(records),
        n_chunks=sum(len(r.chunks) for r in records),
        corpus_sha256=sha,
        built_at=built_at,
    )
    return records, meta


def ingest(corpus_dir: Path, store: Store, embedder: Embedder) -> IndexMeta:
    """Replace the stored corpus; corpus or embedding errors leave the database untouched."""
    records, meta = prepare(corpus_dir, embedder)
    store.replace_corpus(records, meta)
    return meta
