"""On-disk search index: build (ingest) and load (serve)."""

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunker import chunk
from .corpus import Document, load_corpus, normalize_url, parse_date
from .embedder import Embedder

_TOKEN = re.compile(r"\w+")


class IndexLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    doc_id: int
    ord: int
    text: str


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def indexed_text(doc: Document, chunk_text: str) -> str:
    return f"{doc.title}\n{chunk_text}"


class Index:
    def __init__(
        self, docs: list[Document], chunks: list[Chunk], embeddings: np.ndarray, meta: dict
    ):
        self.docs = docs
        self.chunks = chunks
        self.embeddings = embeddings
        self.meta = meta
        self.chunk_doc = np.array([c.doc_id for c in chunks], dtype=np.int64)
        self.bm25 = BM25([tokenize(indexed_text(docs[c.doc_id], c.text)) for c in chunks])
        self.doc_tokens = [tokenize(f"{d.title}\n{d.raw_content}") for d in docs]
        self._by_url = {d.norm_url: i for i, d in enumerate(docs)}

    def doc_id_for_url(self, url: str) -> int | None:
        return self._by_url.get(normalize_url(url))

    @classmethod
    def load(cls, index_dir: Path) -> "Index":
        index_dir = Path(index_dir)
        try:
            meta = json.loads((index_dir / "meta.json").read_text())
            docs = [_doc_from_json(json.loads(line)) for line in _lines(index_dir / "docs.jsonl")]
            chunks = [Chunk(**json.loads(line)) for line in _lines(index_dir / "chunks.jsonl")]
            embeddings = np.load(index_dir / "embeddings.npy")
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise IndexLoadError(f"cannot load index from {index_dir}: {e}") from e
        if not docs or not chunks or embeddings.shape[0] != len(chunks):
            raise IndexLoadError(f"index at {index_dir} is inconsistent; rebuild it")
        return cls(docs, chunks, embeddings, meta)


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _doc_to_json(doc: Document) -> dict:
    return {
        "url": doc.url,
        "title": doc.title,
        "raw_content": doc.raw_content,
        "published_date": doc.published_date.isoformat() if doc.published_date else None,
        "favicon": doc.favicon,
    }


def _doc_from_json(row: dict) -> Document:
    published = row.get("published_date")
    return Document(
        url=row["url"],
        title=row["title"],
        raw_content=row["raw_content"],
        published_date=parse_date(published) if published else None,
        favicon=row.get("favicon"),
    )


def _corpus_sha256(corpus_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build_index(corpus_dir: Path, index_dir: Path, embedder: Embedder) -> dict:
    """Rebuild the index from the corpus; the previous index survives any failure."""
    docs = load_corpus(corpus_dir)
    chunks = [
        Chunk(doc_id=i, ord=n, text=text)
        for i, doc in enumerate(docs)
        for n, text in enumerate(chunk(doc.raw_content))
    ]
    embeddings = embedder.embed([indexed_text(docs[c.doc_id], c.text) for c in chunks])
    meta = {
        "embedding_model": embedder.model,
        "dim": int(embeddings.shape[1]),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "corpus_sha256": _corpus_sha256(corpus_dir),
        "built_at": datetime.now(UTC).isoformat(),
    }

    index_dir = Path(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_dir.parent / f".index-{uuid.uuid4().hex}"
    old = index_dir.parent / f".index-old-{uuid.uuid4().hex}"
    try:
        tmp.mkdir()
        (tmp / "docs.jsonl").write_text(
            "".join(json.dumps(_doc_to_json(d), ensure_ascii=False) + "\n" for d in docs),
            encoding="utf-8",
        )
        (tmp / "chunks.jsonl").write_text(
            "".join(json.dumps(c.__dict__, ensure_ascii=False) + "\n" for c in chunks),
            encoding="utf-8",
        )
        np.save(tmp / "embeddings.npy", embeddings.astype(np.float32))
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        if index_dir.exists():
            index_dir.rename(old)
        tmp.rename(index_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
    return meta
