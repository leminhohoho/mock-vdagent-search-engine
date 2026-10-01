"""Hybrid retrieval: filters → BM25 + vector candidates → RRF → one result per document."""

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from urllib.parse import urlsplit

import numpy as np

from .embedder import Embedder
from .index import Index, tokenize

CANDIDATES = 50
RRF_K = 60
_MAX_RRF = 2 / (RRF_K + 1)
_TIME_RANGES = {"day": 1, "d": 1, "week": 7, "w": 7, "month": 30, "m": 30, "year": 365, "y": 365}
_QUOTED = re.compile(r'"([^"]+)"')
_YMD = re.compile(r"\d{4}-\d{2}-\d{2}")


class SearchError(ValueError):
    """Invalid search input (maps to HTTP 400)."""


@dataclass
class SearchParams:
    query: str
    max_results: int = 10
    chunks_per_source: int = 3
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    prefer_domains: bool = False
    start: datetime | None = None
    end: datetime | None = None
    drop_undated: bool = False
    exact_match: bool = False


@dataclass
class Hit:
    doc_id: int
    score: float
    snippets: list[str]


def _parse_day(raw: str, name: str) -> date:
    if not _YMD.fullmatch(raw):
        raise SearchError(f"{name} must be YYYY-MM-DD, got {raw!r}")
    try:
        return date.fromisoformat(raw)
    except ValueError as e:
        raise SearchError(f"{name} is not a valid date: {raw!r}") from e


def resolve_window(
    time_range: str | None, start_date: str | None, end_date: str | None, now: datetime
) -> tuple[datetime | None, datetime | None]:
    """Turn Tavily date parameters into an inclusive [start, end] UTC window."""
    starts = []
    if time_range:
        if time_range not in _TIME_RANGES:
            raise SearchError(f"invalid time_range {time_range!r}")
        starts.append(now - timedelta(days=_TIME_RANGES[time_range]))
    if start_date:
        starts.append(datetime.combine(_parse_day(start_date, "start_date"), time.min, UTC))
    end = None
    if end_date:
        end = datetime.combine(_parse_day(end_date, "end_date"), time.max, UTC)
    return (max(starts) if starts else None), end


def _domain(raw: str) -> str:
    raw = raw.strip().lower()
    if "://" in raw:
        raw = urlsplit(raw).hostname or ""
    return raw.removeprefix("www.").rstrip(".")


def _host_matches(host: str, domains: list[str]) -> bool:
    host = _domain(host)
    return any(host == d or host.endswith("." + d) for d in map(_domain, domains) if d)


def _contains(tokens: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return any(tokens[i : i + n] == phrase for i in range(len(tokens) - n + 1))


def _allowed_docs(index: Index, p: SearchParams) -> np.ndarray:
    phrases = [tokenize(q) for q in _QUOTED.findall(p.query)] if p.exact_match else []
    phrases = [ph for ph in phrases if ph]
    allowed = np.zeros(len(index.docs), dtype=bool)
    for i, doc in enumerate(index.docs):
        if (
            p.include_domains
            and not p.prefer_domains
            and not _host_matches(doc.host, p.include_domains)
        ):
            continue
        if p.exclude_domains and _host_matches(doc.host, p.exclude_domains):
            continue
        if p.start or p.end:
            if doc.published_date is None:
                if p.drop_undated:
                    continue
            elif (p.start and doc.published_date < p.start) or (
                p.end and doc.published_date > p.end
            ):
                continue
        if phrases and not all(_contains(index.doc_tokens[i], ph) for ph in phrases):
            continue
        allowed[i] = True
    return allowed


def _ranked(chunk_ids: np.ndarray, scores: np.ndarray) -> list[tuple[int, int]]:
    """Top CANDIDATES (chunk_id, rank) by score; tied scores share a rank (1 + #strictly higher)."""
    order = np.lexsort((chunk_ids, -scores))[:CANDIDATES]
    out = []
    for pos, i in enumerate(order):
        rank = out[-1][1] if pos and scores[i] == scores[order[pos - 1]] else pos + 1
        out.append((int(chunk_ids[i]), rank))
    return out


def _fuse(index: Index, embedder: Embedder, query: str, chunk_ids: np.ndarray) -> dict[int, float]:
    """Reciprocal-rank fusion of BM25 and vector candidates over the given chunks."""
    rrf: dict[int, float] = {}
    bm25 = index.bm25.get_scores(tokenize(query))[chunk_ids]
    positive = bm25 > 0
    lexical = _ranked(chunk_ids[positive], bm25[positive])
    semantic = _ranked(chunk_ids, index.embeddings[chunk_ids] @ embedder.embed([query])[0])
    for ranked in (lexical, semantic):
        for cid, rank in ranked:
            rrf[cid] = rrf.get(cid, 0.0) + 1 / (RRF_K + rank)
    return rrf


def search(index: Index, embedder: Embedder, p: SearchParams) -> list[Hit]:
    if p.max_results <= 0:
        return []
    allowed = _allowed_docs(index, p)
    chunk_ids = np.flatnonzero(allowed[index.chunk_doc])
    if chunk_ids.size == 0:
        return []
    rrf = _fuse(index, embedder, p.query, chunk_ids)

    per_doc: dict[int, list[tuple[float, int]]] = {}
    for cid, score in rrf.items():
        per_doc.setdefault(index.chunks[cid].doc_id, []).append((score, cid))
    hits = []
    for doc_id, scored in per_doc.items():
        scored.sort(key=lambda sc: (-sc[0], sc[1]))
        snippets = [index.chunks[cid].text for _, cid in scored[: p.chunks_per_source]]
        hits.append(Hit(doc_id=doc_id, score=scored[0][0] / _MAX_RRF, snippets=snippets))

    hits.sort(key=lambda h: (-h.score, index.docs[h.doc_id].url))
    if p.prefer_domains and p.include_domains:
        hits.sort(key=lambda h: not _host_matches(index.docs[h.doc_id].host, p.include_domains))
    return hits[: p.max_results]


def extract_chunks(index: Index, embedder: Embedder, doc_id: int, query: str, k: int) -> list[str]:
    """Top-k chunks of one document for `query`, most relevant first."""
    chunk_ids = np.flatnonzero(index.chunk_doc == doc_id)
    rrf = _fuse(index, embedder, query, chunk_ids)
    ranked = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))
    return [index.chunks[cid].text for cid, _ in ranked[:k]]
