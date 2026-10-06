"""Search parameters and their normalization; ranking runs in SQL (`mockserp.search`)."""

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from urllib.parse import urlsplit

from .corpus import tokenize
from .db import SearchRow, Store
from .embedder import Embedder

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


def normalize_domain(raw: str) -> str:
    """Lowercased host of a domain or URL, without a leading "www." or trailing "."."""
    raw = raw.strip().lower()
    if "://" in raw:
        raw = urlsplit(raw).hostname or ""
    return raw.removeprefix("www.").rstrip(".")


def _domains(raw: list[str]) -> list[str]:
    return [d for d in map(normalize_domain, raw) if d]


def quoted_phrases(query: str) -> list[str]:
    """Token-joined `"quoted phrases"` of a query, as matched against `search_tokens`."""
    return [" ".join(tokens) for q in _QUOTED.findall(query) if (tokens := tokenize(q))]


def search(store: Store, embedder: Embedder, p: SearchParams) -> list[SearchRow]:
    if p.max_results <= 0:
        return []
    include = _domains(p.include_domains)
    return store.search(
        embedder.embed([p.query])[0],
        embedder.model,
        max_results=p.max_results,
        chunks_per_source=p.chunks_per_source,
        include_domains=include,
        exclude_domains=_domains(p.exclude_domains),
        prefer_domains=p.prefer_domains and bool(include),
        window_start=p.start,
        window_end=p.end,
        drop_undated=p.drop_undated,
        phrases=quoted_phrases(p.query) if p.exact_match else [],
    )
