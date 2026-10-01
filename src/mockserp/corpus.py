"""Corpus documents: JSONL loading, validation, URL normalization, and date handling."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}


class CorpusError(ValueError):
    pass


@dataclass(frozen=True)
class Document:
    url: str
    title: str
    raw_content: str
    published_date: datetime | None = None
    favicon: str | None = None

    @property
    def host(self) -> str:
        return urlsplit(self.url).hostname or ""

    @property
    def norm_url(self) -> str:
        return normalize_url(self.url)


def normalize_url(url: str) -> str:
    """Identity form of a URL: lowercase scheme/host, no default port/fragment/trailing slash."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    netloc = host if parts.port in (None, _DEFAULT_PORTS.get(scheme)) else f"{host}:{parts.port}"
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def parse_date(raw: str) -> datetime:
    """Parse ISO 8601 (date or datetime) or RFC 1123; naive values are treated as UTC."""
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        try:
            dt = parsedate_to_datetime(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"unrecognized date {raw!r}") from e
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def format_date(dt: datetime) -> str:
    """RFC 1123 in GMT, as Tavily returns `published_date`."""
    return format_datetime(dt.astimezone(UTC), usegmt=True)


def _parse_row(row: object) -> Document:
    if not isinstance(row, dict):
        raise ValueError("row must be a JSON object")
    url = row.get("url")
    if (
        not isinstance(url, str)
        or urlsplit(url).scheme not in _DEFAULT_PORTS
        or not urlsplit(url).hostname
    ):
        raise ValueError(f"url must be an absolute http(s) URL, got {url!r}")
    for field in ("title", "raw_content"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    published = row.get("published_date")
    try:
        published_dt = parse_date(published) if published else None
    except ValueError as e:
        raise ValueError(f"published_date: {e}") from e
    favicon = row.get("favicon") or None
    return Document(
        url=url,
        title=row["title"].strip(),
        raw_content=row["raw_content"],
        published_date=published_dt,
        favicon=favicon,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    seen: dict[str, str] = {}
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{path.name}:{lineno}"
                try:
                    doc = _parse_row(json.loads(line))
                except json.JSONDecodeError as e:
                    raise CorpusError(f"{where}: invalid JSON: {e}") from e
                except ValueError as e:
                    raise CorpusError(f"{where}: {e}") from e
                if doc.norm_url in seen:
                    raise CorpusError(
                        f"{where}: duplicate url {doc.url!r} (first at {seen[doc.norm_url]})"
                    )
                seen[doc.norm_url] = where
                docs.append(doc)
    if not docs:
        raise CorpusError(f"no documents found in {corpus_dir}/*.jsonl")
    return docs
