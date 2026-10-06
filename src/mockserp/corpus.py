"""Corpus documents: Markdown loading, validation, URL normalization, dates, and tokens."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

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


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens of NFC-normalized text (used for exact_match phrases)."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())


def _parse_row(row: dict) -> Document:
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
    if isinstance(published, date):  # unquoted YAML date/datetime
        published = published.isoformat()
    try:
        published_dt = parse_date(str(published)) if published else None
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


def _parse_markdown(text: str) -> Document:
    """`---` YAML front matter `---`, then the page body (leading blank lines dropped)."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError("missing front matter: the first line must be '---'")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].rstrip("\r\n") == "---")
    except StopIteration:
        raise ValueError("unterminated front matter: no closing '---' line") from None
    try:
        meta = yaml.safe_load("".join(lines[1:end]))
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML front matter: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError("front matter must be a YAML mapping")
    body = lines[end + 1 :]
    while body and not body[0].strip():
        body.pop(0)
    return _parse_row({**meta, "raw_content": "".join(body)})


def load_corpus(corpus_dir: Path) -> list[Document]:
    """One page per `*.md` file directly inside `corpus_dir`, in sorted filename order."""
    docs: list[Document] = []
    seen: dict[str, str] = {}
    for path in sorted(Path(corpus_dir).glob("*.md")):
        if not path.is_file():
            continue
        try:
            doc = _parse_markdown(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise CorpusError(f"{path.name}: {e}") from e
        if doc.norm_url in seen:
            raise CorpusError(
                f"{path.name}: duplicate url {doc.url!r} (first in {seen[doc.norm_url]})"
            )
        seen[doc.norm_url] = path.name
        docs.append(doc)
    if not docs:
        raise CorpusError(f"no documents found in {corpus_dir}/*.md")
    return docs
