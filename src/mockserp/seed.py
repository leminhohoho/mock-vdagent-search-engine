"""Build a demo corpus by crawling a list of seed URLs (no link following)."""

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from .corpus import normalize_url

USER_AGENT = "mockserp-seed/0.1 (+demo corpus builder)"
MIN_CHARS = 1000
TIMEOUT_S = 20.0
_INLINE_COMMENT = re.compile(r"\s+#.*$")


def parse_seeds(text: str) -> list[str]:
    seeds = []
    for line in text.splitlines():
        line = _INLINE_COMMENT.sub("", line).strip()
        if line and not line.startswith("#"):
            seeds.append(line)
    return seeds


def _page_row(url: str, html: str) -> dict | str:
    """Corpus row for a fetched page, or a skip reason."""
    markdown = trafilatura.extract(
        html, url=url, output_format="markdown", include_comments=False, include_tables=True
    )
    if not markdown or len(markdown) < MIN_CHARS:
        return f"too short ({len(markdown or '')} chars of main content)"
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta else None) or markdown.splitlines()[0].lstrip("# ").strip()
    row = {"url": url, "title": title, "raw_content": markdown}
    if meta and meta.date:
        row["published_date"] = meta.date
    return row


def crawl(
    seeds: list[str],
    client: httpx.Client,
    delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fetch each seed once; return (corpus rows, [(url, skip reason)])."""
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    kept: dict[str, str] = {}
    robots: dict[str, RobotFileParser] = {}
    requests_made = 0

    def get(url: str) -> httpx.Response:
        nonlocal requests_made
        if requests_made:
            sleep(delay)
        requests_made += 1
        return client.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)

    def allowed(url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in robots:
            parser = RobotFileParser()
            try:
                resp = get(f"{origin}/robots.txt")
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser.allow_all = True
            except httpx.HTTPError:
                parser.allow_all = True
            robots[origin] = parser
        return robots[origin].can_fetch(USER_AGENT, url)

    for url in seeds:
        if not allowed(url):
            skipped.append((url, "robots.txt disallows"))
            continue
        try:
            resp = get(url)
        except httpx.HTTPError as e:
            skipped.append((url, f"fetch error: {e}"))
            continue
        if resp.status_code != 200:
            skipped.append((url, f"HTTP {resp.status_code}"))
            continue
        if "html" not in resp.headers.get("content-type", ""):
            skipped.append((url, f"not HTML ({resp.headers.get('content-type')})"))
            continue
        final_url = str(resp.url)
        if normalize_url(final_url) in kept:
            skipped.append((url, f"duplicate of {kept[normalize_url(final_url)]}"))
            continue
        row = _page_row(final_url, resp.text)
        if isinstance(row, str):
            skipped.append((url, row))
            continue
        kept[normalize_url(final_url)] = final_url
        rows.append(row)
    return rows, skipped


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def run(seeds_file: Path, out_file: Path, delay: float = 1.0) -> tuple[int, int]:
    seeds = parse_seeds(seeds_file.read_text(encoding="utf-8"))
    with httpx.Client(follow_redirects=True) as client:
        rows, skipped = crawl(seeds, client, delay=delay)
    write_rows(rows, out_file)
    for url, reason in skipped:
        print(f"skipped {url}: {reason}")
    print(f"kept {len(rows)} / {len(seeds)} pages -> {out_file}")
    return len(rows), len(skipped)
