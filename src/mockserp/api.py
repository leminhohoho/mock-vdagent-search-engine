"""Tavily-compatible HTTP API: POST /search and POST /extract."""

import hashlib
import time
import uuid
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .corpus import Document, format_date
from .embedder import Embedder, EmbeddingError
from .index import Index
from .search import SearchError, SearchParams, extract_chunks, resolve_window, search

SNIPPET_JOIN = " [...] "


class _Request(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchRequest(_Request):
    query: str
    search_depth: Literal["basic", "fast", "advanced", "ultra-fast"] = "basic"
    chunks_per_source: int = Field(3, ge=1, le=3)
    max_results: int = Field(10, ge=0, le=20)
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year", "d", "w", "m", "y"] | None = None
    start_date: str | None = None
    end_date: str | None = None
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    include_domains_mode: Literal["restrict", "prefer"] | None = None
    include_published_date: bool = False
    filter_by_published_date: bool = False
    exact_match: bool = False
    include_raw_content: bool | Literal["markdown", "text"] = False
    include_favicon: bool = False
    include_usage: bool = False


class ExtractRequest(_Request):
    urls: str | list[str]
    query: str | None = None
    chunks_per_source: int = Field(3, ge=1, le=5)
    include_favicon: bool = False
    include_usage: bool = False


def _error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": message})


def _result_id(doc: Document) -> str:
    return hashlib.sha1(doc.norm_url.encode()).hexdigest()[:8]


def _favicon(doc: Document) -> str:
    if doc.favicon:
        return doc.favicon
    parts = urlsplit(doc.url)
    return f"{parts.scheme}://{parts.netloc}/favicon.ico"


def _envelope(started: float, **body) -> dict:
    return {
        **body,
        "response_time": round(time.perf_counter() - started, 2),
        "request_id": str(uuid.uuid4()),
    }


def create_app(index: Index, embedder: Embedder, settings: Settings) -> FastAPI:
    app = FastAPI(title="mockserp", description="Mock Tavily search over a local corpus")

    def now() -> datetime:
        if settings.mock_now is None:
            return datetime.now(UTC)
        pinned = settings.mock_now
        return pinned.replace(tzinfo=UTC) if pinned.tzinfo is None else pinned

    def authorize(request: Request) -> None:
        if (
            settings.mock_api_key
            and request.headers.get("authorization") != f"Bearer {settings.mock_api_key}"
        ):
            raise _error(401, "Unauthorized: missing or invalid API key.")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse(status_code=400, content={"detail": {"error": problems}})

    @app.exception_handler(EmbeddingError)
    async def _embedding(_: Request, exc: EmbeddingError) -> JSONResponse:
        return JSONResponse(
            status_code=500, content={"detail": {"error": f"embedding backend unavailable: {exc}"}}
        )

    @app.post("/search", dependencies=[Depends(authorize)])
    def search_endpoint(req: SearchRequest) -> dict:
        started = time.perf_counter()
        if not req.query.strip():
            raise _error(400, "query must not be empty")
        if req.include_domains_mode and not req.include_domains:
            raise _error(400, "include_domains_mode requires include_domains")
        try:
            start, end = resolve_window(req.time_range, req.start_date, req.end_date, now())
        except SearchError as e:
            raise _error(400, str(e)) from e

        hits = search(
            index,
            embedder,
            SearchParams(
                query=req.query,
                max_results=req.max_results,
                chunks_per_source=1 if req.search_depth == "ultra-fast" else req.chunks_per_source,
                include_domains=req.include_domains,
                exclude_domains=req.exclude_domains,
                prefer_domains=req.include_domains_mode == "prefer",
                start=start,
                end=end,
                drop_undated=req.filter_by_published_date,
                exact_match=req.exact_match,
            ),
        )

        show_date = (
            req.include_published_date or req.filter_by_published_date or req.topic == "news"
        )
        results = []
        for hit in hits:
            doc = index.docs[hit.doc_id]
            result = {
                "id": _result_id(doc),
                "title": doc.title,
                "url": doc.url,
                "content": SNIPPET_JOIN.join(hit.snippets),
                "score": hit.score,
                "raw_content": doc.raw_content if req.include_raw_content else None,
            }
            if show_date:
                result["published_date"] = (
                    format_date(doc.published_date) if doc.published_date else None
                )
            if req.include_favicon:
                result["favicon"] = _favicon(doc)
            results.append(result)

        body = {"query": req.query, "answer": None, "images": [], "results": results}
        if req.include_usage:
            body["usage"] = {"credits": 2 if req.search_depth == "advanced" else 1}
        return _envelope(started, **body)

    @app.post("/extract", dependencies=[Depends(authorize)])
    def extract_endpoint(req: ExtractRequest) -> dict:
        started = time.perf_counter()
        urls = [req.urls] if isinstance(req.urls, str) else req.urls
        if not 1 <= len(urls) <= 20:
            raise _error(400, "urls must contain between 1 and 20 URLs")

        results, failed = [], []
        for url in urls:
            doc_id = index.doc_id_for_url(url)
            if doc_id is None:
                failed.append({"url": url, "error": "Failed to fetch url"})
                continue
            doc = index.docs[doc_id]
            if req.query and req.query.strip():
                raw = SNIPPET_JOIN.join(
                    extract_chunks(index, embedder, doc_id, req.query, req.chunks_per_source)
                )
            else:
                raw = doc.raw_content
            result = {"url": url, "raw_content": raw, "images": []}
            if req.include_favicon:
                result["favicon"] = _favicon(doc)
            results.append(result)

        body = {"results": results, "failed_results": failed}
        if req.include_usage:
            body["usage"] = {"credits": 1}
        return _envelope(started, **body)

    return app
