"""Tavily-compatible HTTP API: POST /search, POST /extract, GET /healthz, GET /playground."""

import hashlib
import time
import uuid
from datetime import UTC, datetime
from importlib.resources import files
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .corpus import format_date, normalize_url
from .db import IndexMismatch, Store, StoreError
from .embedder import Embedder, EmbeddingError
from .search import SearchError, SearchParams, resolve_window, search

SNIPPET_JOIN = " [...] "
PLAYGROUND_HTML = (files(__package__) / "playground.html").read_text(encoding="utf-8")


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


def _result_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode()).hexdigest()[:8]


def _favicon(url: str, favicon: str | None) -> str:
    if favicon:
        return favicon
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/favicon.ico"


def _envelope(started: float, **body) -> dict:
    return {
        **body,
        "response_time": round(time.perf_counter() - started, 2),
        "request_id": str(uuid.uuid4()),
    }


def create_app(store: Store, embedder: Embedder, settings: Settings) -> FastAPI:
    app = FastAPI(title="mockserp", description="Mock Tavily search over a Postgres corpus")

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

    @app.exception_handler(StoreError)
    async def _store(_: Request, exc: StoreError) -> JSONResponse:
        message = (
            str(exc) if isinstance(exc, IndexMismatch) else f"search backend unavailable: {exc}"
        )
        return JSONResponse(status_code=500, content={"detail": {"error": message}})

    @app.get("/playground", include_in_schema=False)
    def playground() -> HTMLResponse:
        return HTMLResponse(PLAYGROUND_HTML)

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        try:
            meta = store.meta()
        except StoreError as e:
            return JSONResponse(status_code=503, content={"status": "unavailable", "error": str(e)})
        if meta is None:
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "error": "nothing ingested"}
            )
        return JSONResponse(
            {
                "status": "ok",
                "index_version": meta.version,
                "embedding_model": meta.embedding_model,
                "n_docs": meta.n_docs,
            }
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

        rows = search(
            store,
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
        for row in rows:
            result = {
                "id": _result_id(row.url),
                "title": row.title,
                "url": row.url,
                "content": SNIPPET_JOIN.join(row.snippets),
                "score": row.score,
                "raw_content": row.raw_content if req.include_raw_content else None,
            }
            if show_date:
                result["published_date"] = (
                    format_date(row.published_date) if row.published_date else None
                )
            if req.include_favicon:
                result["favicon"] = _favicon(row.url, row.favicon)
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

        docs = store.documents_by_norm_url(sorted({normalize_url(u) for u in urls}))
        query = req.query if req.query and req.query.strip() else None
        query_embedding = embedder.embed([query])[0] if query and docs else None

        results, failed = [], []
        for url in urls:
            doc = docs.get(normalize_url(url))
            if doc is None:
                failed.append({"url": url, "error": "Failed to fetch url"})
                continue
            if query_embedding is not None:
                raw = SNIPPET_JOIN.join(
                    store.extract_chunks(
                        query_embedding, embedder.model, doc.id, req.chunks_per_source
                    )
                )
            else:
                raw = doc.raw_content
            result = {"url": url, "raw_content": raw, "images": []}
            if req.include_favicon:
                result["favicon"] = _favicon(doc.url, doc.favicon)
            results.append(result)

        body = {"results": results, "failed_results": failed}
        if req.include_usage:
            body["usage"] = {"credits": 1}
        return _envelope(started, **body)

    return app
