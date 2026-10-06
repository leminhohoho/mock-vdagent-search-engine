"""`mockserp ingest | serve` command line."""

import argparse
import sys

import openai

from .config import Settings
from .corpus import CorpusError
from .db import NotMigrated, Store, StoreError
from .embedder import EmbeddingError, OpenAIEmbedder
from .ingest import prepare

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class StartupError(RuntimeError):
    pass


def _embedder(settings: Settings) -> OpenAIEmbedder:
    if not settings.embedding_api_key:
        raise StartupError("set EMBEDDING_API_KEY (or OPENAI_API_KEY) in the environment or .env")
    client = openai.OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)
    return OpenAIEmbedder(client, settings.embedding_model, settings.embedding_batch_size)


def _database_url(settings: Settings) -> str:
    if not settings.database_url:
        raise StartupError(
            "set DATABASE_URL (Supabase session pooler connection string) in the environment"
            " or .env"
        )
    return settings.database_url


def _store(database_url: str, max_size: int) -> Store:
    try:
        return Store(database_url, max_size=max_size)
    except StoreError as e:
        raise StartupError(f"DATABASE_URL: {e}") from e


def _ingest(settings: Settings) -> None:
    database_url = _database_url(settings)
    embedder = _embedder(settings)
    try:
        records, meta = prepare(settings.corpus_dir, embedder)
    except (CorpusError, EmbeddingError) as e:
        raise StartupError(f"ingest failed: {e}") from e
    store = _store(database_url, max_size=1)
    try:
        store.replace_corpus(records, meta)
    except StoreError as e:
        raise StartupError(f"ingest failed: {e}") from e
    finally:
        store.close()
    print(
        f"ingested {meta.n_docs} documents / {meta.n_chunks} chunks "
        f"({meta.embedding_model}, dim {meta.dim}) version {meta.version}"
    )


def _check_index(store: Store, settings: Settings) -> None:
    try:
        meta = store.meta()
    except NotMigrated as e:
        raise StartupError(
            f'database not migrated; run `supabase db push --db-url "$DATABASE_URL"` ({e})'
        ) from e
    except StoreError as e:
        raise StartupError(f"cannot read the index from the database: {e}") from e
    if meta is None:
        raise StartupError("nothing ingested; run `mockserp ingest`")
    if meta.embedding_model != settings.embedding_model:
        raise StartupError(
            f"index was built with embedding model {meta.embedding_model!r} but EMBEDDING_MODEL "
            f"is {settings.embedding_model!r}; run `mockserp ingest` or change EMBEDDING_MODEL"
        )


def _serve(settings: Settings) -> None:
    database_url = _database_url(settings)
    if not settings.mock_api_key and settings.host not in _LOOPBACK:
        raise StartupError(f"refusing to serve on {settings.host} without MOCK_API_KEY")
    embedder = _embedder(settings)
    store = _store(database_url, max_size=settings.db_pool_size)
    try:
        _check_index(store, settings)

        import uvicorn

        from .api import create_app

        print(f"serving on http://{settings.host}:{settings.port}")
        uvicorn.run(create_app(store, embedder, settings), host=settings.host, port=settings.port)
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mockserp", description="Mock Tavily search engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="load CORPUS_DIR/*.md into the database (replaces the corpus)")
    sub.add_parser("serve", help="serve the Tavily-compatible API")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        {"ingest": _ingest, "serve": _serve}[args.command](settings)
    except StartupError as e:
        print(f"mockserp: {e}", file=sys.stderr)
        return 1
    return 0
