"""`mockserp ingest | serve` command line."""

import argparse
import sys

import openai

from .config import Settings
from .corpus import CorpusError
from .embedder import EmbeddingError, OpenAIEmbedder
from .index import Index, IndexLoadError, build_index


class StartupError(RuntimeError):
    pass


def _embedder(settings: Settings) -> OpenAIEmbedder:
    if not settings.embedding_api_key:
        raise StartupError("set EMBEDDING_API_KEY (or OPENAI_API_KEY) in the environment or .env")
    client = openai.OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)
    return OpenAIEmbedder(client, settings.embedding_model, settings.embedding_batch_size)


def _ingest(settings: Settings) -> None:
    embedder = _embedder(settings)
    try:
        meta = build_index(settings.corpus_dir, settings.index_dir, embedder)
    except (CorpusError, EmbeddingError) as e:
        raise StartupError(f"ingest failed: {e}") from e
    print(
        f"indexed {meta['n_docs']} documents / {meta['n_chunks']} chunks "
        f"({meta['embedding_model']}, dim {meta['dim']}) -> {settings.index_dir}"
    )


def _serve(settings: Settings) -> None:
    embedder = _embedder(settings)
    try:
        index = Index.load(settings.index_dir)
    except IndexLoadError as e:
        raise StartupError(
            f"{e}\nno usable index at {settings.index_dir}; run `make ingest`"
        ) from e
    built_with = index.meta.get("embedding_model")
    if built_with != settings.embedding_model:
        raise StartupError(
            f"index was built with embedding model {built_with!r} but EMBEDDING_MODEL is "
            f"{settings.embedding_model!r}; run `make ingest` or change EMBEDDING_MODEL"
        )

    import uvicorn

    from .api import create_app

    print(f"serving {len(index.docs)} documents on http://{settings.host}:{settings.port}")
    uvicorn.run(create_app(index, embedder, settings), host=settings.host, port=settings.port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mockserp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="build the search index from CORPUS_DIR into INDEX_DIR")
    sub.add_parser("serve", help="serve the Tavily-compatible API")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        {"ingest": _ingest, "serve": _serve}[args.command](settings)
    except StartupError as e:
        print(f"mockserp: {e}", file=sys.stderr)
        return 1
    return 0
