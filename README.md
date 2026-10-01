# mockserp

A local search engine that speaks the **Tavily API** (`POST /search`, `POST /extract`) and serves results from **your own corpus** rather than the web. It is meant for agents that work over generated data, where real web search returns nothing useful.

Inside, it works like a conventional web search engine: BM25 and embedding retrieval merged with reciprocal rank fusion, one result per page, snippets taken from the parts of the page that match the query, domain, date and exact-phrase filters, and a separate page-fetch step (`/extract`).

Design spec: [`docs/superpowers/specs/2026-10-01-mock-tavily-search-engine-design.md`](docs/superpowers/specs/2026-10-01-mock-tavily-search-engine-design.md).

## Quick start

```bash
make install                 # uv sync
cp .env.example .env         # set EMBEDDING_API_KEY (or OPENAI_API_KEY)
make ingest                  # chunk + embed data/corpus/*.jsonl -> data/index/
make serve                   # http://127.0.0.1:8000
make demo                    # in another shell: queries through the official Tavily SDK
```

The repo ships a crawled demo corpus about grid-scale energy storage (`data/corpus/demo.jsonl`, built from `data/seeds.txt`). `make seed` re-crawls it. The crawler checks `robots.txt`, waits 1 second between requests and does not follow links.

## Point your agent at it

Only the base URL changes. Any key works unless `MOCK_API_KEY` is set.

```python
from tavily import TavilyClient

client = TavilyClient(api_key="tvly-mock", api_base_url="http://127.0.0.1:8000")
client.search("pumped hydro round-trip efficiency", max_results=5)
client.extract(["https://en.wikipedia.org/wiki/Flow_battery"], query="vanadium electrolyte")
```

```python
from langchain_tavily import TavilySearch, TavilyExtract

search = TavilySearch(tavily_api_key="tvly-mock", api_base_url="http://127.0.0.1:8000")
extract = TavilyExtract(tavily_api_key="tvly-mock", api_base_url="http://127.0.0.1:8000")
```

## Your own corpus

Put one or more `*.jsonl` files in `CORPUS_DIR` (default `data/corpus/`), one page per line, in the same shape as Tavily results:

```json
{"url": "https://intranet.acme.example/q3", "title": "Acme Q3 report", "raw_content": "# Acme Q3\n...", "published_date": "2026-09-20", "favicon": null}
```

`url`, `title` and `raw_content` are required. `url` must be unique and is the page's identity for `/extract`. Run `make ingest` again after any change.

## Supported Tavily parameters

| Endpoint | Honored | Accepted and ignored |
|---|---|---|
| `/search` | `query`, `max_results` (0–20), `search_depth` (`ultra-fast` returns 1 chunk; `advanced` costs 2 credits), `chunks_per_source` (1–3), `include_domains`, `exclude_domains`, `include_domains_mode`, `time_range`, `start_date`, `end_date`, `filter_by_published_date`, `include_published_date` (also turned on by `topic="news"`), `exact_match`, `include_raw_content`, `include_favicon`, `include_usage` | `include_answer` (`answer` is always `null`), `include_images` (`images` is always `[]`), `country`, `auto_parameters`, `safe_search`, `language`, timeouts, and any unknown field |
| `/extract` | `urls` (1–20; unknown URLs go to `failed_results`), `query` and `chunks_per_source` (1–5) for reranking the page's chunks, `include_favicon`, `include_usage` | `extract_depth`, `format`, `include_images` |

Errors use Tavily's `{"detail": {"error": "..."}}` shape: 400 for invalid input, 401 for a wrong key, 500 if the embedding backend is down. `/crawl`, `/map` and `/research` are not implemented.

## Configuration

See [`.env.example`](.env.example). `EMBEDDING_*` settings take precedence over `OPENAI_BASE_URL` / `OPENAI_API_KEY`. `serve` refuses to start if the index was built with a different `EMBEDDING_MODEL`. Set `MOCK_NOW` to pin "now" so `time_range` filters give reproducible results.

## Development

```bash
make test    # offline; the contract tests drive a live server with the real tavily-python client
make lint
make fmt
```
