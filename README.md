# mockserp

A search engine that speaks the **Tavily API** (`POST /search`, `POST /extract`) and serves results from **your own corpus** rather than the web. It is meant for agents that work over generated data, where real web search returns nothing useful.

Pages are Markdown files. `mockserp ingest` chunks and embeds them into Postgres (Supabase, schema `mockserp`, pgvector). `mockserp serve` is stateless: per request it embeds the query and calls one SQL function, which filters by domain, date and exact phrase, ranks chunks by cosine similarity, and returns one result per page with its best chunks as the snippet.

Design specs: [`2026-10-01-mock-tavily-search-engine-design.md`](docs/superpowers/specs/2026-10-01-mock-tavily-search-engine-design.md) (API contract) and [`2026-10-06-supabase-vector-search-and-deployment-design.md`](docs/superpowers/specs/2026-10-06-supabase-vector-search-and-deployment-design.md) (storage, ranking, deployment).

## Quick start

```bash
make install                 # uv sync
cp .env.example .env         # set DATABASE_URL and EMBEDDING_API_KEY (or OPENAI_API_KEY)
make migrate                 # supabase db push --db-url "$DATABASE_URL" (once per migration)
make ingest                  # chunk + embed data/corpus/*.md into the database
make serve                   # http://127.0.0.1:8000
make demo                    # in another shell: queries through the official Tavily SDK
```

`DATABASE_URL` is the Supabase **session pooler** string (Dashboard → Connect → Session pooler), `postgresql://postgres.<ref>:<password>@<pooler-host>:5432/postgres?sslmode=require`. The direct connection also works if your network has IPv6. The password grants full database access: keep it server-side. `make migrate` needs the [Supabase CLI](https://supabase.com/docs/guides/cli); any Postgres with pgvector works if you apply `supabase/migrations/*.sql` yourself.

Re-running `make ingest` replaces the corpus in one transaction. Running servers serve the new corpus from the next request on, with no restart; until the commit they keep serving the old one.

The repo ships 18 Vietnamese real-estate analysis papers in `data/corpus/`.

## Point your agent at it

Only the base URL changes. Any key works unless `MOCK_API_KEY` is set.

```python
from tavily import TavilyClient

client = TavilyClient(api_key="tvly-mock", api_base_url="http://127.0.0.1:8000")
client.search("giải ngân gói tín dụng nhà ở xã hội", max_results=5)
client.extract(["https://batdongsan-phantich.example/..."], query="lãi suất vay")
```

```python
from langchain_tavily import TavilySearch, TavilyExtract

search = TavilySearch(tavily_api_key="tvly-mock", api_base_url="http://127.0.0.1:8000")
extract = TavilyExtract(tavily_api_key="tvly-mock", api_base_url="http://127.0.0.1:8000")
```

## Your own corpus

One page per `*.md` file directly in `CORPUS_DIR` (default `data/corpus/`), with YAML front matter:

```markdown
---
url: "https://intranet.acme.example/q3-report-2026092001.html"
title: "Acme Q3 report"
published_date: "2026-09-20"
---

# Acme Q3 report
...
```

- `url` (absolute http(s), unique) and `title` are required; `published_date` (ISO 8601 or RFC 1123) and `favicon` are optional; other keys are ignored.
- The body after the closing `---` is returned verbatim as `raw_content`.
- A `.example` host never resolves, so an agent that fetches the URL outside the mock fails safely.
- `ingest` validates every file before touching the database and names the file on error.

## Supported Tavily parameters

| Endpoint | Honored | Accepted and ignored |
|---|---|---|
| `/search` | `query`, `max_results` (0–20), `search_depth` (`ultra-fast` returns 1 chunk; `advanced` costs 2 credits), `chunks_per_source` (1–3), `include_domains`, `exclude_domains`, `include_domains_mode`, `time_range`, `start_date`, `end_date`, `filter_by_published_date`, `include_published_date` (also turned on by `topic="news"`), `exact_match` (every `"quoted phrase"` must occur, compared token by token), `include_raw_content`, `include_favicon`, `include_usage` | `include_answer` (`answer` is always `null`), `include_images` (`images` is always `[]`), `country`, `auto_parameters`, `safe_search`, `language`, timeouts, and any unknown field |
| `/extract` | `urls` (1–20; unknown URLs go to `failed_results`), `query` and `chunks_per_source` (1–5) for reranking the page's chunks, `include_favicon`, `include_usage` | `extract_depth`, `format`, `include_images` |

`score` is the cosine similarity of the page's best chunk to the query, clamped to [0, 1]. Ranking is vector-only; there is no keyword (BM25) signal, so use `exact_match` when an exact figure or name must appear.

Errors use Tavily's `{"detail": {"error": "..."}}` shape: 400 for invalid input, 401 for a wrong key, 500 if the embedding backend or the database is down, or if the database was ingested with a different embedding model. `/crawl`, `/map` and `/research` are not implemented.

`GET /healthz` (no auth) returns `200 {"status": "ok", "index_version", "embedding_model", "n_docs"}`, or 503 when the database is unreachable or empty. Use it as a readiness check, not a liveness check.

`GET /playground` (no auth) is a minimal browser page for trying `/search`: edit the JSON body (starts as `query` + `max_results`), optionally enter the API key, and send. Requests from the page still need the key when `MOCK_API_KEY` is set.

## Configuration

See [`.env.example`](.env.example). Values in `./.env` **override shell environment variables**, so a globally exported `OPENAI_API_KEY` for another provider can't leak in. Shell variables still apply to settings that `.env` doesn't set. Within each source, `EMBEDDING_*` takes precedence over `OPENAI_BASE_URL` / `OPENAI_API_KEY`.

`serve` refuses to start when `DATABASE_URL` is missing, the database is unreachable or not migrated, nothing is ingested, the corpus was embedded with a different `EMBEDDING_MODEL`, or `HOST` is not loopback (`127.0.0.1`, `::1`, `localhost`) while `MOCK_API_KEY` is unset. Set `MOCK_NOW` to pin "now" so `time_range` filters give reproducible results.

## Deployment

```bash
make docker                  # docker build -t mockserp .
docker run -p 8000:8000 \
  -e DATABASE_URL -e EMBEDDING_API_KEY -e EMBEDDING_BASE_URL -e MOCK_API_KEY mockserp
```

The image holds no corpus and no `.env`; it binds `0.0.0.0` (so `MOCK_API_KEY` is required) and listens on `PORT` (default 8000). It also sets `MOCK_NOW=2026-10-06T00:00:00Z`, so `time_range` results don't change from day to day; pass `-e MOCK_NOW=...` to use another date. Run one process per container and scale by adding containers; each holds at most `DB_POOL_SIZE` connections. Deploy close to the Supabase project's region.

## Development

```bash
make db-up        # pgvector/pgvector:pg17 on localhost:54329
make test         # everything; database tests run in throwaway databases on TEST_DATABASE_URL
make test-unit    # database tests skipped
make db-down
make lint
make fmt
```

Database tests are skipped when `TEST_DATABASE_URL` is unset; `MOCKSERP_REQUIRE_DB=1` (set in CI) turns that skip into a failure. The contract tests drive a live server with the real `tavily-python` client. CI (`.github/workflows/ci.yml`) runs lint, the full suite against a pgvector service container, and a Docker build.
