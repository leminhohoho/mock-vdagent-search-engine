# Vector search in Supabase Postgres, Markdown corpus, and deployment — Design

Date: 2026-10-06
Status: Draft, pending written-spec review
Builds on: [`2026-10-01-mock-tavily-search-engine-design.md`](2026-10-01-mock-tavily-search-engine-design.md). Its API contract (§7) and chunker (§5 step 2) stay. This document replaces its corpus format (§4.1), index (§4.2), ingest output (§5 steps 3 and 5), retrieval and ranking (§6), demo corpus and crawler (§8), and the scale note in §10.

## 1. Purpose

Today `mockserp` holds its whole index in server memory and ranks with BM25 plus vectors. This change does three things:

1. **Markdown-only corpus.** The 18 Vietnamese real-estate papers (72 KB) become the corpus, as `data/corpus/*.md` with YAML front matter (`url`, `title`, `published_date`, added 2026-10-06). The following are removed: JSONL corpus input, the English energy demo (`data/corpus/demo.jsonl`), and the `seed` crawler.
2. **Search runs in Supabase Postgres.** Documents, chunks and embeddings live in Postgres tables. `/search` and `/extract` run as SQL functions using pgvector cosine similarity. BM25 is removed, so ranking is vector-only. The server keeps no corpus state: it embeds the query, calls SQL, and formats the Tavily response.
3. **Deployment-ready.** A container image that runs on any container host, configured only through environment variables. An API key is required whenever the server is reachable from outside the machine. Adds a health endpoint, CI, and SQL migrations.

### Success criteria

1. After `supabase db push`, `mockserp ingest` loads all 18 papers into Supabase: 18 rows in `mockserp.documents`, and one row in `mockserp.index_meta` with the embedding model and a version.
2. A container with only environment variables (no data directory) serves `/search` and `/extract` from Supabase through the official `tavily-python` client.
3. A re-run of `ingest` takes effect for every running server immediately on commit, with no restart. Until the commit, servers keep serving the previous corpus.
4. These queries over the papers, with real embeddings, return:
   - `"giải ngân gói tín dụng 145.000 tỷ nhà ở xã hội"` → paper 3 ranked first.
   - `"giá thuê văn phòng hạng A 64,7 USD/m²"` → paper 17 ranked first.
   - `exact_match=true` with the query `"39.000 sản phẩm"`, quotes included → exactly papers 14 and 18, the only papers containing that token sequence.
   - `MOCK_NOW=2026-10-06T00:00:00Z`, `time_range="week"`, `max_results=20` → exactly papers 1, 2, 13, 14, 15, 16.
5. `serve` exits with an error if it would bind a non-loopback host while `MOCK_API_KEY` is unset.
6. CI runs lint, the full test suite against a pgvector Postgres container, and a Docker image build on every push and pull request.

### Non-goals

- **Keyword ranking.** BM25 is removed, and no Postgres full-text ranking replaces it for now. This is an accepted regression: the original spec's success criterion 2 (a query naming an invented code or entity finds its page) now relies on embeddings, or on `exact_match`, which still works as a filter (§6.2). Lexical ranking can come back later as Postgres full-text search with the `simple` configuration, fused in SQL.
- **Approximate nearest-neighbour indexes (HNSW/IVFFlat).** At about 200 chunks, an exact scan is cheaper than maintaining an index and returns exact results. §11 says when to add one.
- **Editing pages in the database.** Pages are authored as Markdown files. `ingest` replaces the database contents wholesale, so rows edited by hand are overwritten.
- Serving several corpora from one database. Choosing a hosting platform, pushing to a container registry, or infrastructure-as-code.
- Cleaning Google Docs export escapes (`\-`, `\[1\]`) out of the papers; `raw_content` stays verbatim.
- Changing the embedding model. `text-embedding-3-small` stays the default.
- Crawling real web pages. `seed`, `data/seeds.txt` and `trafilatura` are removed; git history keeps them.

## 2. Decisions

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Where search runs | **SQL functions in Supabase Postgres (pgvector)** | In-memory index loaded from a Storage bucket: exact BM25, but the server holds and versions the whole index. Vectors in Postgres with BM25 on the server: two sources of truth to keep in sync, and filters written twice. |
| Ranking signals | **Vector similarity only** | Hybrid with Postgres full-text search: deferred (see Non-goals). |
| How the app reaches the database | **Direct Postgres connection (`psycopg` 3) via `DATABASE_URL`** | Supabase Data API (PostgREST RPC with the secret key): no database password needed, but each HTTP call is its own transaction, so ingest can't replace the corpus atomically, and tests would need a full Supabase stack instead of a plain Postgres container. |
| Schema location | **Dedicated schema `mockserp`** | `public`: Supabase exposes it through the Data API by default, so the tables and functions would need RLS policies and revoked grants to stay private. `mockserp` is not exposed, and the `anon` and `authenticated` roles get no grants on it. |
| Migrations | **SQL files in `supabase/migrations/`, applied with `supabase db push --db-url "$DATABASE_URL"`** | A custom migration runner. The Supabase CLI records applied files in `supabase_migrations.schema_migrations`; tests apply the same files directly. |
| Embedding storage type | **Untyped `vector` column, with the dimension recorded in `index_meta`** | `vector(1536)` ties the schema to one model and to the test embedder's dimension. HNSW needs a fixed dimension; when it's needed, an expression index (`(embedding::vector(1536))`) adds one without a table change. |
| Vector values over the wire | **Text literals (`'[0.1,…]'`) cast in SQL** | The `pgvector` Python adapter depends on the connection's `search_path` resolving the `vector` type. That differs between Supabase (extension in schema `extensions`) and a plain test container. |

## 3. Architecture

```mermaid
flowchart LR
  MD[data/corpus/*.md] --> I[mockserp ingest<br/>chunk + embed]
  I -->|one transaction| DB[(Supabase Postgres<br/>schema mockserp:<br/>documents, chunks, index_meta,<br/>search / extract_chunks functions)]
  A[Agents / Tavily SDK] -->|Bearer MOCK_API_KEY| S[mockserp serve<br/>stateless]
  S -->|embed query| E[Embedding API]
  S -->|SELECT mockserp.search(...)| DB
  M[supabase/migrations/*.sql] -->|supabase db push| DB
```

Lifecycle: `supabase db push` (schema, once per migration) → `mockserp ingest` (replace the corpus, whenever pages change) → `mockserp serve` (any number of stateless replicas).

Per `/search` request: one embedding API call, then one SQL function call. The server holds no corpus data.

## 4. Components

| Path | Change |
|---|---|
| `corpus.py` | `load_corpus` reads `*.md` only (§5). `tokenize` moves here from `index.py` and adds NFC normalization (§5.3). |
| `chunker.py`, `embedder.py` | Unchanged. |
| `db.py` (new) | `Store`: wraps a `psycopg_pool.ConnectionPool`. Methods: `meta()`, `replace_corpus(...)`, `search(...)`, `documents_by_norm_url(...)`, `extract_chunks(...)`. This is the only module that writes SQL text. |
| `ingest.py` (new) | Load the corpus → chunk → embed → `Store.replace_corpus` (§7). |
| `search.py` | Keeps `SearchParams`, `SearchError`, `resolve_window`, domain normalization, and quoted-phrase extraction. `search()` embeds the query and calls `Store.search`. The BM25, rank-fusion and candidate code is deleted. |
| `api.py` | `create_app(store, embedder, settings)` replaces `create_app(index, …)`. Adds `GET /healthz`. Maps database errors to the Tavily error shape (§10). |
| `cli.py` | Commands: `ingest`, `serve`. `seed` is removed. `serve` checks startup conditions (§8.1) and the bind/auth guard (§8.3). |
| `config.py` | Settings changes (§9). |
| `index.py`, `bm25.py`, `seed.py` | Deleted, with `tests/test_bm25.py`, `tests/test_index.py`, `tests/test_seed.py`, `data/seeds.txt`, and `data/corpus/demo.jsonl`. |
| `supabase/migrations/20261006000000_mockserp_init.sql` (new) | Schema and functions (§6). |
| `scripts/demo.py` | The English energy queries are replaced with the success criterion 4 queries. |
| Repo | `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`, README, `.env.example`. The papers move from `data/` to `data/corpus/` and are committed. The `data/index/` and `data/.index-*/` entries are removed from `.gitignore`. |

Dependencies:
- Runtime: adds `psycopg[binary]`, `psycopg-pool`, `pyyaml`; removes `trafilatura`.
- `httpx` moves to the dev group (used only by tests now; `openai` brings its own copy at runtime).
- `numpy` stays (embedding normalization).

## 5. Markdown corpus

### 5.1 File format

```markdown
---
url: "https://batdongsan-phantich.example/<ascii-slug-of-title>-<yyyymmdd><NN>.html"
title: "<title from the H1, without ** bold markers>"
published_date: "2026-09-29"
---

# **Original H1** …body…
```

- The opening `---` must be the file's first line. The front matter ends at the next line that is exactly `---`, and it must parse with `yaml.safe_load` to a mapping.
- `url` (absolute http(s), unique across the corpus after normalization) and `title` (non-empty) are required. `published_date` (ISO 8601 or RFC 1123) and `favicon` are optional. Other keys are ignored. An unquoted YAML date that parses to a `date`/`datetime` is converted to an ISO string first. These rules reuse the existing `_parse_row` validation.
- `raw_content` is the text after the closing `---`, with leading blank lines removed. It must not be empty. The H1 stays in `raw_content`.
- A file without front matter is an error; the title is never guessed from the H1.
- Errors raise `CorpusError` with the file name (`paper-RE (3).md: url must be an absolute http(s) URL, got None`). `ingest` exits non-zero without touching the database.

### 5.2 Discovery

`load_corpus(corpus_dir)` reads every `*.md` directly inside `corpus_dir` (no recursion), one page per file, in sorted filename order. It rejects duplicate normalized URLs and ignores all other files. An empty result is an error: `no documents found in <dir>/*.md`.

### 5.3 Tokens (for `exact_match` only)

`tokenize(text)` = `re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())`. With BM25 gone, its only job is `exact_match` (§6.2). NFC means a decomposed Vietnamese query still matches. Stored text and returned text are never normalized.

### 5.4 URL scheme (already applied)

- `.example` is a reserved TLD that never resolves, so an agent fetching the URL outside the mock fails safely.
- The path is the title with accents removed (`đ` → `d`), non-alphanumerics replaced by `-`, then `-<yyyymmdd><NN>.html`. `NN` is the file number, which keeps near-duplicate titles (10/12, 11/13, 14/18) unique.
- New papers follow the scheme; numbers are never reused.

## 6. Database (schema `mockserp`)

### 6.1 Tables

```sql
create schema if not exists extensions;
create extension if not exists vector with schema extensions;
create schema mockserp;

create table mockserp.documents (
  id             bigint generated always as identity primary key,
  url            text not null,              -- as authored, returned to clients
  norm_url       text not null unique,       -- corpus.normalize_url(url); /extract lookup key
  host           text not null,              -- lowercased hostname, leading "www." and trailing "." removed
  title          text not null,
  raw_content    text not null,
  published_date timestamptz,
  favicon        text,
  search_tokens  text not null               -- ' ' || join(tokenize(title || E'\n' || raw_content), ' ') || ' '
);

create table mockserp.chunks (
  id        bigint generated always as identity primary key,
  doc_id    bigint not null references mockserp.documents(id) on delete cascade,
  ord       int not null,                    -- position within the document, from 0
  text      text not null,                   -- verbatim chunk (≤500 chars)
  embedding extensions.vector not null,      -- embedding of title || E'\n' || text, L2-normalized
  unique (doc_id, ord)
);

create table mockserp.index_meta (            -- exactly one row after the first ingest
  singleton       boolean primary key default true check (singleton),
  version         text not null,             -- '<built_at %Y%m%dT%H%M%SZ>-<corpus_sha256[:8]>'
  embedding_model text not null,
  dim             int not null,
  n_docs          int not null,
  n_chunks        int not null,
  corpus_sha256   text not null,             -- over sorted (*.md name, bytes)
  built_at        timestamptz not null
);
```

`host` and `search_tokens` are computed in Python at ingest, so domain and phrase rules have exactly one implementation. No secondary indexes: at this size, every query scans all chunks anyway.

### 6.2 `mockserp.search`

```sql
mockserp.search(
  query_embedding  text,          -- '[f1,f2,…]', L2-normalized
  embedding_model  text,          -- the server's EMBEDDING_MODEL
  max_results      int,           -- 1..20 (0 never reaches SQL)
  chunks_per_source int,          -- 1..3
  include_domains  text[],        -- normalized like `host`; empty = no restriction
  exclude_domains  text[],
  prefer_domains   boolean,       -- include_domains_mode = 'prefer'
  window_start     timestamptz,   -- null = open
  window_end       timestamptz,
  drop_undated     boolean,       -- filter_by_published_date
  phrases          text[],        -- each = join(tokenize(quoted phrase), ' '); empty unless exact_match
  min_score        double precision default null  -- request `min_score` (0..1); null = no threshold
) returns table (url text, title text, raw_content text, published_date timestamptz,
                 favicon text, score double precision, snippets text[])
language plpgsql stable set search_path = mockserp, extensions, pg_catalog
```

Semantics, in order:

1. **Guard.** If `index_meta` has no row, raise SQLSTATE `MS001` with "nothing ingested". If `index_meta.embedding_model <> embedding_model` or `index_meta.dim <> vector_dims(query_embedding::vector)`, raise `MS001` with "index built with <model>/<dim>; query uses <model>/<dim>".
2. **Allowed documents.** These are the same rules as the original spec §6 step 1. A domain `d` matches when `host = d or right(host, length(d) + 1) = '.' || d`.
   - The include filter applies only when `include_domains` is non-empty and `prefer_domains` is false.
   - The exclude filter always applies and wins.
   - Date window: when `window_start` or `window_end` is set, undated documents are dropped only if `drop_undated`; dated documents outside `[window_start, window_end]` are dropped.
   - Phrases: every `p` in `phrases` must satisfy `strpos(search_tokens, ' ' || p || ' ') > 0`.
3. **Chunk similarity.** For every chunk of an allowed document: `sim = 1 - (embedding <=> query)` (cosine). It's an exact scan, with no candidate cutoff.
4. **Roll-up.**
   - Document score = max `sim` over its chunks.
   - Snippets = the texts of its top `chunks_per_source` chunks, ordered by `(sim desc, ord asc)`.
   - **Threshold** (added 2026-10-06, migration `20261006120000_search_min_score.sql`): when `min_score` is set, documents with `greatest(score, 0) < min_score` are dropped here, before ordering and the cut. Snippets are not filtered.
5. **Order and cut.** Sort documents by `(prefer_domains and host matches include_domains) desc, score desc, url asc`, then `limit max_results`.
6. **`score`** = `greatest(document score, 0)`, so it falls in [0, 1]. It's a raw cosine similarity, so absolute values are lower than the old fused scores. Typical values for `text-embedding-3-small` are 0.2–0.7 [INFERENCE].

### 6.3 `mockserp.extract_chunks`

`mockserp.extract_chunks(query_embedding text, embedding_model text, doc_id bigint, k int) returns setof text`. It runs the same guard as §6.2 step 1, then returns the document's top `k` chunk texts ordered by `(sim desc, ord asc)`.

`/extract` without a query never calls it. The server reads `url`, `raw_content` and `favicon` via `select … from mockserp.documents where norm_url = any(%s)`, with URLs normalized in Python. Unknown URLs go to `failed_results` (unchanged contract).

### 6.4 Access

- The tables and functions live in `mockserp`, which Supabase's Data API does not expose. The migration grants nothing to `anon` or `authenticated`.
- The app connects as the database owner (`postgres` / `postgres.<ref>` through the pooler).
- Functions pin `search_path`, so objects in other schemas can't hijack them.

## 7. Ingest (`mockserp ingest`)

1. `load_corpus(CORPUS_DIR)` (§5). Errors exit 1; the database is untouched.
2. Chunk each document (unchanged chunker). Embed `title + "\n" + chunk_text` in batches of `EMBEDDING_BATCH_SIZE`. Compute `host`, `norm_url`, `search_tokens`, `corpus_sha256`, `version`. Embedding errors exit 1; the database is untouched.
3. In **one transaction**:
   1. `select pg_advisory_xact_lock(<constant>)`, so concurrent ingests run one after another.
   2. `delete from mockserp.documents` (chunks cascade).
   3. Insert the documents, then the chunks (`COPY`, embeddings as text literals).
   4. Upsert the single `index_meta` row.
   5. Commit.
4. Print `ingested <n_docs> documents / <n_chunks> chunks (<model>, dim <dim>) version <version>`.

Concurrent readers keep seeing the previous corpus until the commit (Postgres MVCC), then the new one. This replaces the old index directory swap; no server-side reload exists or is needed.

Every run re-embeds the whole corpus, about 200 chunks. Incremental re-embedding by content hash is deferred until the cost matters.

## 8. Serving (`mockserp serve`)

### 8.1 Startup

1. Validate settings (§9).
2. Open the connection pool (`min_size=1`, `max_size=DB_POOL_SIZE`, `prepare_threshold=None` so both Supabase pooler modes work).
3. Read `index_meta`:
   - Schema missing → exit 1, "database not migrated; run `supabase db push --db-url …`".
   - No row → exit 1, "nothing ingested; run `mockserp ingest`".
   - Model differs from `EMBEDDING_MODEL` → exit 1 with both names.
4. Start uvicorn, one worker per container. Scale by adding containers; each holds only a small connection pool.

### 8.2 `GET /healthz`

No auth. It runs `select version, embedding_model, n_docs from mockserp.index_meta`.
- Success: `200 {"status": "ok", "index_version": …, "embedding_model": …, "n_docs": …}`.
- Database unreachable or no row: `503 {"status": "unavailable", "error": "<reason>"}`.

Platforms should use it as a readiness check. As a liveness check, a database outage would restart every container.

### 8.3 Bind/auth guard

`serve` exits 1 with "refusing to serve on <host> without MOCK_API_KEY" when `MOCK_API_KEY` is unset and `HOST` is not `127.0.0.1`, `::1` or `localhost`. This stops a public deployment from spending the embedding budget for anyone who finds it.

### 8.4 Request path

- `/search`: validate (unchanged) → `max_results == 0` returns `[]` without touching the embedding API or the database → embed the query → `Store.search` → format (unchanged envelope, snippets joined with ` [...] `, dates in RFC 1123).
- `/extract`: as §6.3.
- Each request borrows one pooled connection for one statement.

## 9. Configuration

`.env`-over-shell precedence is unchanged; containers have no `.env` and read the environment.

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | unset; required by `ingest` and `serve` | Supabase **session pooler** string (IPv4, keeps session state): Dashboard → Connect → Session pooler, `postgresql://postgres.<ref>:<password>@<pooler-host>:5432/postgres?sslmode=require`. Use the direct connection instead only if your network has IPv6 or the project has the IPv4 add-on. A missing value exits 1 naming the variable. |
| `DB_POOL_SIZE` | `5` | Upper bound on connections per container |
| `CORPUS_DIR` | `data/corpus` | `ingest` only |
| `EMBEDDING_*`, `MOCK_API_KEY`, `MOCK_NOW`, `HOST`, `PORT` | unchanged | |

**Removed:** `INDEX_DIR`.

**Not used by this design:** `SUPABASE_URL` and `SUPABASE_SECRET_KEY`. The app never calls the Supabase HTTP APIs, so they can be deleted from `.env`. The database password in `DATABASE_URL` grants full access to the project database: keep it server-side only.

Deployment environment (platform secret store): `DATABASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` if not the defaults, `MOCK_API_KEY`. The image sets `HOST=0.0.0.0`.

## 10. Failure modes

| Situation | Behavior |
|---|---|
| Corpus error | `ingest` exits 1 naming the file; database untouched |
| Embedding API down during ingest | `ingest` exits 1; database untouched |
| Database error during ingest | Transaction rolls back; exit 1; servers keep serving the previous corpus |
| `serve` startup: no `DATABASE_URL`, unreachable DB, schema missing, nothing ingested, model mismatch | Exit 1 with the reason |
| Database unreachable during a request | 500 `{"detail": {"error": "search backend unavailable: <reason>"}}` |
| `MS001` during a request (re-ingested with another model while servers run) | 500 `{"detail": {"error": "<MS001 message>"}}` until servers are redeployed with the matching `EMBEDDING_MODEL` |
| Embedding API down during a query | Unchanged: 500 `{"detail": {"error": "embedding backend unavailable: …"}}` |
| `HOST` not loopback and no `MOCK_API_KEY` | `serve` exits 1 |

## 11. Risks

- **Vector-only ranking misses exact figures and names.** Near-duplicate papers (10/12, 11/13, 14/18) differ mainly in numbers and wording. Embeddings capture topic, not exact numbers. Success criterion 4 measures this; if it fails, the planned follow-up is Postgres full-text search with `simple`, fused with vectors in `mockserp.search`.
- **Scan cost grows linearly.** Each query reads every allowed chunk's vector: about 6 KB per chunk at 1536 dimensions. Add an HNSW expression index when the corpus passes roughly 50k chunks, or when p95 query time in SQL exceeds about 100 ms [INFERENCE: thresholds not measured]. Filtered HNSW queries need pgvector ≥ 0.8 iterative scans to avoid returning too few results.
- **Network latency.** Each request adds a database round trip. Deploy the server in the Supabase project's region.
- **Pooler connection limits.** Total connections are `replicas × DB_POOL_SIZE`. The shared pooler's limit depends on plan size; keep the total well below it.
- **Migrations through the pooler.** Supabase recommends the direct connection for migrations. This design pushes them through the session-pooler URL, because the direct connection is IPv6-only on the Free plan. Session mode keeps the session state migrations rely on, so this should work [INFERENCE: not yet run against the project]. If `supabase db push` fails through the pooler, run it once from an IPv6-capable network with the direct connection string; the app itself is unaffected.

## 12. Testing

**Unit tests (no database)** — `make test-unit`:
- Markdown loader: a valid file → `Document` with `raw_content` starting at the H1. Each of these is rejected with the file name: missing or unterminated front matter, non-mapping YAML, missing `url`, empty body. An unquoted YAML date is accepted. A duplicate URL across files is rejected. Non-`.md` files are ignored.
- `tokenize`: NFD input yields the same tokens as NFC.
- `resolve_window`, domain normalization, and quoted-phrase extraction (existing cases kept).
- Settings and CLI:
  - Missing `DATABASE_URL` exits 1 naming it.
  - Non-loopback `HOST` without `MOCK_API_KEY` exits 1.

**Database tests** against `pgvector/pgvector:pg17`, with migrations applied by executing `supabase/migrations/*.sql` in order. They run when `TEST_DATABASE_URL` is set and are skipped otherwise. CI sets `MOCKSERP_REQUIRE_DB=1`, which turns that skip into a failure. Locally, `make db-up` starts the container and `make test` runs everything. `HashEmbedder` stays the test embedder.
- **Ingest:**
  - Replaces the previous corpus and `index_meta`.
  - A failure inside the transaction leaves the previous corpus intact.
  - Two concurrent ingests serialize.
- **`search`:**
  - Documents are ordered by their best chunk's similarity, then URL.
  - One result per document.
  - Snippets are the top `chunks_per_source` by similarity.
  - `max_results` cut.
  - Include, exclude, subdomain, `www.`, and prefer domain rules.
  - Date window with undated documents kept or dropped.
  - `exact_match` phrases, including across punctuation (`39.000` vs `39,000`).
  - Model or dimension mismatch raises `MS001`.
- **`extract_chunks`:** top `k` of one document; unknown URLs → `failed_results`.
- **API contract tests** (existing `tavily-python` tests against a live uvicorn server): same assertions, now backed by the test database. Score assertions change from fused values to cosine values.
- **`/healthz`:** 200 with the version; 503 when the database is unreachable.
- **Deleted:** tests that pin BM25, rank fusion, the index file format, JSONL corpora, or the crawler.

**Manual smoke check**, against the real Supabase project with real embeddings:
1. `supabase db push --db-url "$DATABASE_URL"`.
2. `make ingest`.
3. `docker run` the image with only environment variables. Run the success criterion 4 queries through `tavily-python`; check `/healthz`.
4. Edit one paper and re-run `make ingest`. The running container serves the change on the next request, with no restart, and `/healthz` shows the new version.

## 13. Packaging and CI

- **Dockerfile (multi-stage):**
  - Build stage: `ghcr.io/astral-sh/uv` with Python 3.12, `uv sync --frozen --no-dev --no-editable`.
  - Runtime stage: `python:3.12-slim`, venv only, non-root user, `ENV HOST=0.0.0.0 PYTHONUNBUFFERED=1`, `CMD ["mockserp", "serve"]`. `PORT` comes from the platform (default 8000).
  - `.dockerignore` excludes `data/`, `.env`, `.venv`, `tests/`, `docs/`, `supabase/`, and caches.
- **GitHub Actions** (`.github/workflows/ci.yml`), on push and pull request:
  - A `pgvector/pgvector:pg17` service container.
  - `uv sync --frozen` → `ruff check` + `ruff format --check` → `pytest` with `TEST_DATABASE_URL` and `MOCKSERP_REQUIRE_DB=1` → `docker build`.
  - No secrets needed.
- **Makefile:**
  - Removes `seed`.
  - Adds `db-up` and `db-down` (local pgvector container), `migrate` (`supabase db push --db-url "$DATABASE_URL"`), `test-unit`, and `docker`.
  - Keeps `install`, `ingest`, `serve`, `test`, `lint`, `fmt`, `demo`.
