# Served document URLs — Design

Date: 2026-10-06
Status: Draft, pending written-spec review
Builds on: [`2026-10-06-supabase-vector-search-and-deployment-design.md`](2026-10-06-supabase-vector-search-and-deployment-design.md). This document replaces its URL rules in §5.1 (`url` must be absolute http(s)) and §5.4 (URLs on a `.example` host that never resolves). It also changes the favicon rule in the API contract and adds a public route to §8.

## 1. Purpose

Corpus URLs currently point at `https://batdongsan-phantich.example/...`, a host that never resolves, so following a search result leads nowhere. After this change every document URL points back at the search engine, and opening it shows the full document:

```
old: https://batdongsan-phantich.example/he-so-pir-...-2026092902.html
new: <search engine base>/batdongsan-phantich.example/he-so-pir-...-2026092902.html
```

The corpus files and the database store only the relative part (`batdongsan-phantich.example/he-so-pir-...-2026092902.html`). The base address is added when a response is built.

### Success criteria

1. After `mockserp ingest`, all 18 rows in `mockserp.documents` have a relative `url` such as `batdongsan-phantich.example/he-so-pir-tiem-can-30-lan-bai-toan-an-sinh-va-vien-canh-the-he-thue-nha-vinh-vien-2026092902.html`.
2. With `PUBLIC_BASE_URL` unset, a `/search` request to `http://127.0.0.1:8000` returns `results[].url` values starting with `http://127.0.0.1:8000/batdongsan-phantich.example/`. The same server reached at another address (for example a tailscale IP) returns URLs on that address.
3. With `PUBLIC_BASE_URL=https://serp.example.com`, every returned URL starts with `https://serp.example.com/batdongsan-phantich.example/`, whatever address the request used.
4. Opening a returned URL in a browser shows the document rendered as HTML, with the document title as the page title.
5. `/extract` returns the same document for the returned absolute URL, the same path on a different host, and the bare relative form.
6. `include_domains=["batdongsan-phantich.example"]` still returns results, and `exclude_domains=["batdongsan-phantich.example"]` returns none.
7. With `MOCK_API_KEY` set, document pages open without an `Authorization` header, while `/search` and `/extract` still return 401 without it.

### Non-goals

- Supporting both URL forms. Absolute `url` values in front matter are rejected; there is no compatibility path.
- Serving favicons, images or any other per-site asset.
- Styling the document page beyond a minimal readable shell.
- Trusting `X-Forwarded-Host` or adding proxy configuration. Behind a proxy, set `PUBLIC_BASE_URL` (§6.1).
- A schema migration. The existing columns already hold what is needed.

## 2. Decisions

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Base address | `PUBLIC_BASE_URL` if set, otherwise the request's scheme + `Host` | Request-only: wrong behind a TLS-terminating proxy. Env-only: one more required setting for local runs and tests. |
| Page format | Markdown rendered to minimal HTML | Raw `text/markdown`: unreadable in a browser. Content negotiation: more surface than needed. |
| Page auth | Public, even when `MOCK_API_KEY` is set | Bearer auth: a browser can't send the header, so links wouldn't open. |
| URL identity | `<site>/<path>`; scheme and host of incoming URLs ignored | Requiring the host to equal the current base: URLs from a local run would fail against a deployed server. |
| Favicon without a front-matter value | `null` | Deriving `<base>/favicon.ico`: that path doesn't exist, so the field would always be a dead link. |
| Markdown renderer | `markdown-it-py`, CommonMark with raw HTML disabled, tables enabled | Raw HTML enabled: a custom corpus could inject scripts into the page. |

## 3. URL model

**Canonical form** (stored in front matter and in `mockserp.documents.url`): `<site>/<path>`.

- `<site>` is the first segment. It must be a hostname-like label: letters, digits, `-` and `.`, containing at least one `.`, not starting or ending with `.` or `-`.
- `<path>` is everything after the first `/`. It must be non-empty and use only RFC 3986 unreserved characters, sub-delims, `:`, `@` and `/` (`A–Z a–z 0–9 - . _ ~ ! $ & ' ( ) * + , ; = : @ /`). There is no scheme, leading `/`, query, fragment, `%` or whitespace. Excluding `%` means a stored URL never needs percent-encoding and always matches the decoded request path.
- The existing 18 URLs become their current value with `https://` removed.

**Identity key** (`corpus.normalize_url`, stored in `mockserp.documents.norm_url`): `<site lowercased>/<path without trailing "/">`. Path case is kept. `normalize_url` accepts three input forms and maps them all to the key:

| Input | Key |
|---|---|
| `http://127.0.0.1:8000/batdongsan-phantich.example/x.html` | `batdongsan-phantich.example/x.html` |
| `https://other.host/Batdongsan-Phantich.example/x.html/` | `batdongsan-phantich.example/x.html` |
| `batdongsan-phantich.example/x.html` or `/batdongsan-phantich.example/x.html` | `batdongsan-phantich.example/x.html` |

For an input containing `://`, only the URL path is used. The scheme, host, port, query and fragment are dropped, and the path is percent-decoded. For a bare input, anything from the first `?` or `#` is dropped and a leading `/` is removed. An input with no `/` after the site produces a key that matches no document. An old absolute URL such as `https://batdongsan-phantich.example/x.html` reduces to `x.html`, which matches nothing.

**Absolute URL** (built at response time): `f"{base}/{url}"`, where `base` has no trailing `/` (§6.1).

## 4. Corpus

- All 18 `data/corpus/*.md` files have `url:` rewritten to the canonical form.
- `corpus._parse_row` validates the canonical form (§3) instead of requiring an absolute http(s) URL. The error reads `url must be a relative <site>/<path> URL, got '<value>'`. As before, it is raised as `CorpusError` with the file name, and `ingest` exits non-zero without touching the database.
- `Document.host` is renamed `Document.site` and returns the normalized `<site>` (lowercase, leading `www.` and trailing `.` removed, the same rules as `search.normalize_domain`).
- Duplicate detection is unchanged: it compares `norm_url`.
- §5.4 of the previous spec still governs `<site>` and the slug: `.example` is a reserved TLD and the slug rules are unchanged. Only the scheme is gone.

## 5. Database and ingest

- No migration. `documents.url` holds the canonical relative URL, `norm_url` holds the identity key, and `host` holds the normalized `<site>`. Their column comments in the init migration are already accurate for this ("as authored", "lookup key", "lowercased hostname"), and applied migrations are not edited.
- `ingest` sets `host=doc.site`. Today it calls `normalize_domain(doc.url)`, which would return the whole `site/path` string for a URL without a scheme and break domain filters.
- `mockserp.search` and `host_matches` are unchanged, so `include_domains`/`exclude_domains` keep matching `batdongsan-phantich.example` and its subdomains.
- `Store.documents_by_norm_url` also selects `title`, and `StoredDocument` gains a `title` field for the document page.
- The live Supabase data is updated by running `mockserp ingest` after the corpus rewrite. It replaces the corpus in one transaction, as today.

## 6. Serving

### 6.1 Base address

- New setting `public_base_url: str | None` in `config.Settings`, read from `PUBLIC_BASE_URL`. If it is set, it must be an absolute http(s) URL with a host and no query or fragment, or startup fails with a clear error. Trailing `/` characters are removed. A path is allowed: `https://host/serp` produces links of the form `https://host/serp/<site>/<path>`, for a proxy that strips `/serp` before forwarding.
- `_base(request)` returns `settings.public_base_url`, or otherwise `str(request.base_url).rstrip("/")`. Starlette builds `request.base_url` from the ASGI scheme, the `Host` header and `root_path`.
- Without `PUBLIC_BASE_URL`, a TLS-terminating proxy makes links come out as `http://` unless the proxy's IP is in uvicorn's `FORWARDED_ALLOW_IPS` (default `127.0.0.1,::1`) and it sends `X-Forwarded-Proto`. The proxy must also pass the original `Host` header, because uvicorn ignores `X-Forwarded-Host`. The README states this and recommends `PUBLIC_BASE_URL` behind a proxy.
- A client can send any `Host` header, but that only changes the links in its own response.

### 6.2 `POST /search`

- `results[].url` = `_absolute(_base(request), row.url)`.
- `results[].id` stays `sha1(normalize_url(row.url))[:8]`. It is computed from the stored relative URL, so it doesn't change with the base.
- `results[].favicon` (when `include_favicon`) = the front-matter `favicon`, or `null`.

### 6.3 `POST /extract`

- Each input URL is looked up by `normalize_url(input)` (§3), so all three input forms work.
- `results[].url` repeats the input exactly as sent (unchanged, as Tavily does).
- Unknown inputs, including old absolute URLs, go to `failed_results` with `"Failed to fetch url"` (unchanged).
- `results[].favicon` follows the same rule as `/search`.

### 6.4 `GET /{site}/{path:path}` (document page)

- Registered after every other route and has no `authorize` dependency. The existing paths (`/search`, `/extract`, `/healthz`, `/playground`, `/docs`, `/redoc`, `/openapi.json`) are a single segment, and this route needs at least two, so they can't collide. It is excluded from the OpenAPI schema.
- The handler looks up `normalize_url(f"{site}/{path}")` with `documents_by_norm_url`.
- **Found:** `200`, `text/html; charset=utf-8`:

  ```html
  <!doctype html>
  <html>
  <head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html-escaped title}</title></head>
  <body><article>{rendered raw_content}</article></body>
  </html>
  ```

  Rendering uses `MarkdownIt("commonmark", {"html": False}).enable("table")`, built once at module load. Raw HTML in the Markdown is escaped, not passed through.
- **Unknown:** `404`, `text/html`, a page with the title and body `Not found`.
- **Database errors:** the existing `StoreError` handler applies (`500`, JSON body), the same as for the API endpoints.
- Only `GET` is registered.

### 6.5 Dependencies

- Runtime: adds `markdown-it-py`.

## 7. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PUBLIC_BASE_URL` | unset | Base for document URLs in responses, such as `https://serp.example.com`. Unset uses the address each request was sent to. |

It is added to `.env.example` with a comment explaining the proxy case.

## 8. Failure modes

| Situation | Result |
|---|---|
| Front matter `url` is absolute or malformed | `ingest` fails with `CorpusError` naming the file; database untouched. |
| `PUBLIC_BASE_URL` not an absolute http(s) URL | `serve` exits with a startup error. |
| Behind a proxy without `PUBLIC_BASE_URL` | Links may have the wrong scheme or host (§6.1); documented, not detected. |
| Document page for an unknown path | `404` HTML page. |
| `/extract` with an old absolute URL | Entry in `failed_results`. |

## 9. Testing

- `test_corpus`: the canonical URL is accepted. These are rejected: absolute URLs, a missing path, a site without a dot, `%`, whitespace, `?` and `#`. `normalize_url` maps the three input forms in §3 to the same key, keeps path case, lowercases the site, and drops a trailing `/`. `Document.site` is normalized.
- `test_ingest`: `host` is the normalized site, not the whole URL.
- `test_api`:
  - `/search` URLs use `PUBLIC_BASE_URL` when it is set, and the TestClient base (`http://testserver`) otherwise. `id` is the same under both bases.
  - `/extract` resolves the absolute form on the current host, the absolute form on a foreign host, and the bare form to one document. An old `https://<site>/<path>` URL lands in `failed_results`.
  - `include_favicon` returns the front-matter favicon, or `null` when none is set.
  - The document page returns `200` HTML with the escaped title in `<title>` and the rendered body, and `<script>` in the Markdown comes back escaped. An unknown path returns `404`. With `MOCK_API_KEY` set, the page opens without a token while `/search` still returns 401.
- `test_cli`: an invalid `PUBLIC_BASE_URL` fails startup.
- Existing fixtures move from `https://…` URLs to the canonical form.
- Smoke: run `ingest` against Supabase, then `make serve`. Call `/search` through `tavily-python`, open a returned URL in a browser, and call `/extract` on it.

## 10. Documentation

- README: the URL model, `PUBLIC_BASE_URL`, document pages, and the proxy caveat.
- `.env.example`: `PUBLIC_BASE_URL`.
