-- mockserp: documents, chunks and embeddings for the mock Tavily search API.
-- The schema is not exposed through the Supabase Data API; the app connects as the owner.

create schema if not exists extensions;
create extension if not exists vector with schema extensions;
create schema mockserp;
revoke all on schema mockserp from public;

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
  text      text not null,                   -- verbatim chunk (<= 500 chars)
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
