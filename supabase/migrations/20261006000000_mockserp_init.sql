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

-- A host matches a domain d when host = d or host ends with '.' || d. Both sides are normalized
-- in Python (lowercase, no leading "www.", no trailing ".").
create function mockserp.host_matches(host text, domains text[])
returns boolean
language sql immutable
set search_path = pg_catalog
as $$
  select exists (
    select 1 from unnest(domains) as dom(d)
    where host = dom.d or right(host, length(dom.d) + 1) = '.' || dom.d
  )
$$;

-- Raises SQLSTATE MS001 unless an index exists and was built in the query's embedding space.
create function mockserp.check_index(embedding_model text, query_embedding extensions.vector)
returns void
language plpgsql stable
set search_path = mockserp, extensions, pg_catalog
as $$
declare
  m mockserp.index_meta;
begin
  select * into m from mockserp.index_meta;
  if not found then
    raise exception using errcode = 'MS001', message = 'nothing ingested; run `mockserp ingest`';
  end if;
  if m.embedding_model <> check_index.embedding_model
     or m.dim <> vector_dims(check_index.query_embedding) then
    raise exception using errcode = 'MS001', message = format(
      'index built with %s/%s; query uses %s/%s',
      m.embedding_model, m.dim, check_index.embedding_model,
      vector_dims(check_index.query_embedding));
  end if;
end
$$;

-- Vector-only search: filter documents, score every chunk by cosine similarity, roll up to one
-- row per document (score = best chunk, snippets = top chunks_per_source chunks).
create function mockserp.search(
  query_embedding   text,          -- '[f1,f2,...]', L2-normalized
  embedding_model   text,          -- the server's EMBEDDING_MODEL
  max_results       int,
  chunks_per_source int,
  include_domains   text[],        -- normalized like documents.host; empty = no restriction
  exclude_domains   text[],
  prefer_domains    boolean,       -- include_domains_mode = 'prefer'
  window_start      timestamptz,   -- null = open
  window_end        timestamptz,
  drop_undated      boolean,       -- filter_by_published_date
  phrases           text[]         -- each = join(tokenize(quoted phrase), ' ')
) returns table (url text, title text, raw_content text, published_date timestamptz,
                 favicon text, score double precision, snippets text[])
language plpgsql stable
set search_path = mockserp, extensions, pg_catalog
as $$
#variable_conflict use_column
declare
  q vector := search.query_embedding::vector;
  windowed boolean := search.window_start is not null or search.window_end is not null;
begin
  perform mockserp.check_index(search.embedding_model, q);
  return query
  with allowed as (
    select d.id, d.url, d.title, d.raw_content, d.published_date, d.favicon,
           mockserp.host_matches(d.host, search.include_domains) as preferred
    from mockserp.documents d
    where (cardinality(search.include_domains) = 0 or search.prefer_domains
           or mockserp.host_matches(d.host, search.include_domains))
      and not mockserp.host_matches(d.host, search.exclude_domains)
      and (not windowed
           or (d.published_date is null and not search.drop_undated)
           or (d.published_date >= coalesce(search.window_start, '-infinity')
               and d.published_date <= coalesce(search.window_end, 'infinity')))
      and not exists (
        select 1 from unnest(search.phrases) as ph(p)
        where strpos(d.search_tokens, ' ' || ph.p || ' ') = 0
      )
  ),
  scored as (
    select c.doc_id, c.text, 1 - (c.embedding <=> q) as sim,
           row_number() over (partition by c.doc_id
                              order by c.embedding <=> q, c.ord) as rn
    from mockserp.chunks c
    join allowed a on a.id = c.doc_id
  ),
  per_doc as (
    select s.doc_id, max(s.sim) as best,
           array_agg(s.text order by s.rn) filter (where s.rn <= search.chunks_per_source) as top
    from scored s
    group by s.doc_id
  )
  select a.url, a.title, a.raw_content, a.published_date, a.favicon,
         greatest(p.best, 0)::double precision, p.top
  from per_doc p
  join allowed a on a.id = p.doc_id
  order by (search.prefer_domains and a.preferred) desc, p.best desc, a.url collate "C" asc
  limit search.max_results;
end
$$;

-- Top k chunk texts of one document for a query, most similar first.
create function mockserp.extract_chunks(
  query_embedding text, embedding_model text, doc_id bigint, k int
) returns setof text
language plpgsql stable
set search_path = mockserp, extensions, pg_catalog
as $$
#variable_conflict use_column
declare
  q vector := extract_chunks.query_embedding::vector;
begin
  perform mockserp.check_index(extract_chunks.embedding_model, q);
  return query
  select c.text from mockserp.chunks c
  where c.doc_id = extract_chunks.doc_id
  order by c.embedding <=> q, c.ord
  limit extract_chunks.k;
end
$$;

revoke execute on all functions in schema mockserp from public;
