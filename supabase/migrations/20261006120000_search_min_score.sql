-- /search: optional similarity threshold (min_score) applied inside mockserp.search.
-- The argument list changes, so the function is dropped and recreated.

drop function mockserp.search(text, text, int, int, text[], text[], boolean, timestamptz, timestamptz, boolean, text[]);

-- Vector-only search: filter documents, score every chunk by cosine similarity, roll up to one
-- row per document (score = best chunk, snippets = top chunks_per_source chunks), and drop
-- documents whose clamped score is below min_score before ordering and the max_results cut.
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
  phrases           text[],        -- each = join(tokenize(quoted phrase), ' ')
  min_score         double precision default null  -- null = no similarity threshold
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
    having greatest(max(s.sim), 0) >= coalesce(search.min_score, 0)
  )
  select a.url, a.title, a.raw_content, a.published_date, a.favicon,
         greatest(p.best, 0)::double precision, p.top
  from per_doc p
  join allowed a on a.id = p.doc_id
  order by (search.prefer_domains and a.preferred) desc, p.best desc, a.url collate "C" asc
  limit search.max_results;
end
$$;

revoke execute on function mockserp.search(
  text, text, int, int, text[], text[], boolean, timestamptz, timestamptz, boolean, text[], double precision
) from public;
