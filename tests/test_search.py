import unicodedata
from datetime import UTC, datetime

import numpy as np
import pytest

from mockserp.chunker import chunk
from mockserp.db import IndexMismatch, Store
from mockserp.ingest import ingest
from mockserp.search import SearchError, SearchParams, resolve_window, search

from .conftest import HashEmbedder, para, write_corpus

ZEN = {
    "url": "https://news.zentrix.example/q3",
    "title": "Zentrix quarterly report",
    "raw_content": "\n".join(
        [
            para("Zentrix Q3 revenue rose 12 percent to 4.1 billion."),
            para("The board approved a new headquarters in Lisbon."),
        ]
    ),
    "published_date": "2026-09-20",
}
BAT = {
    "url": "https://www.energy.example/batteries",
    "title": "Grid batteries",
    "raw_content": "\n".join(
        [
            para("Lithium-ion batteries dominate grid storage deployments."),
            para("Battery degradation reduces usable capacity after many cycles."),
            para("Battery storage cost fell sharply as lithium prices dropped."),
            para("Unrelated note about weather in Oslo."),
        ]
    ),
    "published_date": "2025-01-10",
}
BLOG = {
    "url": "https://blog.energy.example/pumped",
    "title": "Pumped hydro explained",
    "raw_content": para("Pumped hydro storage moves water uphill to store energy."),
}
NOT = {
    "url": "https://notenergy.example/flywheels",
    "title": "Flywheels",
    "raw_content": para("Flywheel storage spins a rotor to store kinetic energy."),
    "published_date": "2026-09-30",
}
VN = {
    "url": "https://nha.example/ban-giao",
    "title": "Bàn giao nhà",
    "raw_content": "Chủ đầu tư bàn giao 39.000 sản phẩm trong năm 2026.",
    "published_date": "2026-09-25",
}
TWIN_B = {"url": "https://b.example/", "title": "Same", "raw_content": "Capacitors store charge."}
TWIN_A = {**TWIN_B, "url": "https://a.example/"}
DOCS = [ZEN, BAT, BLOG, NOT, VN, TWIN_B, TWIN_A]
NOW = datetime(2026, 10, 1, tzinfo=UTC)
EMB = HashEmbedder()


@pytest.fixture(scope="module")
def db(store, tmp_path_factory) -> Store:
    corpus = tmp_path_factory.mktemp("search-corpus")
    write_corpus(corpus, DOCS)
    ingest(corpus, store, EMB)
    return store


def expected(query: str, docs=DOCS) -> list[tuple[str, float, list[str]]]:
    """(url, max chunk cosine, chunks by similarity) per document, best first, ties by URL."""
    q = EMB.embed([query])[0]
    out = []
    for doc in docs:
        texts = chunk(doc["raw_content"])
        sims = EMB.embed([f"{doc['title']}\n{t}" for t in texts]) @ q
        order = sorted(range(len(texts)), key=lambda i: (-sims[i], i))
        out.append((doc["url"], float(sims.max()), [texts[i] for i in order]))
    return sorted(out, key=lambda r: (-r[1], r[0]))


def urls(rows) -> list[str]:
    return [r.url for r in rows]


def test_documents_rank_by_best_chunk_cosine_one_result_each(db):
    query = "battery storage cost"
    rows = search(db, EMB, SearchParams(query=query, max_results=20))
    want = expected(query)
    assert urls(rows) == [u for u, _, _ in want]
    assert [r.score for r in rows] == pytest.approx([max(s, 0) for _, s, _ in want], abs=1e-5)
    assert rows[0].url == BAT["url"]


def test_snippets_are_top_chunks_by_similarity(db):
    query = "battery degradation cycles"
    (bat,) = [r for r in search(db, EMB, SearchParams(query=query)) if r.url == BAT["url"]]
    want = next(w for w in expected(query) if w[0] == BAT["url"])[2]
    assert bat.snippets == want[:3]
    assert bat.snippets[0].startswith("Battery degradation reduces usable capacity")

    one = search(db, EMB, SearchParams(query=query, chunks_per_source=1, max_results=1))
    assert [r.snippets for r in one] == [want[:1]]


def test_result_carries_document_fields(db):
    (row,) = search(db, EMB, SearchParams(query="flywheel rotor", max_results=1))
    assert (row.url, row.title, row.raw_content) == (NOT["url"], NOT["title"], NOT["raw_content"])
    assert row.published_date == datetime(2026, 9, 30, tzinfo=UTC) and row.favicon is None


def test_max_results_cuts_the_ranking(db):
    full = urls(search(db, EMB, SearchParams(query="storage", max_results=20)))
    assert urls(search(db, EMB, SearchParams(query="storage", max_results=2))) == full[:2]


def test_zero_max_results_returns_nothing_without_embedding(db):
    class Refuses(HashEmbedder):
        def embed(self, texts):
            raise AssertionError("must not embed")

    assert search(db, Refuses(), SearchParams(query="storage", max_results=0)) == []


def test_equal_scores_break_ties_by_url(db):
    rows = search(db, EMB, SearchParams(query="capacitors store charge", max_results=2))
    assert urls(rows) == [TWIN_A["url"], TWIN_B["url"]]
    assert rows[0].score == rows[1].score


@pytest.mark.parametrize(
    ("include", "exclude", "want"),
    [
        (["energy.example"], [], {BAT["url"], BLOG["url"]}),  # www. and blog. subdomains match
        (["www.energy.example"], [], {BAT["url"], BLOG["url"]}),  # listed www. is dropped too
        (["https://blog.energy.example/"], [], {BLOG["url"]}),  # URL form accepted
        (["ENERGY.example."], [], {BAT["url"], BLOG["url"]}),  # case and trailing dot
        (["energy.example"], ["blog.energy.example"], {BAT["url"]}),  # exclude wins
        (
            [],
            ["energy.example", "zentrix.example", "nha.example", "a.example", "b.example"],
            {NOT["url"]},
        ),  # notenergy.example is not energy.example
    ],
)
def test_domain_filters_use_host_suffix_matching(db, include, exclude, want):
    params = SearchParams(
        query="storage energy", include_domains=include, exclude_domains=exclude, max_results=20
    )
    assert set(urls(search(db, EMB, params))) == want


def test_filters_apply_before_cutoff(db):
    # The pumped-hydro page is a weak match for this query but must still fill the single slot.
    params = SearchParams(
        query="lithium battery cost", include_domains=["blog.energy.example"], max_results=1
    )
    assert urls(search(db, EMB, params)) == [BLOG["url"]]


def test_prefer_mode_ranks_included_domains_first_without_dropping_others(db):
    params = SearchParams(
        query="lithium battery storage",
        include_domains=["notenergy.example"],
        prefer_domains=True,
        max_results=20,
    )
    got = urls(search(db, EMB, params))
    assert got[0] == NOT["url"]
    assert got[1:] == [u for u, _, _ in expected("lithium battery storage") if u != NOT["url"]]


# --- date window -------------------------------------------------------------------------------


def test_resolve_window_time_range_counts_back_from_now():
    assert resolve_window("week", None, None, NOW) == (datetime(2026, 9, 24, tzinfo=UTC), None)
    assert resolve_window("y", None, None, NOW) == (datetime(2025, 10, 1, tzinfo=UTC), None)


def test_resolve_window_dates_are_inclusive_days():
    start, end = resolve_window(None, "2025-01-10", "2026-09-20", NOW)
    assert start == datetime(2025, 1, 10, tzinfo=UTC)
    assert end == datetime(2026, 9, 20, 23, 59, 59, 999999, tzinfo=UTC)


def test_resolve_window_intersects_time_range_and_start_date():
    assert resolve_window("year", "2026-01-01", None, NOW)[0] == datetime(2026, 1, 1, tzinfo=UTC)
    assert resolve_window("year", "2020-01-01", None, NOW)[0] == datetime(2025, 10, 1, tzinfo=UTC)


@pytest.mark.parametrize("bad", ["2025/01/10", "yesterday", "2025-13-01"])
def test_resolve_window_rejects_bad_dates(bad):
    with pytest.raises(SearchError):
        resolve_window(None, bad, None, NOW)


def test_date_window_keeps_undated_docs_unless_asked_to_drop(db):
    start, end = resolve_window("month", None, None, NOW)  # since 2026-09-01
    kept = search(db, EMB, SearchParams(query="storage", start=start, end=end, max_results=20))
    dated = {ZEN["url"], NOT["url"], VN["url"]}
    undated = {BLOG["url"], TWIN_A["url"], TWIN_B["url"]}
    assert set(urls(kept)) == dated | undated

    params = SearchParams(query="storage", start=start, end=end, drop_undated=True, max_results=20)
    assert set(urls(search(db, EMB, params))) == dated


def test_end_date_is_inclusive(db):
    start, end = resolve_window(None, "2025-01-10", "2025-01-10", NOW)
    params = SearchParams(query="storage", start=start, end=end, drop_undated=True)
    assert urls(search(db, EMB, params)) == [BAT["url"]]


# --- exact match -------------------------------------------------------------------------------


def test_exact_match_requires_contiguous_quoted_phrase(db):
    hit = search(db, EMB, SearchParams(query='"usable capacity" storage', exact_match=True))
    assert urls(hit) == [BAT["url"]]
    # Same words, wrong order: no document contains this phrase.
    assert search(db, EMB, SearchParams(query='"capacity usable"', exact_match=True)) == []


@pytest.mark.parametrize(
    "phrase",
    [
        '"39.000 sản phẩm"',
        '"39,000 sản phẩm"',  # punctuation inside numbers is a token boundary either way
        unicodedata.normalize("NFD", '"39.000 SẢN PHẨM"'),  # decomposed, upper case
    ],
)
def test_exact_match_compares_tokens(db, phrase):
    assert urls(search(db, EMB, SearchParams(query=phrase, exact_match=True))) == [VN["url"]]


def test_every_quoted_phrase_must_match(db):
    params = SearchParams(query='"39.000 sản phẩm" "usable capacity"', exact_match=True)
    assert search(db, EMB, params) == []


def test_quotes_filter_only_with_exact_match(db):
    params = SearchParams(query='"capacity usable" storage', max_results=20)
    assert len(search(db, EMB, params)) == len(DOCS)


# --- index guard ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("embedder", "message"),
    [
        (
            HashEmbedder(model="other-model"),
            "index built with hash-test/256; query uses other-model/256",
        ),
        (HashEmbedder(dim=32), "index built with hash-test/256; query uses hash-test/32"),
    ],
)
def test_query_from_another_embedding_space_is_refused(db, embedder, message):
    with pytest.raises(IndexMismatch, match=message):
        search(db, embedder, SearchParams(query="storage"))


def test_search_before_any_ingest_is_refused(fresh_db_url):
    empty = Store(fresh_db_url)
    try:
        with pytest.raises(IndexMismatch, match="nothing ingested"):
            search(empty, EMB, SearchParams(query="storage"))
    finally:
        empty.close()


# --- extract -----------------------------------------------------------------------------------


def test_extract_chunks_returns_top_k_of_one_document(db):
    query = "lithium prices cost"
    doc = db.documents_by_norm_url(["https://www.energy.example/batteries"])[
        "https://www.energy.example/batteries"
    ]
    got = db.extract_chunks(EMB.embed([query])[0], EMB.model, doc.id, k=2)
    want = next(w for w in expected(query) if w[0] == BAT["url"])[2]
    assert got == want[:2]
    assert got[0].startswith("Battery storage cost fell sharply")


def test_extract_chunks_checks_the_embedding_space(db):
    doc = next(iter(db.documents_by_norm_url(["https://notenergy.example/flywheels"]).values()))
    with pytest.raises(IndexMismatch):
        db.extract_chunks(np.ones(8, dtype=np.float32) / np.sqrt(8), EMB.model, doc.id, k=1)
