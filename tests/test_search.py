from datetime import UTC, datetime

import numpy as np
import pytest

from mockserp.embedder import l2_normalize
from mockserp.index import Index, build_index
from mockserp.search import SearchError, SearchParams, extract_chunks, resolve_window, search

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
NOW = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> Index:
    root = tmp_path_factory.mktemp("search")
    write_corpus(root / "corpus", [ZEN, BAT, BLOG, NOT])
    build_index(root / "corpus", root / "index", HashEmbedder())
    return Index.load(root / "index")


EMB = HashEmbedder()


def urls(hits, index):
    return [index.docs[h.doc_id].url for h in hits]


def test_returns_one_result_per_document_best_first(index):
    hits = search(index, EMB, SearchParams(query="battery storage cost"))
    assert urls(hits, index)[0] == BAT["url"]
    assert len(set(urls(hits, index))) == len(hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1 for s in scores)


def test_invented_name_found_by_bm25_even_when_embeddings_point_elsewhere(index):
    class DecoyEmbedder:
        """Texts mentioning "Zentrix" get [0, 1]; everything else (incl. the query) gets [1, 0]."""

        model = "decoy"

        def embed(self, texts):
            return l2_normalize(
                np.array([[0.0, 1.0] if "Zentrix" in t else [1.0, 0.0] for t in texts])
            )

    decoy = DecoyEmbedder()
    texts = [f"{index.docs[c.doc_id].title}\n{c.text}" for c in index.chunks]
    rigged = Index(index.docs, index.chunks, decoy.embed(texts), index.meta)

    hits = search(rigged, decoy, SearchParams(query="zentrix q3 revenue", max_results=2))

    # Vector list ranks the 6 non-Zentrix chunks first; Zentrix chunks are 7th/8th.
    # Zentrix best chunk: 1/61 (BM25 rank 1) + 1/67 > 1/61, the best any vector-only chunk gets.
    assert urls(hits, rigged)[0] == ZEN["url"]
    assert hits[0].score == pytest.approx((1 / 61 + 1 / 67) / (2 / 61))


def test_score_is_one_when_top_in_both_lists(index):
    hits = search(
        index, EMB, SearchParams(query="Flywheels Flywheel storage spins a rotor kinetic")
    )
    assert urls(hits, index)[0] == NOT["url"]
    assert hits[0].score == pytest.approx(1.0)


def test_max_results_limits_and_zero_returns_nothing(index):
    assert len(search(index, EMB, SearchParams(query="storage", max_results=2))) == 2
    assert search(index, EMB, SearchParams(query="storage", max_results=0)) == []


def test_snippets_are_top_chunks_of_the_document_in_relevance_order(index):
    hits = search(index, EMB, SearchParams(query="battery degradation cycles", chunks_per_source=2))
    top = hits[0]
    assert index.docs[top.doc_id].url == BAT["url"]
    assert len(top.snippets) == 2
    assert top.snippets[0].startswith("Battery degradation reduces usable capacity")
    assert all(len(s) <= 500 for s in top.snippets)

    one = search(index, EMB, SearchParams(query="battery degradation cycles", chunks_per_source=1))
    assert one[0].snippets == [top.snippets[0]]


@pytest.mark.parametrize(
    ("include", "exclude", "want"),
    [
        (["energy.example"], [], {BAT["url"], BLOG["url"]}),  # www. and blog. subdomains match
        (["www.energy.example"], [], {BAT["url"], BLOG["url"]}),  # listed www. is dropped too
        (["https://blog.energy.example/"], [], {BLOG["url"]}),  # URL form accepted
        (["energy.example"], ["blog.energy.example"], {BAT["url"]}),  # exclude wins
        (
            [],
            ["energy.example", "zentrix.example"],
            {NOT["url"]},
        ),  # notenergy.example is not energy.example
    ],
)
def test_domain_filters_use_host_suffix_matching(index, include, exclude, want):
    params = SearchParams(query="storage energy", include_domains=include, exclude_domains=exclude)
    assert set(urls(search(index, EMB, params), index)) == want


def test_filters_apply_before_cutoff(index):
    # The pumped-hydro page is a weak match for this query but must still fill the single slot.
    params = SearchParams(
        query="lithium battery cost", include_domains=["blog.energy.example"], max_results=1
    )
    assert urls(search(index, EMB, params), index) == [BLOG["url"]]


def test_prefer_mode_ranks_included_domains_first_without_dropping_others(index):
    params = SearchParams(
        query="lithium battery storage", include_domains=["notenergy.example"], prefer_domains=True
    )
    got = urls(search(index, EMB, params), index)
    assert got[0] == NOT["url"]
    assert BAT["url"] in got[1:]


def test_ties_break_by_url(tmp_path):
    same = para("Identical body text about capacitors.")
    write_corpus(
        tmp_path / "c",
        [
            {"url": "https://b.example/", "title": "Same", "raw_content": same},
            {"url": "https://a.example/", "title": "Same", "raw_content": same},
        ],
    )
    build_index(tmp_path / "c", tmp_path / "i", HashEmbedder())
    idx = Index.load(tmp_path / "i")
    assert urls(search(idx, EMB, SearchParams(query="capacitors")), idx) == [
        "https://a.example/",
        "https://b.example/",
    ]


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


def test_date_window_keeps_undated_docs_unless_asked_to_drop(index):
    start, end = resolve_window("month", None, None, NOW)  # since 2026-09-01
    kept = search(index, EMB, SearchParams(query="storage revenue", start=start, end=end))
    assert set(urls(kept, index)) == {ZEN["url"], NOT["url"], BLOG["url"]}

    dropped = search(
        index, EMB, SearchParams(query="storage revenue", start=start, end=end, drop_undated=True)
    )
    assert set(urls(dropped, index)) == {ZEN["url"], NOT["url"]}


def test_end_date_is_inclusive(index):
    start, end = resolve_window(None, "2025-01-10", "2025-01-10", NOW)
    params = SearchParams(query="storage", start=start, end=end, drop_undated=True)
    assert urls(search(index, EMB, params), index) == [BAT["url"]]


# --- exact match -------------------------------------------------------------------------------


def test_exact_match_requires_contiguous_quoted_phrase(index):
    hit = search(index, EMB, SearchParams(query='"usable capacity" storage', exact_match=True))
    assert urls(hit, index) == [BAT["url"]]
    # Same words, wrong order: no document contains this phrase.
    assert search(index, EMB, SearchParams(query='"capacity usable"', exact_match=True)) == []


def test_exact_match_without_quotes_does_not_filter(index):
    params = SearchParams(query="storage", exact_match=True)
    assert len(search(index, EMB, params)) == 4


# --- extract -----------------------------------------------------------------------------------


def test_extract_chunks_reranks_one_documents_chunks_by_query(index):
    doc_id = index.doc_id_for_url(BAT["url"])
    got = extract_chunks(index, EMB, doc_id, "lithium prices cost", k=2)
    assert len(got) == 2
    assert got[0].startswith("Battery storage cost fell sharply")
    assert all(c.startswith(("Battery", "Lithium")) for c in got)
