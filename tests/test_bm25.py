import math

import pytest

from mockserp.bm25 import BM25


def test_score_matches_hand_computed_lucene_formula():
    # N=2 docs; "a" appears in both, "b" only in doc 0. k1=1.2, b=0.75.
    bm = BM25([["a", "b"], ["a", "c", "c", "c"]])
    avgdl = 3.0
    idf_a = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    idf_b = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))

    def tf_part(tf, dl):
        return tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))

    scores = bm.get_scores(["a", "b"])
    assert scores[0] == pytest.approx(idf_a * tf_part(1, 2) + idf_b * tf_part(1, 2))
    assert scores[1] == pytest.approx(idf_a * tf_part(1, 4))


def test_term_in_every_document_still_scores_positive():
    # Okapi IDF would be <= 0 here and erase the match.
    bm = BM25([["storage", "x"], ["storage", "y"]])
    assert all(s > 0 for s in bm.get_scores(["storage"]))


def test_unknown_and_repeated_query_terms():
    bm = BM25([["a"], ["b"]])
    assert list(bm.get_scores(["zzz"])) == [0.0, 0.0]
    once, twice = bm.get_scores(["a"]), bm.get_scores(["a", "a"])
    assert twice[0] == pytest.approx(2 * once[0])
