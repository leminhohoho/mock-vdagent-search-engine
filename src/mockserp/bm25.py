"""BM25 with Lucene's always-positive IDF: ln(1 + (N - n + 0.5) / (n + 0.5))."""

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.2, b: float = 0.75):
        n_docs = len(corpus)
        lengths = np.array([len(doc) for doc in corpus], dtype=np.float64)
        avgdl = lengths.mean() if n_docs and lengths.mean() > 0 else 1.0
        norm = k1 * (1 - b + b * lengths / avgdl)

        postings: dict[str, tuple[list[int], list[int]]] = {}
        for doc_id, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(doc_id)
                tfs.append(tf)

        self._n_docs = n_docs
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, (ids, tfs) in postings.items():
            ids_arr = np.array(ids, dtype=np.int64)
            tf_arr = np.array(tfs, dtype=np.float64)
            idf = math.log(1 + (n_docs - len(ids) + 0.5) / (len(ids) + 0.5))
            self._postings[term] = (ids_arr, idf * tf_arr * (k1 + 1) / (tf_arr + norm[ids_arr]))

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self._n_docs, dtype=np.float64)
        for term in query:
            posting = self._postings.get(term)
            if posting is not None:
                np.add.at(scores, posting[0], posting[1])
        return scores
