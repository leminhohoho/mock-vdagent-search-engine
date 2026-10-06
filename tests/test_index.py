import json
from datetime import UTC, datetime

import pytest

from mockserp.corpus import CorpusError
from mockserp.embedder import EmbeddingError
from mockserp.index import Index, IndexLoadError, build_index, tokenize

from .conftest import HashEmbedder, RecordingEmbedder, write_corpus

ROWS = [
    {
        "url": "https://a.example/one",
        "title": "Alpha Page",
        "raw_content": "First sentence here.\nSecond line.",
        "published_date": "2024-05-14",
    },
    {"url": "https://b.example/two", "title": "Beta", "raw_content": "Zentrix Q3 revenue rose."},
]


def test_tokenize_lowercases_and_keeps_codes():
    assert tokenize('Zentrix "Q3" revenue: ZX-200!') == ["zentrix", "q3", "revenue", "zx", "200"]


def test_build_embeds_title_prefixed_chunks_and_load_round_trips(tmp_path):
    write_corpus(tmp_path / "corpus", ROWS)
    emb = RecordingEmbedder(dim=16, model="rec-model")

    meta = build_index(tmp_path / "corpus", tmp_path / "index", emb)

    assert emb.calls == [
        ["Alpha Page\nFirst sentence here.\nSecond line.", "Beta\nZentrix Q3 revenue rose."]
    ]
    assert meta["embedding_model"] == "rec-model"
    assert (meta["dim"], meta["n_docs"], meta["n_chunks"]) == (16, 2, 2)

    idx = Index.load(tmp_path / "index")
    assert [d.url for d in idx.docs] == ["https://a.example/one", "https://b.example/two"]
    assert idx.docs[0].published_date == datetime(2024, 5, 14, tzinfo=UTC)
    assert idx.docs[1].published_date is None
    assert [(c.doc_id, c.text) for c in idx.chunks] == [
        (0, "First sentence here.\nSecond line."),
        (1, "Zentrix Q3 revenue rose."),
    ]
    assert idx.embeddings.shape == (2, 16)
    assert idx.meta["embedding_model"] == "rec-model"
    assert idx.doc_id_for_url("HTTPS://B.example/two/") == 1
    assert idx.doc_id_for_url("https://c.example/") is None


def test_bm25_is_rebuilt_on_load_over_title_prefixed_chunks(tmp_path):
    write_corpus(tmp_path / "corpus", ROWS)
    build_index(tmp_path / "corpus", tmp_path / "index", HashEmbedder())
    idx = Index.load(tmp_path / "index")
    scores = idx.bm25.get_scores(tokenize("beta"))  # title-only term
    assert scores[1] > 0 and scores[0] == 0


@pytest.mark.parametrize("failure", ["embedder", "corpus"])
def test_failed_rebuild_leaves_previous_index_untouched(tmp_path, failure):
    write_corpus(tmp_path / "corpus", ROWS)
    build_index(tmp_path / "corpus", tmp_path / "index", HashEmbedder())
    before = (tmp_path / "index" / "meta.json").read_text()

    class Boom(HashEmbedder):
        def embed(self, texts):
            raise EmbeddingError("down")

    if failure == "embedder":
        embedder, err = Boom(), EmbeddingError
    else:
        (tmp_path / "corpus" / "00.md").write_text('---\nurl: "nope"\n---\nB\n')
        embedder, err = HashEmbedder(), CorpusError
    with pytest.raises(err):
        build_index(tmp_path / "corpus", tmp_path / "index", embedder)

    assert (tmp_path / "index" / "meta.json").read_text() == before
    assert {p.name for p in tmp_path.iterdir()} == {"corpus", "index"}  # no temp dirs left behind


def test_load_missing_or_incomplete_index_raises(tmp_path):
    with pytest.raises(IndexLoadError):
        Index.load(tmp_path / "nothing")
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "meta.json").write_text(json.dumps({"embedding_model": "x"}))
    with pytest.raises(IndexLoadError):
        Index.load(tmp_path / "partial")
