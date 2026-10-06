import pytest

from mockserp.cli import main
from mockserp.index import build_index

from .conftest import HashEmbedder, write_corpus

DOC = {"url": "https://a.example/", "title": "A", "raw_content": "Alpha text."}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run the CLI from an empty directory with a clean embedding environment."""
    monkeypatch.chdir(tmp_path)
    for k in ("EMBEDDING_API_KEY", "OPENAI_API_KEY", "EMBEDDING_MODEL", "INDEX_DIR", "CORPUS_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    return tmp_path


def test_serve_refuses_to_start_without_index(workdir, capsys):
    assert main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "data/index" in err and "make ingest" in err


def test_serve_refuses_index_built_with_another_embedding_model(workdir, capsys):
    write_corpus(workdir / "data" / "corpus", [DOC])
    build_index(
        workdir / "data" / "corpus", workdir / "data" / "index", HashEmbedder(model="old-model")
    )
    assert main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "old-model" in err and "text-embedding-3-small" in err


def test_ingest_reports_corpus_error_location(workdir, capsys):
    (workdir / "data" / "corpus").mkdir(parents=True)
    (workdir / "data" / "corpus" / "bad.md").write_text('---\nurl: "https://a.example/"\n---\nB\n')
    assert main(["ingest"]) == 1
    assert "bad.md: title" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["ingest", "serve"])
def test_missing_embedding_key_is_reported(workdir, monkeypatch, capsys, command):
    monkeypatch.delenv("EMBEDDING_API_KEY")
    assert main([command]) == 1
    assert "EMBEDDING_API_KEY" in capsys.readouterr().err
