import re

import psycopg
import pytest

from mockserp import cli
from mockserp.cli import main
from mockserp.db import Store
from mockserp.embedder import EmbeddingError
from mockserp.ingest import ingest

from .conftest import HashEmbedder, write_corpus

DOC = {"url": "https://a.example/", "title": "A", "raw_content": "Alpha text."}
UNREACHABLE = "postgresql://mockserp@127.0.0.1:1/none?connect_timeout=1"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run the CLI from an empty directory (no .env) with a clean environment."""
    monkeypatch.chdir(tmp_path)
    for k in (
        "EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
        "EMBEDDING_MODEL",
        "CORPUS_DIR",
        "DATABASE_URL",
        "MOCK_API_KEY",
        "HOST",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE)
    return tmp_path


@pytest.mark.parametrize("command", ["ingest", "serve"])
def test_missing_database_url_is_reported(workdir, monkeypatch, capsys, command):
    monkeypatch.delenv("DATABASE_URL")
    assert main([command]) == 1
    assert "DATABASE_URL" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["ingest", "serve"])
def test_missing_embedding_key_is_reported(workdir, monkeypatch, capsys, command):
    monkeypatch.delenv("EMBEDDING_API_KEY")
    assert main([command]) == 1
    assert "EMBEDDING_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["0.0.0.0", "10.1.2.3", "::"])
def test_serve_refuses_public_bind_without_api_key(workdir, monkeypatch, capsys, host):
    monkeypatch.setenv("HOST", host)
    assert main(["serve"]) == 1
    assert f"refusing to serve on {host} without MOCK_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_bind_without_api_key_passes_the_guard(workdir, monkeypatch, capsys, host):
    monkeypatch.setenv("HOST", host)
    monkeypatch.setenv("DATABASE_URL", "not a connection string")
    assert main(["serve"]) == 1  # fails later, at the database
    assert "refusing" not in capsys.readouterr().err


def test_ingest_reports_corpus_error_location(workdir, capsys):
    (workdir / "data" / "corpus").mkdir(parents=True)
    (workdir / "data" / "corpus" / "bad.md").write_text('---\nurl: "https://a.example/"\n---\nB\n')
    assert main(["ingest"]) == 1
    assert "bad.md: title" in capsys.readouterr().err


# --- with a database -----------------------------------------------------------------------------


def _count_documents(url: str) -> int:
    with psycopg.connect(url) as conn:
        return conn.execute("select count(*) from mockserp.documents").fetchone()[0]


def test_ingest_loads_the_corpus_and_reports_it(workdir, monkeypatch, capsys, fresh_db_url):
    monkeypatch.setenv("DATABASE_URL", fresh_db_url)
    monkeypatch.setattr(cli, "_embedder", lambda settings: HashEmbedder())
    write_corpus(workdir / "data" / "corpus", [DOC])

    assert main(["ingest"]) == 0

    out = capsys.readouterr().out.strip()
    assert re.fullmatch(
        r"ingested 1 documents / 1 chunks \(hash-test, dim 256\) version \d{8}T\d{6}Z-[0-9a-f]{8}",
        out,
    )
    assert _count_documents(fresh_db_url) == 1


def test_ingest_embedding_failure_leaves_database_untouched(
    workdir, monkeypatch, capsys, fresh_db_url
):
    class Down(HashEmbedder):
        def embed(self, texts):
            raise EmbeddingError("quota exceeded")

    monkeypatch.setenv("DATABASE_URL", fresh_db_url)
    store = Store(fresh_db_url)
    write_corpus(workdir / "before", [DOC])
    ingest(workdir / "before", store, HashEmbedder())
    monkeypatch.setattr(cli, "_embedder", lambda settings: Down())
    write_corpus(workdir / "data" / "corpus", [DOC, {**DOC, "url": "https://b.example/"}])

    assert main(["ingest"]) == 1

    assert "quota exceeded" in capsys.readouterr().err
    assert store.meta().n_docs == 1 and _count_documents(fresh_db_url) == 1
    store.close()


def test_serve_refuses_unmigrated_database(workdir, monkeypatch, capsys, unmigrated_db_url):
    monkeypatch.setenv("DATABASE_URL", unmigrated_db_url)
    assert main(["serve"]) == 1
    assert "database not migrated; run `supabase db push" in capsys.readouterr().err


def test_serve_refuses_database_with_nothing_ingested(workdir, monkeypatch, capsys, fresh_db_url):
    monkeypatch.setenv("DATABASE_URL", fresh_db_url)
    assert main(["serve"]) == 1
    assert "nothing ingested; run `mockserp ingest`" in capsys.readouterr().err


def test_serve_refuses_index_built_with_another_embedding_model(
    workdir, monkeypatch, capsys, fresh_db_url
):
    monkeypatch.setenv("DATABASE_URL", fresh_db_url)
    store = Store(fresh_db_url)
    write_corpus(workdir / "corpus", [DOC])
    ingest(workdir / "corpus", store, HashEmbedder(model="old-model"))
    store.close()

    assert main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "old-model" in err and "text-embedding-3-small" in err
