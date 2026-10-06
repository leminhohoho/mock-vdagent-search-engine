import hashlib
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
import pytest
import yaml
from psycopg.conninfo import make_conninfo

from mockserp.db import Store
from mockserp.embedder import l2_normalize

MIGRATIONS = Path(__file__).resolve().parent.parent / "supabase" / "migrations"


PADDING = "filler"


class HashEmbedder:
    """Deterministic offline bag-of-words embedder (test utility); ignores `para` padding."""

    def __init__(self, dim: int = 256, model: str = "hash-test"):
        self.dim = dim
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                if tok == PADDING:
                    continue
                h = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "big")
                out[i, h % self.dim] += 1.0
            if not out[i].any():
                out[i, 0] = 1.0
        return l2_normalize(out)


class RecordingEmbedder(HashEmbedder):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)


def para(text: str) -> str:
    """One sentence unit of ~300 chars (no inner sentence breaks), so two never share a chunk."""
    return text.rstrip(".") + " " + " ".join([PADDING] * ((300 - len(text)) // 7 + 1)) + "."


def write_corpus(directory, rows) -> None:
    """Write each row as `NN.md`: YAML front matter (all keys but raw_content), then the body."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        meta = {k: v for k, v in row.items() if k != "raw_content"}
        front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
        (directory / f"{i:02d}.md").write_text(f"---\n{front}---\n{row['raw_content']}")


@pytest.fixture
def hash_embedder():
    return HashEmbedder()


# --- database ----------------------------------------------------------------------------------
# Database tests need TEST_DATABASE_URL (a pgvector Postgres, e.g. `make db-up`); each session
# works in throwaway databases created next to it. MOCKSERP_REQUIRE_DB=1 turns a skip into a fail.


@pytest.fixture(scope="session")
def admin_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        if os.environ.get("MOCKSERP_REQUIRE_DB") == "1":
            pytest.fail("MOCKSERP_REQUIRE_DB=1 but TEST_DATABASE_URL is not set")
        pytest.skip("TEST_DATABASE_URL not set")
    return url


def apply_migrations(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            conn.execute(path.read_text())


@contextmanager
def scratch_database(admin_url: str, *, migrate: bool) -> Iterator[str]:
    name = f"mockserp_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'create database "{name}"')
    url = make_conninfo(admin_url, dbname=name)
    try:
        if migrate:
            apply_migrations(url)
        yield url
    finally:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(f'drop database if exists "{name}" with (force)')


@pytest.fixture(scope="session")
def db_url(admin_url) -> Iterator[str]:
    """A migrated database shared by the session; modules re-ingest the corpus they need."""
    with scratch_database(admin_url, migrate=True) as url:
        yield url


@pytest.fixture
def fresh_db_url(admin_url) -> Iterator[str]:
    """A migrated database with nothing ingested."""
    with scratch_database(admin_url, migrate=True) as url:
        yield url


@pytest.fixture
def unmigrated_db_url(admin_url) -> Iterator[str]:
    with scratch_database(admin_url, migrate=False) as url:
        yield url


@pytest.fixture(scope="session")
def store(db_url) -> Iterator[Store]:
    s = Store(db_url)
    yield s
    s.close()
