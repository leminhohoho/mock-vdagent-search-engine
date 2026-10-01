import hashlib
import json
import re

import numpy as np
import pytest

from mockserp.embedder import l2_normalize


class HashEmbedder:
    """Deterministic offline bag-of-words embedder (test utility)."""

    def __init__(self, dim: int = 64, model: str = "hash-test"):
        self.dim = dim
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
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


def write_corpus(directory, rows) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def hash_embedder():
    return HashEmbedder()
