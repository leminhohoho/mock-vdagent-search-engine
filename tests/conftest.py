import hashlib
import re

import numpy as np
import pytest
import yaml

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


def para(text: str) -> str:
    """One sentence unit of ~300 chars (no inner sentence breaks), so two never share a chunk."""
    return text.rstrip(".") + " " + " ".join(["filler"] * ((300 - len(text)) // 7 + 1)) + "."


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
