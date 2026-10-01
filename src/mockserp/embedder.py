"""Text embedding behind a small protocol; OpenAI-compatible implementation."""

from typing import Protocol

import numpy as np
import openai


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 [len(texts), dim], each row L2-normalized."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class OpenAIEmbedder:
    def __init__(self, client: openai.OpenAI, model: str, batch_size: int = 128):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
            except openai.OpenAIError as e:
                raise EmbeddingError(str(e)) from e
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return l2_normalize(np.array(rows))
