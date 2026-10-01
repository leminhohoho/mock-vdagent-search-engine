import json

import httpx
import numpy as np
import openai
import pytest

from mockserp.config import Settings
from mockserp.embedder import EmbeddingError, OpenAIEmbedder


def _client(handler) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="k",
        base_url="http://emb.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_embeds_in_batches_preserving_input_order_and_normalizing():
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "emb-model"
        batches.append(body["input"])
        # Vector [len(text), 0, 1]; return data shuffled to prove we reorder by `index`.
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(t)), 0.0, 1.0]}
            for i, t in enumerate(body["input"])
        ][::-1]
        return httpx.Response(200, json={"object": "list", "data": data, "model": "emb-model",
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    emb = OpenAIEmbedder(_client(handler), model="emb-model", batch_size=2)
    out = emb.embed(["a", "bbb", "cccc"])

    assert batches == [["a", "bbb"], ["cccc"]]
    assert out.dtype == np.float32
    want = np.array([[1, 0, 1], [3, 0, 1], [4, 0, 1]], dtype=np.float64)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    np.testing.assert_allclose(out, want, rtol=1e-6)


def test_api_failure_raises_embedding_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    emb = OpenAIEmbedder(_client(handler), model="m", batch_size=8)
    with pytest.raises(EmbeddingError):
        emb.embed(["x"])


def test_settings_prefer_embedding_vars_and_fall_back_to_openai_vars(monkeypatch):
    for k in ("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.local/v1")
    s = Settings(_env_file=None)
    assert (s.embedding_api_key, s.embedding_base_url) == ("openai-key", "http://openai.local/v1")

    monkeypatch.setenv("EMBEDDING_API_KEY", "emb-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://emb.local/v1")
    s = Settings(_env_file=None)
    assert (s.embedding_api_key, s.embedding_base_url) == ("emb-key", "http://emb.local/v1")


def test_settings_default_base_url_is_openai(monkeypatch):
    for k in ("EMBEDDING_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    assert Settings(_env_file=None).embedding_base_url == "https://api.openai.com/v1"
