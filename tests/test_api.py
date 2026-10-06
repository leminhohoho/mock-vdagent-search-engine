"""Contract tests: the real tavily-python client against a live mockserp server."""

import re
import socket
import threading
import time
from datetime import UTC, datetime

import pytest
import uvicorn
from fastapi.testclient import TestClient
from tavily import BadRequestError, InvalidAPIKeyError, TavilyClient

from mockserp.api import create_app
from mockserp.config import Settings
from mockserp.db import Store
from mockserp.embedder import EmbeddingError
from mockserp.ingest import ingest

from .conftest import HashEmbedder, RecordingEmbedder, para, write_corpus

KEY = "tvly-mock-key"
BAT = {
    "url": "https://www.energy.example/batteries",
    "title": "Grid batteries",
    "raw_content": "\n".join(
        [
            para("Lithium-ion batteries dominate grid storage deployments."),
            para("Battery degradation reduces usable capacity after many cycles."),
            para("Battery storage cost fell sharply as lithium prices dropped."),
            para("Battery recycling recovers lithium and cobalt."),
        ]
    ),
    "published_date": "2025-01-10",
    "favicon": "https://www.energy.example/static/icon.png",
}
HYDRO = {
    "url": "https://blog.energy.example/pumped",
    "title": "Pumped hydro explained",
    "raw_content": para("Pumped hydro storage moves water uphill to store energy."),
}
FLY = {
    "url": "https://notenergy.example/flywheels",
    "title": "Flywheels",
    "raw_content": para("Flywheel storage spins a rotor to store kinetic energy."),
    "published_date": "2026-09-30T08:00:00Z",
}


@pytest.fixture(scope="module")
def db(store, tmp_path_factory) -> Store:
    corpus = tmp_path_factory.mktemp("api-corpus")
    write_corpus(corpus, [BAT, HYDRO, FLY])
    ingest(corpus, store, HashEmbedder())
    return store


@pytest.fixture
def unreachable() -> Store:
    """A store whose database never answers (nothing listens on port 1)."""
    s = Store("postgresql://mockserp@127.0.0.1:1/none?connect_timeout=1", timeout=0.5)
    yield s
    s.close()


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, mock_now=datetime(2026, 10, 1, tzinfo=UTC), **kw)


@pytest.fixture(scope="module")
def base_url(db):
    app = create_app(db, HashEmbedder(), _settings(mock_api_key=KEY))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def client(base_url) -> TavilyClient:
    return TavilyClient(api_key=KEY, api_base_url=base_url)


# --- /search ---------------------------------------------------------------------------------


def test_search_default_response_shape(client):
    r = client.search("battery degradation cycles")
    assert {"query", "answer", "images", "results", "response_time", "request_id"} <= r.keys()
    assert (r["query"], r["answer"], r["images"]) == ("battery degradation cycles", None, [])
    assert "usage" not in r
    top = r["results"][0]
    assert top["url"] == BAT["url"] and top["title"] == "Grid batteries"
    assert re.fullmatch(r"[0-9a-f]{8}", top["id"])
    assert top["raw_content"] is None
    assert "published_date" not in top and "favicon" not in top
    parts = top["content"].split(" [...] ")
    assert len(parts) == 3 and parts[0].startswith("Battery degradation reduces usable capacity")
    assert 0 < top["score"] <= 1


def test_result_id_is_stable_across_requests(client):
    a = client.search("flywheel rotor")["results"][0]
    b = client.search("kinetic energy rotor spins")["results"][0]
    assert a["url"] == b["url"] == FLY["url"] and a["id"] == b["id"]


def test_search_optional_fields(client):
    r = client.search(
        "battery storage",
        search_depth="advanced",
        include_raw_content=True,
        include_published_date=True,
        include_favicon=True,
        include_usage=True,
        max_results=3,
    )
    assert r["usage"] == {"credits": 2}
    by_url = {x["url"]: x for x in r["results"]}
    assert by_url[BAT["url"]]["raw_content"] == BAT["raw_content"]
    assert by_url[BAT["url"]]["published_date"] == "Fri, 10 Jan 2025 00:00:00 GMT"
    assert by_url[BAT["url"]]["favicon"] == BAT["favicon"]
    assert by_url[HYDRO["url"]]["published_date"] is None
    assert by_url[HYDRO["url"]]["favicon"] == "https://blog.energy.example/favicon.ico"


def test_news_topic_turns_on_published_date(client):
    r = client.search("flywheel", topic="news", include_usage=True)
    assert r["results"][0]["published_date"] == "Wed, 30 Sep 2026 08:00:00 GMT"
    assert r["usage"] == {"credits": 1}


def test_ultra_fast_returns_single_chunk(client):
    r = client.search("battery degradation cycles", search_depth="ultra-fast")
    assert " [...] " not in r["results"][0]["content"]


def test_chunks_per_source_limits_snippet(client):
    r = client.search("battery degradation cycles", chunks_per_source=2)
    assert len(r["results"][0]["content"].split(" [...] ")) == 2


def test_time_range_counts_back_from_mock_now(client):
    r = client.search("storage energy", time_range="month", filter_by_published_date=True)
    assert [x["url"] for x in r["results"]] == [FLY["url"]]


def test_domain_filters_and_prefer_mode(client):
    r = client.search("storage", include_domains=["energy.example"])
    assert {x["url"] for x in r["results"]} == {BAT["url"], HYDRO["url"]}
    r = client.search(
        "battery lithium", include_domains=["notenergy.example"], include_domains_mode="prefer"
    )
    assert r["results"][0]["url"] == FLY["url"] and len(r["results"]) == 3


def test_unknown_and_ignored_parameters_are_accepted(client):
    r = client.search(
        "battery",
        include_answer="advanced",
        include_images=True,
        country="norway",
        auto_parameters=True,
        some_future_param={"x": 1},
    )
    assert r["answer"] is None and r["images"] == [] and r["results"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_results": 21},
        {"chunks_per_source": 4},
        {"search_depth": "deep"},
        {"time_range": "decade"},
        {"start_date": "2025/01/01"},
        {"include_domains_mode": "prefer"},
    ],
)
def test_invalid_search_parameters_raise_bad_request(client, kwargs):
    with pytest.raises(BadRequestError):
        client.search("battery", **kwargs)


def test_blank_query_is_bad_request(client):
    with pytest.raises(BadRequestError):
        client.search("   ")


def test_wrong_api_key_is_rejected(base_url):
    with pytest.raises(InvalidAPIKeyError):
        TavilyClient(api_key="wrong", api_base_url=base_url).search("battery")


# --- /extract --------------------------------------------------------------------------------


def test_extract_known_and_unknown_urls_in_input_order(client):
    unknown = "https://nowhere.example/page"
    r = client.extract(
        [unknown, "HTTPS://Blog.Energy.example/pumped/", BAT["url"]], include_usage=True
    )
    assert [x["url"] for x in r["results"]] == ["HTTPS://Blog.Energy.example/pumped/", BAT["url"]]
    assert r["results"][0]["raw_content"] == HYDRO["raw_content"]
    assert r["results"][0]["images"] == []
    assert r["failed_results"] == [{"url": unknown, "error": "Failed to fetch url"}]
    assert r["usage"] == {"credits": 1}
    assert {"response_time", "request_id"} <= r.keys()


def test_extract_with_query_returns_top_chunks(client):
    r = client.extract(
        BAT["url"], query="lithium prices cost", chunks_per_source=1, include_favicon=True
    )
    (res,) = r["results"]
    assert res["raw_content"].startswith("Battery storage cost fell sharply")
    assert " [...] " not in res["raw_content"]
    assert res["favicon"] == BAT["favicon"]


def test_extract_more_than_20_urls_is_bad_request(client):
    with pytest.raises(BadRequestError):
        client.extract([f"https://x.example/{i}" for i in range(21)])


# --- non-SDK paths ---------------------------------------------------------------------------


def test_no_configured_key_accepts_requests_without_authorization(db):
    tc = TestClient(create_app(db, HashEmbedder(), _settings()))
    assert tc.post("/search", json={"query": "battery"}).status_code == 200


def test_validation_errors_use_tavily_error_shape(unreachable):
    tc = TestClient(create_app(unreachable, HashEmbedder(), _settings()))
    r = tc.post("/search", json={"query": "battery", "max_results": 99})
    assert r.status_code == 400
    assert "max_results" in r.json()["detail"]["error"]


def test_embedding_backend_failure_is_500(unreachable):
    class Down(HashEmbedder):
        def embed(self, texts):
            raise EmbeddingError("connection refused")

    tc = TestClient(create_app(unreachable, Down(), _settings()))
    r = tc.post("/search", json={"query": "battery"})
    assert r.status_code == 500
    assert r.json() == {"detail": {"error": "embedding backend unavailable: connection refused"}}


def test_unreachable_database_during_a_request_is_500(unreachable):
    tc = TestClient(create_app(unreachable, HashEmbedder(), _settings()))
    for path, body in [("/search", {"query": "battery"}), ("/extract", {"urls": BAT["url"]})]:
        r = tc.post(path, json=body)
        assert r.status_code == 500
        assert r.json()["detail"]["error"].startswith("search backend unavailable: ")


def test_zero_max_results_touches_neither_embedding_api_nor_database(unreachable):
    embedder = RecordingEmbedder()
    tc = TestClient(create_app(unreachable, embedder, _settings()))
    r = tc.post("/search", json={"query": "battery", "max_results": 0})
    assert r.status_code == 200 and r.json()["results"] == []
    assert embedder.calls == []


def test_query_in_another_embedding_space_is_500_with_the_reason(db):
    tc = TestClient(create_app(db, HashEmbedder(model="other"), _settings()))
    for path, body in [
        ("/search", {"query": "battery"}),
        ("/extract", {"urls": BAT["url"], "query": "battery"}),
    ]:
        r = tc.post(path, json=body)
        assert r.status_code == 500
        assert r.json() == {
            "detail": {"error": "index built with hash-test/256; query uses other/256"}
        }


# --- /healthz ------------------------------------------------------------------------------------


def test_healthz_reports_the_index_without_requiring_a_key(db):
    tc = TestClient(create_app(db, HashEmbedder(), _settings(mock_api_key=KEY)))
    r = tc.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "index_version": db.meta().version,
        "embedding_model": "hash-test",
        "n_docs": 3,
    }


def test_healthz_is_503_when_the_database_is_unreachable(unreachable):
    r = TestClient(create_app(unreachable, HashEmbedder(), _settings())).get("/healthz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable" and body["error"]


def test_healthz_is_503_before_the_first_ingest(fresh_db_url):
    empty = Store(fresh_db_url)
    try:
        r = TestClient(create_app(empty, HashEmbedder(), _settings())).get("/healthz")
    finally:
        empty.close()
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable", "error": "nothing ingested"}
