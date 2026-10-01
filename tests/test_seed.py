import json

import httpx

from mockserp.corpus import load_corpus
from mockserp.seed import USER_AGENT, crawl, parse_seeds, write_rows

LONG = "Pumped storage moves water uphill when power is cheap and releases it later. " * 20


def page(title: str, body: str = LONG, date: str | None = "2024-05-14T10:00:00Z") -> str:
    meta = f'<meta property="article:published_time" content="{date}">' if date else ""
    return (
        f"<html><head><title>{title}</title>{meta}</head><body><nav>Menu Home About</nav>"
        f"<article><h1>{title}</h1><p>{body}</p></article><footer>Footer</footer></body></html>"
    )


def make_client(routes: dict[str, httpx.Response], seen_agents: list[str] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen_agents is not None:
            seen_agents.append(request.headers.get("user-agent", ""))
        return routes.get(str(request.url), httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_parse_seeds_skips_comments_and_blank_lines():
    text = "# energy storage\nhttps://a.example/x\n\n  https://b.example/y  # trailing note\n"
    assert parse_seeds(text) == ["https://a.example/x", "https://b.example/y"]


def test_crawl_extracts_markdown_title_and_date():
    client = make_client(
        {"https://a.example/pumped": httpx.Response(200, html=page("Pumped Storage"))}
    )
    rows, skipped = crawl(["https://a.example/pumped"], client, delay=0, sleep=lambda s: None)
    assert skipped == []
    (row,) = rows
    assert row["url"] == "https://a.example/pumped"
    assert row["title"] == "Pumped Storage"
    assert row["published_date"] == "2024-05-14"
    assert row["raw_content"].startswith("# Pumped Storage")
    assert "Menu Home About" not in row["raw_content"] and "Footer" not in row["raw_content"]


def test_crawl_skips_with_reasons():
    robots = "User-agent: *\nDisallow: /private/\n"
    client = make_client(
        {
            "https://a.example/robots.txt": httpx.Response(200, text=robots),
            "https://a.example/private/page": httpx.Response(200, html=page("Secret")),
            "https://a.example/short": httpx.Response(200, html=page("Short", body="Too short.")),
            "https://a.example/ok": httpx.Response(200, html=page("Ok")),
            "https://a.example/alias": httpx.Response(
                301, headers={"location": "https://a.example/ok"}
            ),
        }
    )
    seeds = [
        "https://a.example/private/page",
        "https://a.example/missing",
        "https://a.example/short",
        "https://a.example/ok",
        "https://a.example/alias",  # redirects to an already-kept page
    ]
    rows, skipped = crawl(seeds, client, delay=0, sleep=lambda s: None)
    assert [r["url"] for r in rows] == ["https://a.example/ok"]
    reasons = dict(skipped)
    assert "robots.txt" in reasons["https://a.example/private/page"]
    assert "404" in reasons["https://a.example/missing"]
    assert "too short" in reasons["https://a.example/short"]
    assert "duplicate" in reasons["https://a.example/alias"]


def test_crawl_is_polite():
    agents: list[str] = []
    sleeps: list[float] = []
    client = make_client(
        {
            "https://a.example/1": httpx.Response(200, html=page("One")),
            "https://a.example/2": httpx.Response(200, html=page("Two")),
        },
        agents,
    )
    crawl(["https://a.example/1", "https://a.example/2"], client, delay=1.5, sleep=sleeps.append)
    assert agents and all(a == USER_AGENT for a in agents)
    assert sleeps == [1.5, 1.5]  # one pause before each page fetch after the first request


def test_written_rows_load_as_a_valid_corpus(tmp_path):
    client = make_client({"https://a.example/p": httpx.Response(200, html=page("P", date=None))})
    rows, _ = crawl(["https://a.example/p"], client, delay=0, sleep=lambda s: None)
    out = tmp_path / "corpus" / "demo.jsonl"
    write_rows(rows, out)
    (doc,) = load_corpus(out.parent)
    assert doc.title == "P" and doc.published_date is None
    assert json.loads(out.read_text().splitlines()[0])["url"] == "https://a.example/p"
