"""Exercise a running mockserp server through the official Tavily SDK (`make demo`)."""

import os

from tavily import TavilyClient

from mockserp.config import Settings


def show(title: str, response: dict) -> None:
    print(f"\n=== {title} ({response['response_time']}s)")
    for r in response["results"]:
        date = f"  [{r['published_date']}]" if r.get("published_date") else ""
        print(f"  {r['score']:.3f}  {r['url']}{date}\n         {r['content'][:140]!r}")


def main() -> None:
    settings = Settings()
    base_url = os.environ.get("MOCK_BASE_URL", f"http://{settings.host}:{settings.port}")
    client = TavilyClient(api_key=settings.mock_api_key or "tvly-mock", api_base_url=base_url)

    show("search", client.search("pumped hydro round-trip efficiency", max_results=5))
    show(
        "search, government sites only, with dates",
        client.search(
            "battery storage cost",
            include_domains=["energy.gov", "eia.gov", "nrel.gov", "pnnl.gov"],
            include_published_date=True,
            max_results=5,
        ),
    )
    show(
        'search, exact_match "Hornsdale"',
        client.search('"Hornsdale" Tesla battery South Australia', exact_match=True, max_results=3),
    )

    top = client.search("compressed air energy storage caverns", max_results=1)["results"][0]
    extracted = client.extract([top["url"], "https://unknown.example/page"], query="salt cavern")
    print(f"\n=== extract {top['url']} (query='salt cavern', chunks_per_source=3)")
    for r in extracted["results"]:
        print(f"  {r['raw_content'][:300]!r}")
    print(f"  failed: {extracted['failed_results']}")


if __name__ == "__main__":
    main()
