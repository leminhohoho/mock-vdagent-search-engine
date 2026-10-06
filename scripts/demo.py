"""Exercise a running mockserp server through the official Tavily SDK (`make demo`).

Runs the success-criterion queries of the Supabase design spec over the real-estate papers.
Start the server with MOCK_NOW=2026-10-06T00:00:00Z for the `time_range="week"` query to
match the expected result.
"""

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

    show(
        "search (expect paper 3 first)",
        client.search("giải ngân gói tín dụng 145.000 tỷ nhà ở xã hội", max_results=5),
    )
    show(
        "search (expect paper 17 first)",
        client.search("giá thuê văn phòng hạng A 64,7 USD/m²", max_results=5),
    )
    show(
        'exact_match "39.000 sản phẩm" (expect papers 14 and 18 only)',
        client.search('"39.000 sản phẩm"', exact_match=True, max_results=20),
    )
    show(
        "time_range=week, MOCK_NOW=2026-10-06 (expect papers 1, 2, 13, 14, 15, 16)",
        client.search(
            "thị trường bất động sản",
            time_range="week",
            include_published_date=True,
            max_results=20,
        ),
    )

    top = client.search("nhà ở xã hội", max_results=1)["results"][0]
    extracted = client.extract([top["url"], "https://unknown.example/page"], query="lãi suất vay")
    print(f"\n=== extract {top['url']} (query='lãi suất vay', chunks_per_source=3)")
    for r in extracted["results"]:
        print(f"  {r['raw_content'][:300]!r}")
    print(f"  failed: {extracted['failed_results']}")


if __name__ == "__main__":
    main()
