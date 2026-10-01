import json
from datetime import UTC, datetime

import pytest

from mockserp.corpus import CorpusError, format_date, load_corpus, normalize_url, parse_date


@pytest.mark.parametrize(
    ("url", "want"),
    [
        ("HTTPS://Example.COM:443/a/b/#frag", "https://example.com/a/b"),
        ("http://example.com:80/", "http://example.com/"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com/a?x=1#y", "https://example.com/a?x=1"),
        ("https://example.com:8443/a/", "https://example.com:8443/a"),
        ("https://example.com/Case/Path", "https://example.com/Case/Path"),
    ],
)
def test_normalize_url(url, want):
    assert normalize_url(url) == want


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("2024-05-14", datetime(2024, 5, 14, tzinfo=UTC)),
        ("2024-05-14T10:30:00Z", datetime(2024, 5, 14, 10, 30, tzinfo=UTC)),
        ("2024-05-14T12:30:00+02:00", datetime(2024, 5, 14, 10, 30, tzinfo=UTC)),
        ("2024-05-14T10:30:00", datetime(2024, 5, 14, 10, 30, tzinfo=UTC)),
        ("Tue, 14 May 2024 10:30:00 GMT", datetime(2024, 5, 14, 10, 30, tzinfo=UTC)),
    ],
)
def test_parse_date_accepts_iso_and_rfc1123(raw, want):
    assert parse_date(raw) == want


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("last tuesday")


def test_format_date_is_rfc1123_gmt():
    assert format_date(datetime(2024, 5, 14, tzinfo=UTC)) == "Tue, 14 May 2024 00:00:00 GMT"


def _write(path, rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")


DOC = {
    "url": "https://www.energy.gov/storage",
    "title": "Storage",
    "raw_content": "Batteries.",
    "published_date": "2024-05-14",
    "favicon": "https://www.energy.gov/favicon.ico",
}


def test_load_corpus_reads_all_jsonl_files_in_name_order(tmp_path):
    _write(tmp_path / "b.jsonl", [{**DOC, "url": "https://b.example/x"}])
    _write(
        tmp_path / "a.jsonl",
        [DOC, "", {"url": "https://a.example/", "title": "T", "raw_content": "C"}],
    )
    docs = load_corpus(tmp_path)
    assert [d.url for d in docs] == [
        "https://www.energy.gov/storage",
        "https://a.example/",
        "https://b.example/x",
    ]
    first = docs[0]
    assert (first.title, first.raw_content, first.favicon) == (
        "Storage",
        "Batteries.",
        DOC["favicon"],
    )
    assert first.published_date == datetime(2024, 5, 14, tzinfo=UTC)
    assert first.host == "www.energy.gov"
    assert docs[1].published_date is None and docs[1].favicon is None


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ({**DOC, "title": "  "}, "title"),
        ({k: v for k, v in DOC.items() if k != "raw_content"}, "raw_content"),
        ({**DOC, "url": "ftp://x.example/a"}, "url"),
        ({**DOC, "url": "/relative"}, "url"),
        ({**DOC, "published_date": "soon"}, "published_date"),
        ("{not json", "JSON"),
    ],
)
def test_load_corpus_reports_file_and_line_of_bad_rows(tmp_path, row, reason):
    _write(tmp_path / "c.jsonl", [{**DOC, "url": "https://ok.example/"}, row])
    with pytest.raises(CorpusError) as e:
        load_corpus(tmp_path)
    assert "c.jsonl:2" in str(e.value)
    assert reason in str(e.value)


def test_load_corpus_rejects_duplicate_normalized_urls_across_files(tmp_path):
    _write(tmp_path / "a.jsonl", [{**DOC, "url": "https://Example.com/page/"}])
    _write(tmp_path / "b.jsonl", [{**DOC, "url": "https://example.com/page#top"}])
    with pytest.raises(CorpusError) as e:
        load_corpus(tmp_path)
    assert "b.jsonl:1" in str(e.value) and "duplicate" in str(e.value)


def test_load_corpus_requires_at_least_one_document(tmp_path):
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)
