import unicodedata
from datetime import UTC, datetime

import pytest

from mockserp.corpus import (
    CorpusError,
    format_date,
    load_corpus,
    normalize_url,
    parse_date,
    tokenize,
)


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


FRONT = """---
url: "https://www.energy.gov/storage"
title: "Storage"
published_date: "2024-05-14"
favicon: "https://www.energy.gov/favicon.ico"
ignored_key: 1
---

"""
BODY = "# **Storage**\n\nBatteries store energy.\n"


def test_markdown_file_becomes_document_with_body_starting_at_h1(tmp_path):
    (tmp_path / "paper.md").write_text(FRONT + BODY, encoding="utf-8")
    (doc,) = load_corpus(tmp_path)
    assert (doc.url, doc.title, doc.favicon) == (
        "https://www.energy.gov/storage",
        "Storage",
        "https://www.energy.gov/favicon.ico",
    )
    assert doc.raw_content == BODY
    assert doc.published_date == datetime(2024, 5, 14, tzinfo=UTC)


def test_files_load_in_sorted_name_order_and_other_files_are_ignored(tmp_path):
    for name in ("b.md", "a.md"):
        (tmp_path / name).write_text(
            f'---\nurl: "https://{name}.example/"\ntitle: "{name}"\n---\nBody {name}\n'
        )
    (tmp_path / "notes.txt").write_text("not a page")
    (tmp_path / "old.jsonl").write_text('{"url": "https://x.example/"}\n')
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text('---\nurl: "https://c.example/"\ntitle: c\n---\nC\n')
    docs = load_corpus(tmp_path)
    assert [d.title for d in docs] == ["a.md", "b.md"]
    assert docs[0].published_date is None and docs[0].favicon is None


def test_unquoted_yaml_date_is_accepted(tmp_path):
    (tmp_path / "p.md").write_text(
        "---\nurl: https://a.example/\ntitle: T\npublished_date: 2026-09-29\n---\nBody\n"
    )
    assert load_corpus(tmp_path)[0].published_date == datetime(2026, 9, 29, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("# No front matter\nBody\n", "front matter"),
        ("\n---\nurl: https://a.example/\ntitle: T\n---\nBody\n", "front matter"),
        ('---\nurl: "https://a.example/"\ntitle: T\nBody without closing marker\n', "front matter"),
        ("---\n- a list\n- not a mapping\n---\nBody\n", "mapping"),
        ("---\nurl: [unclosed\n---\nBody\n", "YAML"),
        ("---\ntitle: T\n---\nBody\n", "url must be an absolute http(s) URL, got None"),
        ('---\nurl: "https://a.example/"\ntitle: "  "\n---\nBody\n', "title"),
        ('---\nurl: "https://a.example/"\ntitle: T\n---\n\n\n   \n', "raw_content"),
        (
            '---\nurl: "https://a.example/"\ntitle: T\npublished_date: soon\n---\nB\n',
            "published_date",
        ),
    ],
)
def test_invalid_markdown_file_is_rejected_with_its_name(tmp_path, text, reason):
    (tmp_path / "paper-RE (3).md").write_text(text)
    with pytest.raises(CorpusError) as e:
        load_corpus(tmp_path)
    assert str(e.value).startswith("paper-RE (3).md: ")
    assert reason in str(e.value)


def test_duplicate_normalized_urls_across_files_are_rejected(tmp_path):
    (tmp_path / "a.md").write_text('---\nurl: "https://Example.com/page/"\ntitle: A\n---\nA\n')
    (tmp_path / "b.md").write_text('---\nurl: "https://example.com/page#top"\ntitle: B\n---\nB\n')
    with pytest.raises(CorpusError) as e:
        load_corpus(tmp_path)
    assert (
        str(e.value).startswith("b.md: ") and "duplicate" in str(e.value) and "a.md" in str(e.value)
    )


def test_empty_corpus_directory_is_an_error(tmp_path):
    (tmp_path / "readme.txt").write_text("x")
    with pytest.raises(CorpusError, match=r"no documents found in .*/\*\.md"):
        load_corpus(tmp_path)


def test_tokenize_lowercases_and_splits_on_non_word_characters():
    assert tokenize("Giá thuê 64,7 USD/m²") == ["giá", "thuê", "64", "7", "usd", "m²"]


def test_tokenize_treats_decomposed_and_composed_text_alike():
    composed = unicodedata.normalize("NFC", "Nhà ở xã hội")
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert tokenize(decomposed) == tokenize(composed) == ["nhà", "ở", "xã", "hội"]
