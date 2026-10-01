from mockserp.chunker import MAX_CHUNK_CHARS, chunk


def _norm(s: str) -> str:
    return " ".join(s.split())


def test_empty_or_blank_text_yields_no_chunks():
    assert chunk("") == []
    assert chunk("  \n\n \t") == []


def test_short_text_is_one_chunk_verbatim():
    assert chunk("Hello world. Second sentence!") == ["Hello world. Second sentence!"]


def test_sentences_are_packed_greedily_without_splitting_them():
    s1 = "A" * 199 + "."  # 200 chars
    s2 = "B" * 199 + "."
    s3 = "C" * 199 + "."
    text = f"{s1} {s2} {s3}"
    # s1 + " " + s2 = 401 chars fits; adding " " + s3 would be 602 > 500.
    assert chunk(text) == [f"{s1} {s2}", s3]


def test_newlines_are_boundaries_and_markdown_is_preserved():
    heading = "# Pumped storage"
    body = "x" * 490 + "."
    text = f"{heading}\n{body}"
    # Together they exceed 500, so the heading line must end its own chunk
    # instead of being cut mid-line.
    assert chunk(text) == [heading, body]


def test_overlong_sentence_is_hard_split_at_whitespace():
    words = ["word%03d" % i for i in range(200)]  # 7 chars each
    text = " ".join(words) + "."
    chunks = chunk(text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    # No word is cut in half: every chunk boundary falls between words.
    assert [w for c in chunks for w in c.rstrip(".").split()] == words


def test_unbreakable_run_is_hard_split_at_limit():
    text = "z" * 1100
    assert chunk(text) == ["z" * 500, "z" * 500, "z" * 100]


def test_long_document_loses_no_text_and_respects_limit():
    paragraphs = [
        "Grid batteries store energy. They discharge at peak demand! Do they degrade? Yes.",
        "## Economics",
        "Levelized cost of storage depends on cycles per year and round-trip efficiency. " * 9,
        "- item one\n- item two",
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunk(text)
    assert all(0 < len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert _norm(" ".join(chunks)) == _norm(text)
