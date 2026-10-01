"""Split page text into sentence-aware chunks of at most MAX_CHUNK_CHARS characters.

Chunks are verbatim slices of the source text (Markdown preserved), never overlap,
and together cover every non-whitespace character of the input.
"""

import re

MAX_CHUNK_CHARS = 500

# Unit boundaries: whitespace after sentence-ending punctuation, or any line break.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")


def _units(text: str) -> list[tuple[int, int]]:
    """Spans of sentences/lines, trimmed of surrounding whitespace."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    out = []
    for s, e in spans:
        piece = text[s:e]
        stripped = piece.strip()
        if stripped:
            s += len(piece) - len(piece.lstrip())
            out.append((s, s + len(stripped)))
    return out


def _hard_split(text: str, s: int, e: int) -> list[tuple[int, int]]:
    """Split an over-long span at the last whitespace before the limit (or at the limit)."""
    out = []
    while e - s > MAX_CHUNK_CHARS:
        window = text[s : s + MAX_CHUNK_CHARS]
        cut = s + MAX_CHUNK_CHARS
        if not text[cut].isspace():
            ws = max(window.rfind(" "), window.rfind("\t"))
            if ws > 0:
                cut = s + ws
        out.append((s, s + len(text[s:cut].rstrip())))
        s = cut
        while s < e and text[s].isspace():
            s += 1
    if s < e:
        out.append((s, e))
    return out


def chunk(text: str) -> list[str]:
    units: list[tuple[int, int]] = []
    for s, e in _units(text):
        units.extend(_hard_split(text, s, e) if e - s > MAX_CHUNK_CHARS else [(s, e)])

    chunks: list[str] = []
    cur: tuple[int, int] | None = None
    for s, e in units:
        if cur is not None and e - cur[0] <= MAX_CHUNK_CHARS:
            cur = (cur[0], e)
            continue
        if cur is not None:
            chunks.append(text[cur[0] : cur[1]])
        cur = (s, e)
    if cur is not None:
        chunks.append(text[cur[0] : cur[1]])
    return chunks
