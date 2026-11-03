"""Split parsed documents into chunks that are small enough to search.

Why these sizes (in plain language):

* max_words=300: a search result has to be focused. A chunk of about 300
  words is roughly three to five paragraphs, enough to hold one complete
  idea, but short enough that the matching passage is not buried in
  unrelated text. It also fits comfortably in the small input window of an
  embedding model and in a prompt alongside several other chunks.
* overlap_words=40: when a long section is cut into several chunks, the cut
  can land in the middle of an idea. Repeating the last ~40 words (about two
  sentences) at the start of the next chunk means that idea is still whole
  in at least one chunk. More overlap would only create near-duplicates.

Chunks never cross a section boundary (and so never cross a PDF page), so
each chunk can be cited with an accurate heading and page number.
"""

import re

from .models import Chunk, ParsedDocument

# Sections shorter than this may be merged into the previous related section.
TINY_SECTION_WORDS = 30

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> int:
    return len(text.split())


def _paragraphs(text: str) -> list[tuple[str, bool]]:
    """Split on blank lines, keeping fenced code blocks whole.

    Returns (paragraph, is_code_block) pairs.
    """
    paragraphs: list[tuple[str, bool]] = []
    current: list[str] = []
    fence: str | None = None

    def flush() -> None:
        if current and "".join(current).strip():
            paragraphs.append(("\n".join(current), bool(_FENCE_RE.match(current[0]))))
        current.clear()

    for line in text.split("\n"):
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        if fence is None and not line.strip():
            flush()
        else:
            current.append(line)
    flush()
    return paragraphs


def _word_windows(text: str, limit: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]


def _pieces(text: str, limit: int) -> list[tuple[str, str]]:
    """Break text into (piece, separator-before-it) units of at most `limit` words.

    Paragraphs are kept whole when they fit. Longer ones are split into
    sentences (or lines, for code), and anything still too long into words.
    """
    pieces: list[tuple[str, str]] = []
    for paragraph, is_code in _paragraphs(text):
        if _words(paragraph) <= limit:
            pieces.append((paragraph, "\n\n"))
            continue
        if is_code:
            parts, joiner = paragraph.split("\n"), "\n"
        else:
            parts, joiner = _SENTENCE_END_RE.split(paragraph), " "
        for j, part in enumerate(parts):
            if not part.strip():
                continue
            fragments = [part] if _words(part) <= limit else _word_windows(part, limit)
            for k, fragment in enumerate(fragments):
                if not pieces or (j == 0 and k == 0):
                    sep = "\n\n"
                else:
                    sep = joiner if k == 0 else " "
                pieces.append((fragment, sep))
    return pieces


def _overlap_tail(text: str, overlap_words: int) -> str:
    """The last ~overlap_words words of text, starting at a sentence if possible."""
    if overlap_words <= 0:
        return ""
    starts = [m.start() for m in re.finditer(r"\S+", text)]
    tail = text[starts[-overlap_words] :] if len(starts) > overlap_words else text
    boundary = _SENTENCE_END_RE.search(tail)
    if boundary and _words(tail[boundary.end() :]) >= overlap_words // 2:
        tail = tail[boundary.end() :]
    return tail


def _split_section(text: str, max_words: int, overlap_words: int) -> list[str]:
    if _words(text) <= max_words:
        return [text]

    # Leave room for the overlap so a chunk (overlap + new text) fits max_words.
    pieces = _pieces(text, max_words - overlap_words)
    chunks: list[str] = []
    current = ""
    current_words = 0
    has_new_text = False
    for piece, sep in pieces:
        piece_words = _words(piece)
        if has_new_text and current_words + piece_words > max_words:
            chunks.append(current)
            current = _overlap_tail(current, overlap_words)
            current_words = _words(current)
            has_new_text = False
        current = current + sep + piece if current else piece
        current_words += piece_words
        has_new_text = True
    if has_new_text:
        chunks.append(current)
    return chunks


def _is_related(previous: str | None, heading: str | None) -> bool:
    """True if `heading` is the same as, or a sub-heading of, `previous`."""
    if previous is None or heading is None:
        return previous is heading
    return heading == previous or heading.startswith(previous + " > ")


def _merge_tiny_sections(doc: ParsedDocument, max_words: int) -> list[dict]:
    """Drop empty sections and fold tiny ones into the previous related section."""
    groups: list[dict] = []
    for section in doc.sections:
        text = section.text.strip()
        if not text:
            continue
        previous = groups[-1] if groups else None
        if (
            previous
            and _words(text) < TINY_SECTION_WORDS
            and previous["page"] == section.page_number
            and _is_related(previous["heading"], section.heading)
            and _words(previous["text"]) + _words(text) + 8 <= max_words
        ):
            # Keep the sub-heading's name so that context is not lost.
            if section.heading and section.heading != previous["heading"]:
                text = section.heading.rsplit(" > ", 1)[-1] + "\n\n" + text
            previous["text"] += "\n\n" + text
            continue
        groups.append(
            {"heading": section.heading, "page": section.page_number, "text": text}
        )
    return groups


def chunk_document(
    doc: ParsedDocument, max_words: int = 300, overlap_words: int = 40
) -> list[Chunk]:
    if overlap_words < 0 or max_words <= overlap_words:
        raise ValueError("max_words must be greater than overlap_words (which must be >= 0)")

    chunks: list[Chunk] = []
    for group in _merge_tiny_sections(doc, max_words):
        for text in _split_section(group["text"], max_words, overlap_words):
            chunks.append(
                Chunk(
                    text=text.strip(),
                    chunk_index=len(chunks),
                    section_heading=group["heading"],
                    page_number=group["page"],
                )
            )
    return chunks
