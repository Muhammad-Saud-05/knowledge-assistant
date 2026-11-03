"""Turn Markdown and PDF files into ParsedDocument objects."""

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pymupdf

from .models import ParsedDocument, Section

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
PDF_EXTENSIONS = {".pdf"}

# A line must appear on at least this share of pages to count as a header/footer.
REPEATED_LINE_SHARE = 0.6
# Only text in the top or bottom share of the page height can be a header/footer.
EDGE_BAND = 0.10


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_date(path: Path) -> date:
    return datetime.fromtimestamp(path.stat().st_mtime).date()


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _split_front_matter(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return (front matter fields, remaining lines)."""
    if not lines or lines[0].strip() != "---":
        return {}, lines
    for end in range(1, len(lines)):
        if lines[end].strip() in ("---", "..."):
            fields: dict[str, str] = {}
            for line in lines[1:end]:
                key, sep, value = line.partition(":")
                if sep and key.strip():
                    fields[key.strip().lower()] = value.strip().strip("\"'")
            return fields, lines[end + 1 :]
    return {}, lines  # no closing marker: it was just a horizontal rule


def parse_markdown(path: str | Path) -> ParsedDocument:
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    fields, lines = _split_front_matter(lines)

    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current: list[str] = []
    current_heading: str | None = None
    first_h1: str | None = None
    fence: str | None = None  # the opening fence marker while inside a code block

    def flush() -> None:
        text = "\n".join(current).strip("\n").rstrip()
        if text.strip() or current_heading is not None:
            sections.append(Section(current_heading, None, text))

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            current.append(line)
            continue

        heading = None if fence else _HEADING_RE.match(line)
        if heading is None:
            current.append(line)
            continue

        flush()
        level, text = len(heading.group(1)), heading.group(2)
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, text))
        current_heading = " > ".join(t for _, t in heading_stack)
        current = []
        if level == 1 and first_h1 is None:
            first_h1 = text
    flush()

    doc_date = None
    if "date" in fields:
        try:
            doc_date = date.fromisoformat(fields["date"][:10])
        except ValueError:
            doc_date = None

    return ParsedDocument(
        source_type="markdown",
        path=str(path),
        title=fields.get("title") or first_h1 or path.stem,
        content_hash=_sha256(raw),
        doc_date=doc_date or _file_date(path),
        sections=sections,
    )


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


def _normalise_line(line: str) -> str:
    """Make "Page 3" and "Page 4" compare equal by masking digits."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines into one paragraph, repairing hyphenated words."""
    out = lines[0]
    for nxt in lines[1:]:
        if re.search(r"[A-Za-z]-$", out) and nxt[:1].islower():
            out = out[:-1] + nxt
        else:
            out += " " + nxt
    return out


@dataclass
class _Block:
    lines: list[str]
    at_edge: bool  # inside the top or bottom margin band of the page


def _page_blocks(page: pymupdf.Page) -> list[_Block]:
    """Text blocks of a page in reading order."""
    height = page.rect.height
    blocks = []
    for x0, y0, x1, y1, text, _, kind in page.get_text("blocks", sort=True):
        if kind != 0:  # skip images
            continue
        text = text.replace("­", "").replace("\xa0", " ")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            at_edge = y1 <= height * EDGE_BAND or y0 >= height * (1 - EDGE_BAND)
            blocks.append(_Block(lines, at_edge))
    return blocks


def _find_repeated_lines(pages: list[list[_Block]]) -> set[str]:
    """Normalised margin lines that appear on most pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for blocks in pages:
        keys = {
            _normalise_line(line)
            for block in blocks
            if block.at_edge
            for line in block.lines
        }
        counts.update(keys)
    needed = max(2, math.ceil(REPEATED_LINE_SHARE * len(pages)))
    return {key for key, n in counts.items() if n >= needed}


def _clean_page(blocks: list[_Block], repeated: set[str]) -> list[str]:
    """Paragraphs of a page with headers/footers removed."""
    paragraphs = []
    for block in blocks:
        kept = [
            line
            for line in block.lines
            if not (block.at_edge and _normalise_line(line) in repeated)
        ]
        if kept:
            paragraphs.append(_join_lines(kept))
    return paragraphs


def _looks_like_heading(paragraph: str) -> bool:
    return (
        bool(_NUMBERED_HEADING_RE.match(paragraph))
        and "\n" not in paragraph
        and len(paragraph.split()) <= 12
        and paragraph[-1] not in ".,;:!?"
    )


def _pdf_date(value: str | None) -> date | None:
    match = _PDF_DATE_RE.match(value or "")
    if not match:
        return None
    try:
        return date(*map(int, match.groups()))
    except ValueError:
        return None


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    raw = path.read_bytes()
    with pymupdf.open(stream=raw, filetype="pdf") as pdf:
        metadata = pdf.metadata or {}
        pages = [_page_blocks(page) for page in pdf]

    repeated = _find_repeated_lines(pages)
    sections: list[Section] = []
    for number, blocks in enumerate(pages, start=1):
        paragraphs = _clean_page(blocks, repeated)
        heading = None
        if paragraphs and _looks_like_heading(paragraphs[0]):
            heading = paragraphs.pop(0)
        text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(paragraphs)).strip()
        sections.append(Section(heading, number, text))

    title = (metadata.get("title") or "").strip()
    if not title or title.lower() == "untitled":
        first_text = next((s.text for s in sections if s.text), "")
        first_line = first_text.split("\n", 1)[0].strip()
        title = first_line if 0 < len(first_line.split()) <= 15 else path.stem

    return ParsedDocument(
        source_type="pdf",
        path=str(path),
        title=title,
        content_hash=_sha256(raw),
        doc_date=_pdf_date(metadata.get("creationDate")) or _file_date(path),
        sections=sections,
    )


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


def parse_file(path: str | Path) -> ParsedDocument:
    path = Path(path)
    extension = path.suffix.lower()
    if extension in MARKDOWN_EXTENSIONS:
        return parse_markdown(path)
    if extension in PDF_EXTENSIONS:
        return parse_pdf(path)
    supported = ", ".join(sorted(MARKDOWN_EXTENSIONS | PDF_EXTENSIONS))
    raise ValueError(
        f"Unsupported file type '{extension or '(none)'}' for {path.name}. "
        f"Supported types: {supported}"
    )
