"""Parse and chunk documents, then print stats to spot extraction problems.

Usage: python -m src.ingestion.inspect_cli <file-or-folder>
"""

import argparse
import sys
from pathlib import Path

from .chunker import chunk_document
from .models import ParsedDocument
from .parsers import MARKDOWN_EXTENSIONS, PDF_EXTENSIONS, parse_file

SUPPORTED = MARKDOWN_EXTENSIONS | PDF_EXTENSIONS
NEAR_EMPTY_WORDS = 30  # a page/section with fewer words than this gets flagged
PREVIEW_CHARS = 220


def _find_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED)


def _words(text: str) -> int:
    return len(text.split())


def _describe_units(doc: ParsedDocument) -> tuple[str, list[str]]:
    """Return a "pages"/"sections" summary and any empty or near-empty warnings."""
    warnings = []
    for section in doc.sections:
        count = _words(section.text)
        if doc.source_type == "pdf":
            # A near-empty PDF page usually means a cover page or failed extraction.
            if count < NEAR_EMPTY_WORDS:
                kind = "EMPTY" if count == 0 else f"near-empty ({count} words)"
                warnings.append(f"page {section.page_number}: {kind}")
        elif count == 0:
            # Short Markdown sections are normal; a heading with only
            # sub-headings under it has no text of its own.
            warnings.append(f"note: {section.heading!r} has no text of its own")
    unit = "pages" if doc.source_type == "pdf" else "sections"
    return f"{len(doc.sections)} {unit}", warnings


def inspect_file(path: Path) -> None:
    print("=" * 78)
    print(path)
    try:
        doc = parse_file(path)
    except Exception as error:  # report and carry on with the next file
        print(f"  ERROR: could not parse: {error}")
        return

    chunks = chunk_document(doc)
    sizes = [_words(c.text) for c in chunks]
    units, warnings = _describe_units(doc)

    print(f"  title:   {doc.title}")
    print(f"  type:    {doc.source_type}")
    print(f"  date:    {doc.doc_date}")
    print(f"  size:    {units}")
    print(f"  chunks:  {len(chunks)}")
    if sizes:
        print(
            f"  words/chunk: avg {sum(sizes) / len(sizes):.0f}, "
            f"min {min(sizes)}, max {max(sizes)}"
        )
    else:
        print("  words/chunk: n/a (no text was extracted)")
        print("  WARNING: no text at all. A scanned PDF needs OCR first.")
    for warning in warnings:
        print(f"  warning: {warning}")

    for chunk in chunks[:3]:
        page = chunk.page_number if chunk.page_number is not None else "-"
        preview = " ".join(chunk.text.split())
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "..."
        print(
            f"\n  [chunk {chunk.chunk_index}] heading: {chunk.section_heading!r} "
            f"| page: {page} | {_words(chunk.text)} words"
        )
        print(f"    {preview}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse and chunk documents and print stats for checking extraction."
    )
    parser.add_argument("target", help="a .pdf/.md file or a folder containing them")
    args = parser.parse_args(argv)

    # Windows consoles often default to a legacy encoding; avoid crashing on odd characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    target = Path(args.target)
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        return 1
    files = _find_files(target)
    if not files:
        print(f"No supported files ({', '.join(sorted(SUPPORTED))}) found in {target}")
        return 1
    for path in files:
        inspect_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
