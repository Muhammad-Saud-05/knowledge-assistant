"""Ingest documents into the database.

Usage:
    python -m src.ingestion.ingest <file-or-folder> [--dry-run] [--min-words N]
    python -m src.ingestion.ingest --delete <path>
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import psycopg

from ..db.connection import get_connection
from .embedder import Embedder
from .pipeline import (
    DEFAULT_MIN_CHUNK_WORDS,
    SUPPORTED_EXTENSIONS,
    delete_document,
    find_documents,
    get_document_info,
    ingest_files,
)

STATUSES = ("new", "updated", "unchanged", "failed")


def _print_result(result) -> None:
    line = f"{result.status:<10}{result.chunks:>4} chunks  {result.seconds:>6.1f}s  {result.path}"
    if result.error:
        line += f"\n{'':<10}{result.error}"
    print(line)


def _delete(path: str, conn) -> int:
    info = get_document_info(path, conn)
    if info is None or not delete_document(path, conn):
        print(f"Nothing to delete: no stored document with path {path!r}")
        return 1
    title, chunk_count = info
    print(f"Deleted {path} (title: {title!r}) and its {chunk_count} chunks")
    return 0


def _ingest(target: str, conn, min_words: int, dry_run: bool) -> int:
    path = Path(target)
    if not path.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        return 1
    files = find_documents(path)
    if not files:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"No supported files ({extensions}) found in {target}")
        return 1

    if dry_run:
        print("Dry run: nothing will be embedded or written.")
    counts: Counter[str] = Counter()
    total_chunks = 0
    for result in ingest_files(files, conn, Embedder(), min_words, dry_run):
        _print_result(result)
        counts[result.status] += 1
        if result.status != "failed":
            total_chunks += result.chunks
    summary = ", ".join(f"{counts[s]} {s}" for s in STATUSES)
    verb = "would be" if dry_run else "are"
    print(f"\n{len(files)} files: {summary}. {total_chunks} chunks {verb} stored in total.")
    return 1 if counts["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed documents and store them in the database.")
    parser.add_argument("target", nargs="?", help="a .pdf/.md file or a folder containing them")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen, write nothing")
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_CHUNK_WORDS,
        help=f"drop chunks shorter than this (default {DEFAULT_MIN_CHUNK_WORDS})",
    )
    parser.add_argument("--delete", metavar="PATH", help="delete one stored document and its chunks")
    args = parser.parse_args(argv)

    if bool(args.target) == bool(args.delete):
        parser.error("give either a file/folder to ingest, or --delete <path>")
    if args.delete and args.dry_run:
        parser.error("--dry-run applies to ingesting, not --delete")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        conn = get_connection()
    except (psycopg.OperationalError, RuntimeError) as error:
        # Do not echo the error text: connection errors can mention host and user.
        reason = str(error) if isinstance(error, RuntimeError) else "is the database running?"
        print(f"Could not connect to the database ({reason})", file=sys.stderr)
        return 2
    with conn:
        if args.delete:
            return _delete(args.delete, conn)
        return _ingest(args.target, conn, args.min_words, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
