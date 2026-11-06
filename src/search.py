"""Search the stored documents without any language model.

Usage:
    python -m src.search "query"                  vector, keyword and hybrid results
    python -m src.search "query" --mode keyword   one mode only
    python -m src.search "query" --type pdf --since 2026-10-01 --path-prefix sample_corpus/
"""

import argparse
import sys

import psycopg

from .cli_filters import add_filter_arguments, filters_from_args
from .db.connection import get_connection
from .ingestion.embedder import Embedder
from .retrieval.filters import SearchFilters
from .retrieval.fusion import reciprocal_rank_fusion
from .retrieval.hybrid import DEFAULT_CANDIDATES, MODES, search
from .retrieval.keyword import keyword_search
from .retrieval.models import RetrievedChunk
from .retrieval.vector import vector_search

PREVIEW_CHARS = 100


def where(chunk: RetrievedChunk) -> str:
    """Page and/or heading of a chunk, for display."""
    parts = []
    if chunk.page_number is not None:
        parts.append(f"p. {chunk.page_number}")
    if chunk.section_heading:
        parts.append(chunk.section_heading)
    return ", ".join(parts) or "-"


def rank_text(rank: int | None) -> str:
    return f"#{rank}" if rank is not None else "-"


def format_result(position: int, chunk: RetrievedChunk) -> str:
    fused = f"{chunk.fused_score:.4f}" if chunk.fused_score is not None else "-"
    preview = " ".join(chunk.text.split())[:PREVIEW_CHARS]
    return (
        f"{position:>2}. {chunk.title or 'Untitled'} | {where(chunk)}\n"
        f"    vector {rank_text(chunk.vector_rank)} | keyword {rank_text(chunk.keyword_rank)} | fused {fused}\n"
        f"    {preview}"
    )


def print_results(heading: str, results: list[RetrievedChunk]) -> None:
    print(f"\n=== {heading} ===")
    if not results:
        print("(no results)")
    for position, chunk in enumerate(results, start=1):
        print(format_result(position, chunk))


def search_all_modes(
    conn, embedder, query: str, k: int, filters: SearchFilters | None
) -> dict[str, list[RetrievedChunk]]:
    """Vector, keyword and hybrid results, with each result's rank in the other lists.

    Both methods are searched once (30 candidates each); the three views come from
    those same two lists, so the ranks shown are consistent between them.
    """
    vector_results = vector_search(conn, embedder, query, k=DEFAULT_CANDIDATES, filters=filters)
    keyword_results = keyword_search(conn, query, k=DEFAULT_CANDIDATES, filters=filters)
    vector_by_id = {c.chunk_id: c for c in vector_results}
    keyword_by_id = {c.chunk_id: c for c in keyword_results}

    def with_cross_ranks(chunk: RetrievedChunk) -> RetrievedChunk:
        if chunk.vector_rank is None and chunk.chunk_id in vector_by_id:
            chunk.vector_rank = vector_by_id[chunk.chunk_id].vector_rank
        if chunk.keyword_rank is None and chunk.chunk_id in keyword_by_id:
            chunk.keyword_rank = keyword_by_id[chunk.chunk_id].keyword_rank
        return chunk

    return {
        "vector": [with_cross_ranks(c) for c in vector_results[:k]],
        "keyword": [with_cross_ranks(c) for c in keyword_results[:k]],
        "hybrid": reciprocal_rank_fusion([vector_results, keyword_results], limit=k),
    }


def print_comparison(views: dict[str, list[RetrievedChunk]]) -> None:
    """One row per chunk that appears in any view, with its position in each."""
    rows: dict[int, dict] = {}
    for mode, results in views.items():
        for position, chunk in enumerate(results, start=1):
            row = rows.setdefault(chunk.chunk_id, {"label": f"{chunk.title} | {where(chunk)}"})
            row[mode] = position
    print("\n=== Rank in each mode (top results of any mode; '-' = not in that mode's top results) ===")
    print(f"{'vector':>7} {'keyword':>8} {'hybrid':>7}  chunk")
    for row in sorted(rows.values(), key=lambda r: (r.get("hybrid", 99), r.get("vector", 99), r.get("keyword", 99))):
        cells = [rank_text(row.get(mode)) for mode in ("vector", "keyword", "hybrid")]
        print(f"{cells[0]:>7} {cells[1]:>8} {cells[2]:>7}  {row['label'][:80]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search your documents (no language model involved).")
    parser.add_argument("query", help="what to search for, in quotes")
    parser.add_argument("--k", type=int, default=5, help="results to show per mode (default 5)")
    parser.add_argument("--mode", choices=MODES, help="show one mode only (default: all three)")
    add_filter_arguments(parser)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")
    if not args.query.strip():
        parser.error("the query must not be empty")
    filters = filters_from_args(parser, args)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        conn = get_connection()
    except (psycopg.OperationalError, RuntimeError) as error:
        # Do not echo connection errors: they can mention the host and user.
        reason = str(error) if isinstance(error, RuntimeError) else "is the database running?"
        print(f"Could not connect to the database ({reason})", file=sys.stderr)
        return 2

    embedder = Embedder()
    print(f"Query: {args.query}")
    if filters:
        print(f"Filters: {filters}")
    with conn:
        if args.mode:
            results = search(conn, embedder, args.query, k=args.k, mode=args.mode, filters=filters)
            print_results(args.mode.upper(), results)
        else:
            views = search_all_modes(conn, embedder, args.query, args.k, filters)
            for mode, results in views.items():
                print_results(mode.upper(), results)
            print_comparison(views)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
