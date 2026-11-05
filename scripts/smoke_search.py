"""Throwaway sanity check: print the 5 chunks closest to a question.

This only proves that embedding and storage work end to end. It is NOT the
real retrieval: hybrid search, reranking and answer generation come on Day 4.

Usage: python scripts/smoke_search.py "how does a token bucket work"
"""

import sys
from pathlib import Path

# Allow running as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgvector import Vector  # noqa: E402

from src.db.connection import get_connection  # noqa: E402
from src.ingestion.embedder import Embedder  # noqa: E402

TOP_K = 5
PREVIEW_CHARS = 150

# <=> is pgvector's cosine distance; similarity = 1 - distance.
QUERY = """
    SELECT 1 - (c.embedding <=> %(q)s) AS similarity,
           d.title, c.page_number, c.section_heading, c.text
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    ORDER BY c.embedding <=> %(q)s
    LIMIT %(k)s
"""


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print('Usage: python scripts/smoke_search.py "your question"', file=sys.stderr)
        return 2
    question = sys.argv[1]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vector = Vector(Embedder().embed_query(question))
    with get_connection() as conn:
        rows = conn.execute(QUERY, {"q": vector, "k": TOP_K}).fetchall()

    print(f"Question: {question}\n")
    if not rows:
        print("No chunks stored yet. Run: python -m src.ingestion.ingest <folder>")
        return 1
    for rank, (similarity, title, page, heading, text) in enumerate(rows, start=1):
        where = f"page {page}" if page is not None else (heading or "-")
        preview = " ".join(text.split())[:PREVIEW_CHARS]
        print(f"{rank}. similarity {similarity:.3f} | {title} | {where}")
        print(f"   {preview}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
