"""Vector (meaning-based) retrieval: find the chunks closest to a question.

This is the first of several retrieval strategies. Later days can add keyword
search and a way to combine strategies next to it, in this package, and reuse
RetrievedChunk as the common result type.
"""

from pgvector import Vector

from .models import RetrievedChunk  # noqa: F401  (re-exported: existing imports keep working)


# <=> is pgvector's cosine distance operator. The question vector is passed as
# a query parameter, never pasted into the SQL text.
_SEARCH_SQL = """
    SELECT c.id, c.document_id, d.title, d.path, c.page_number,
           c.section_heading, c.text,
           1 - (c.embedding <=> %(query_vector)s) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    ORDER BY c.embedding <=> %(query_vector)s, c.id
    LIMIT %(k)s
"""


def vector_search(conn, embedder, query: str, k: int = 5) -> list[RetrievedChunk]:
    """The k chunks whose embeddings are closest to the query, best first."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        raise ValueError("query must not be empty")

    query_vector = Vector(embedder.embed_query(query))
    rows = conn.execute(_SEARCH_SQL, {"query_vector": query_vector, "k": k}).fetchall()
    return [RetrievedChunk(*row) for row in rows]
