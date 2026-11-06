"""Vector (meaning-based) retrieval: find the chunks closest to a question.

This is one of several retrieval strategies (see keyword.py, fusion.py and
hybrid.py). They all return RetrievedChunk objects.
"""

from pgvector import Vector

from .filters import SearchFilters, build_filter_sql
from .models import RetrievedChunk

# <=> is pgvector's cosine distance operator. The question vector and the
# filter values are passed as query parameters, never pasted into the SQL text;
# {condition} is only ever fixed text from build_filter_sql.
#
# The index scan sits in a MATERIALIZED subquery and the result is sorted again
# outside it: with hnsw.iterative_scan = relaxed_order the index may hand back
# rows slightly out of order, and this makes the final order exact.
# The inner ORDER BY must be only the distance so that Postgres can use the index.
_SEARCH_SQL = """
    WITH nearest AS MATERIALIZED (
        SELECT c.id, c.document_id, d.title, d.path, c.page_number,
               c.section_heading, c.text,
               c.embedding <=> %(query_vector)s AS distance
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {condition}
        ORDER BY c.embedding <=> %(query_vector)s
        LIMIT %(k)s
    )
    SELECT id, document_id, title, path, page_number, section_heading, text,
           1 - distance AS score
    FROM nearest
    ORDER BY distance, id
"""


def vector_search(
    conn, embedder, query: str, k: int = 5, filters: SearchFilters | None = None
) -> list[RetrievedChunk]:
    """The k chunks whose embeddings are closest to the query, best first.

    Needs pgvector 0.8 or newer (for hnsw.iterative_scan).
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        raise ValueError("query must not be empty")

    condition, filter_params = build_filter_sql(filters)
    params = {"query_vector": Vector(embedder.embed_query(query)), "k": k, **filter_params}

    # An HNSW index only returns its ~40 nearest candidates, and a filter is applied
    # after that. A selective filter could leave far fewer than k rows (it did: 1 of
    # 10 in a test). Iterative scan makes the index keep going until k rows pass the
    # filter. SET LOCAL lasts only for this transaction.
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(_SEARCH_SQL.format(condition=condition), params).fetchall()

    results = []
    for rank, row in enumerate(rows, start=1):
        chunk = RetrievedChunk(*row)
        chunk.vector_rank = rank
        chunk.vector_score = chunk.score
        results.append(chunk)
    return results
