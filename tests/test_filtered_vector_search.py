"""A filtered HNSW search must still return k rows (hnsw.iterative_scan)."""

import pytest
from pgvector import Vector

from src.retrieval import vector as vector_module
from src.retrieval.filters import SearchFilters, build_filter_sql
from src.retrieval.vector import vector_search

DOCS, CHUNKS_PER_DOC, K = 30, 40, 10  # a filter on one document keeps 40 of 1200 chunks (3%)


@pytest.fixture
def hnsw_only(conn, embedder):
    """Fill the test database and make HNSW the only way to find nearest rows.

    On a small table Postgres would simply scan everything (always exact), which
    hides the problem. Everything here is undone by the rollback when the test ends:
    dropping an index and SET LOCAL are both transactional.
    """
    for d in range(DOCS):
        doc_id = conn.execute(
            "INSERT INTO documents (source_type, path, title, content_hash) "
            "VALUES ('markdown', %s, %s, %s) RETURNING id",
            (f"doc{d:02d}/file.md", f"Doc {d}", f"hash{d}"),
        ).fetchone()[0]
        texts = [f"document {d} chunk {i}" for i in range(CHUNKS_PER_DOC)]
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (document_id, chunk_index, text, embedding) VALUES (%s, %s, %s, %s)",
                [(doc_id, i, t, Vector(v)) for i, (t, v) in enumerate(zip(texts, embedder.embed_documents(texts)))],
            )
    conn.execute("DROP INDEX chunks_document_id_idx")
    conn.execute("ANALYZE chunks")
    conn.execute("SET LOCAL enable_seqscan = off")
    conn.execute("SET LOCAL enable_bitmapscan = off")
    conn.execute("SET LOCAL enable_hashjoin = off")
    conn.execute("SET LOCAL enable_mergejoin = off")
    yield
    conn.rollback()


def test_selective_filter_still_returns_k_results(conn, embedder, hnsw_only):
    filters = SearchFilters(path_prefix="doc07/")

    # Control: with iterative scan off, this setup really does fall short. If this
    # ever fails, the planner no longer uses HNSW here and the test proves nothing.
    condition, filter_params = build_filter_sql(filters)
    conn.execute("SET LOCAL hnsw.iterative_scan = off")
    params = {"query_vector": Vector(embedder.embed_query("some question")), "k": K, **filter_params}
    short = conn.execute(vector_module._SEARCH_SQL.format(condition=condition), params).fetchall()
    assert len(short) < K, "control failed: the filter no longer cuts the HNSW results short"

    results = vector_search(conn, embedder, "some question", k=K, filters=filters)

    assert len(results) == K
    assert {r.path for r in results} == {"doc07/file.md"}
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.vector_rank for r in results] == list(range(1, K + 1))


def test_results_match_the_exact_nearest_neighbours(conn, embedder, hnsw_only):
    filters = SearchFilters(path_prefix="doc12/")
    found = {r.chunk_id for r in vector_search(conn, embedder, "another question", k=K, filters=filters)}

    conn.execute("SET LOCAL enable_indexscan = off")  # exact scan, no HNSW
    conn.execute("SET LOCAL enable_seqscan = on")
    exact = {
        r[0]
        for r in conn.execute(
            """SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE starts_with(d.path, 'doc12/') ORDER BY c.embedding <=> %s LIMIT %s""",
            (Vector(embedder.embed_query("another question")), K),
        )
    }
    assert found == exact
