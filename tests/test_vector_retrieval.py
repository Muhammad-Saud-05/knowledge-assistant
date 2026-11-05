import pytest

from src.retrieval.vector import RetrievedChunk, vector_search

TEXTS = [f"chunk number {i} about topic {i}" for i in range(8)]


@pytest.fixture
def stored(conn, add_document):
    chunks = [(text, i + 1, f"Heading {i}") for i, text in enumerate(TEXTS)]
    add_document(conn, "Big Doc", "docs/big.md", chunks)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_closest_chunk_comes_first_with_similarity_one(conn, embedder, stored):
    results = vector_search(conn, embedder, TEXTS[3], k=5)
    assert results[0].text == TEXTS[3]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_results_are_ordered_by_similarity_descending(conn, embedder, stored):
    scores = [r.score for r in vector_search(conn, embedder, TEXTS[0], k=8)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) > 1


def test_score_is_one_minus_cosine_distance(conn, embedder, stored):
    query = "a question that matches nothing in particular"
    query_vector = embedder.embed_query(query)
    for result in vector_search(conn, embedder, query, k=8):
        expected = dot(query_vector, embedder.embed_documents([result.text])[0])  # unit vectors
        assert result.score == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("k, expected", [(1, 1), (3, 3), (8, 8), (20, 8)])
def test_k_is_respected(conn, embedder, stored, k, expected):
    assert len(vector_search(conn, embedder, "anything", k=k)) == expected


def test_result_carries_document_details(conn, embedder, stored):
    result = vector_search(conn, embedder, TEXTS[2], k=1)[0]
    assert isinstance(result, RetrievedChunk)
    assert (result.title, result.path) == ("Big Doc", "docs/big.md")
    assert (result.page_number, result.section_heading) == (3, "Heading 2")
    assert result.chunk_id and result.document_id


def test_empty_database_returns_no_results(conn, embedder):
    assert vector_search(conn, embedder, "anything") == []


def test_invalid_arguments_are_rejected(conn, embedder):
    with pytest.raises(ValueError):
        vector_search(conn, embedder, "question", k=0)
    with pytest.raises(ValueError):
        vector_search(conn, embedder, "   ")


def test_query_with_sql_characters_is_just_text(conn, embedder, stored):
    hostile = "'; DROP TABLE chunks; --"
    assert len(vector_search(conn, embedder, hostile, k=3)) == 3
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == len(TEXTS)
