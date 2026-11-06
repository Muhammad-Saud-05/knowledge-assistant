from datetime import date

import pytest

from src.retrieval.filters import SearchFilters
from src.retrieval.hybrid import MODES, search

VECTOR_ONLY = "completely unrelated gibberish about zebras"  # no word in common with the query
KEYWORD_ONLY = "this passage mentions the quokka by name"
FILLERS = [f"filler chunk number {i} about nothing in particular" for i in range(6)]


class PointedEmbedder:
    """Embeds every query as the vector of one chosen text, so that text is the exact nearest match."""

    def __init__(self, base, target_text):
        self._base, self._target = base, target_text

    def embed_documents(self, texts):
        return self._base.embed_documents(texts)

    def embed_query(self, text):
        return self._base.embed_documents([self._target])[0]


@pytest.fixture
def stored(conn, add_document):
    chunks = [(VECTOR_ONLY, None, "Zebras"), (KEYWORD_ONLY, None, "Quokka")]
    chunks += [(text, None, f"Filler {i}") for i, text in enumerate(FILLERS)]
    add_document(conn, "Notes", "notes/all.md", chunks)


@pytest.fixture
def pointed(embedder):
    return PointedEmbedder(embedder, VECTOR_ONLY)


def headings(results):
    return [r.section_heading for r in results]


# ---------------------------------------------------------------------- modes


def test_all_three_modes_work(conn, pointed, stored):
    for mode in MODES:
        results = search(conn, pointed, "quokka", k=3, mode=mode)
        assert 1 <= len(results) <= 3, mode


def test_vector_mode_finds_the_pointed_chunk_first(conn, pointed, stored):
    results = search(conn, pointed, "quokka", k=3, mode="vector")
    assert results[0].section_heading == "Zebras"
    assert results[0].vector_rank == 1 and results[0].keyword_rank is None
    assert results[0].score == pytest.approx(1.0, abs=1e-5)  # a real similarity


def test_keyword_mode_finds_the_word_match_only(conn, pointed, stored):
    results = search(conn, pointed, "quokka", k=3, mode="keyword")
    assert headings(results) == ["Quokka"]
    assert results[0].keyword_rank == 1 and results[0].vector_rank is None


def test_default_mode_is_hybrid(conn, pointed, stored):
    assert headings(search(conn, pointed, "quokka", k=2)) == headings(
        search(conn, pointed, "quokka", k=2, mode="hybrid")
    )


def test_unknown_mode_is_rejected(conn, pointed):
    with pytest.raises(ValueError, match="unknown mode"):
        search(conn, pointed, "quokka", mode="fuzzy")


def test_invalid_k_and_candidates_are_rejected(conn, pointed):
    with pytest.raises(ValueError):
        search(conn, pointed, "quokka", k=0)
    with pytest.raises(ValueError):
        search(conn, pointed, "quokka", mode="hybrid", candidates=0)


# --------------------------------------------------------------------- hybrid


def test_hybrid_includes_a_chunk_only_keyword_search_finds_and_one_only_vector_search_finds(conn, pointed, stored):
    # One candidate per method: vector offers only the pointed chunk, keyword only the quokka chunk.
    results = search(conn, pointed, "quokka", k=2, mode="hybrid", candidates=1)

    by_heading = {r.section_heading: r for r in results}
    assert set(by_heading) == {"Zebras", "Quokka"}
    assert by_heading["Zebras"].vector_rank == 1 and by_heading["Zebras"].keyword_rank is None
    assert by_heading["Quokka"].keyword_rank == 1 and by_heading["Quokka"].vector_rank is None


def test_hybrid_returns_at_most_k_and_scores_are_fused_not_similarities(conn, pointed, stored):
    results = search(conn, pointed, "quokka", k=3, mode="hybrid")
    assert len(results) == 3
    for r in results:
        assert r.fused_score == r.score
        assert 0 < r.fused_score <= 2 / 61 + 1e-9  # a ranking value, never near a similarity of 1
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_chunk_found_by_both_methods_ranks_first(conn, embedder, stored):
    # Pointing the vector query at the quokka chunk makes both methods agree on it.
    both = PointedEmbedder(embedder, KEYWORD_ONLY)
    results = search(conn, both, "quokka", k=3, mode="hybrid")
    assert results[0].section_heading == "Quokka"
    assert results[0].vector_rank == 1 and results[0].keyword_rank == 1
    assert results[0].fused_score == pytest.approx(2 / 61)


def test_keyword_ranks_help_a_chunk_that_vector_search_ranks_low(conn, pointed, stored):
    vector_ranked = search(conn, pointed, "quokka", k=8, mode="vector")
    hybrid_ranked = search(conn, pointed, "quokka", k=8, mode="hybrid")
    position = lambda results: headings(results).index("Quokka")  # noqa: E731
    assert position(hybrid_ranked) <= position(vector_ranked)
    assert position(hybrid_ranked) <= 1  # within the top two


def test_filters_apply_in_every_mode(conn, add_document, pointed, stored):
    add_document(conn, "Old PDF", "pdfs/old.pdf", [("quokka facts from a pdf", 4, None)], source_type="pdf",
                 doc_date=date(2020, 1, 1))
    for mode in MODES:
        only_pdf = search(conn, pointed, "quokka", k=5, mode=mode, filters=SearchFilters(source_types=["pdf"]))
        assert only_pdf and {r.path for r in only_pdf} == {"pdfs/old.pdf"}, mode
        none = search(conn, pointed, "quokka", k=5, mode=mode, filters=SearchFilters(date_from=date(2030, 1, 1)))
        assert none == [], mode


def test_empty_database_returns_nothing_in_every_mode(conn, embedder):
    for mode in MODES:
        assert search(conn, embedder, "anything at all", mode=mode) == []
