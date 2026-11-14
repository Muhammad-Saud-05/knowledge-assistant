from dataclasses import replace

import pytest

from src.retrieval.bm25 import Bm25Searcher, tokenize
from src.retrieval.hybrid import ALL_MODES, MODES, parse_mode, search
from src.retrieval.models import RetrievedChunk
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL, Reranker, build_text
from tests.test_hybrid import KEYWORD_ONLY, VECTOR_ONLY, PointedEmbedder  # noqa: F401  (fixtures below)


def make_chunk(cid, text="some text", title="Title", heading="Heading", page=3, **extra):
    return RetrievedChunk(cid, 7, title, f"docs/{cid}.md", page, heading, text, 0.5, vector_rank=cid, keyword_rank=None, **extra)


def scores_by_text(table):
    """A fake cross-encoder: the score of a pair is looked up by the passage text."""
    return lambda pairs: [table[passage] for _, passage in pairs]


# ------------------------------------------------------------------ Reranker


def test_orders_by_score_and_sets_rank_fields():
    chunks = [make_chunk(1, "a"), make_chunk(2, "b"), make_chunk(3, "c")]
    rr = Reranker(score_fn=scores_by_text({"a": 0.1, "b": 2.5, "c": -1.0}))
    out = rr.rerank("q", chunks, k=3)
    assert [c.chunk_id for c in out] == [2, 1, 3]
    assert [c.rerank_rank for c in out] == [1, 2, 3]
    assert [c.rerank_score for c in out] == [2.5, 0.1, -1.0]
    assert [c.score for c in out] == [2.5, 0.1, -1.0]  # the raw score, no sigmoid


def test_ties_keep_incoming_order():
    chunks = [make_chunk(5, "a"), make_chunk(3, "b"), make_chunk(9, "c")]
    rr = Reranker(score_fn=lambda pairs: [1.0] * len(pairs))
    assert [c.chunk_id for c in rr.rerank("q", chunks, 3)] == [5, 3, 9]
    assert [c.chunk_id for c in rr.rerank("q", chunks, 3)] == [5, 3, 9]


def test_empty_input_and_k_larger_than_list():
    rr = Reranker(score_fn=lambda pairs: [0.0] * len(pairs))
    assert rr.rerank("q", [], 5) == []
    assert len(rr.rerank("q", [make_chunk(1)], 10)) == 1


def test_k_is_cut_off_and_must_be_positive():
    chunks = [make_chunk(i, str(i)) for i in range(1, 5)]
    rr = Reranker(score_fn=lambda pairs: [float(p[1]) for p in pairs])
    assert [c.chunk_id for c in rr.rerank("q", chunks, 2)] == [4, 3]
    with pytest.raises(ValueError):
        rr.rerank("q", chunks, 0)


def test_other_metadata_is_preserved_and_input_not_modified():
    original = make_chunk(1, "a", fused_score=0.03, list_ranks=(1, None))
    snapshot = replace(original)
    out = Reranker(score_fn=lambda pairs: [4.0]).rerank("q", [original], 1)[0]
    assert (out.title, out.page_number, out.section_heading, out.path, out.document_id) == ("Title", 3, "Heading", "docs/1.md", 7)
    assert (out.vector_rank, out.keyword_rank, out.fused_score, out.list_ranks) == (1, None, 0.03, (1, None))
    assert original == snapshot and original.rerank_score is None


def test_input_styles_build_the_expected_text():
    chunk = make_chunk(1, "body", title="Doc", heading="Part")
    assert build_text(chunk, "plain") == "body"
    assert build_text(chunk, "with_context") == "Doc\nPart\nbody"
    assert build_text(make_chunk(1, "body", title=None, heading=None), "with_context") == "body"
    with pytest.raises(ValueError):
        build_text(chunk, "other")
    with pytest.raises(ValueError):
        Reranker(input_style="other")


def test_the_model_reads_the_style_text_with_the_query():
    seen = []

    def spy(pairs):
        seen.extend(pairs)
        return [0.0] * len(pairs)

    Reranker(input_style="with_context", score_fn=spy).rerank("my question", [make_chunk(1, "body", "Doc", "Part")], 1)
    assert seen == [("my question", "Doc\nPart\nbody")]


def test_cache_avoids_recomputing_and_is_keyed_by_model_style_and_query():
    calls = []

    def fn(pairs):
        calls.append(len(pairs))
        return [1.0] * len(pairs)

    cache = {}
    chunks = [make_chunk(1), make_chunk(2)]
    Reranker(score_fn=fn, cache=cache).rerank("q", chunks, 2)
    Reranker(score_fn=fn, cache=cache).rerank("q", chunks, 2)
    assert calls == [2]  # the second call was served from the cache
    Reranker(score_fn=fn, cache=cache).rerank("another query", chunks, 2)
    Reranker(score_fn=fn, cache=cache, input_style="with_context").rerank("q", chunks, 2)
    Reranker(model_name="other/model", score_fn=fn, cache=cache).rerank("q", chunks, 2)
    assert calls == [2, 2, 2, 2]


def test_wrong_number_of_scores_is_an_error():
    with pytest.raises(ValueError):
        Reranker(score_fn=lambda pairs: [1.0]).rerank("q", [make_chunk(1), make_chunk(2)], 2)


def test_model_name_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "x/y")
    assert Reranker().model_name == "x/y"
    monkeypatch.delenv("RERANKER_MODEL")
    assert Reranker().model_name == DEFAULT_RERANKER_MODEL


@pytest.mark.slow
def test_real_model_ranks_a_relevant_passage_above_an_irrelevant_one():
    rr = Reranker()
    good = make_chunk(1, "To start the server, run uvicorn main:app --reload in the project folder.")
    bad = make_chunk(2, "Quokkas are small marsupials that live on islands off Western Australia.")
    out = rr.rerank("How do I start the development server?", [bad, good], 2)
    assert [c.chunk_id for c in out] == [1, 2]
    assert out[0].rerank_score > out[1].rerank_score


# -------------------------------------------------------------- mode names


def test_mode_names():
    assert parse_mode("hybrid") == ("hybrid", False)
    assert parse_mode("hybrid-bm25+rerank") == ("hybrid-bm25", True)
    assert parse_mode("vector+rerank") == ("vector", True)
    assert MODES == ("vector", "keyword", "hybrid")  # existing callers see no new modes
    assert len(ALL_MODES) == 8
    with pytest.raises(ValueError, match="unknown mode"):
        parse_mode("rerank")


# ------------------------------------------------- search with a reranker


@pytest.fixture
def stored(conn, add_document):
    chunks = [(VECTOR_ONLY, None, "Zebras"), (KEYWORD_ONLY, None, "Quokka")]
    chunks += [(f"filler chunk number {i} about nothing in particular", None, f"Filler {i}") for i in range(6)]
    add_document(conn, "Notes", "notes/all.md", chunks)


@pytest.fixture
def pointed(embedder):
    return PointedEmbedder(embedder, VECTOR_ONLY)


def test_search_without_a_reranker_is_unchanged(conn, pointed, stored):
    for mode in MODES:
        results = search(conn, pointed, "quokka", k=3, mode=mode)
        assert all(r.rerank_score is None and r.rerank_rank is None for r in results)


def test_rerank_mode_reorders_the_first_stage(conn, pointed, stored):
    plain = search(conn, pointed, "quokka", k=8, mode="hybrid")
    assert plain[0].section_heading != "Filler 0"
    prefer = Reranker(score_fn=lambda pairs: [5.0 if "filler chunk number 0" in p else 0.0 for _, p in pairs])
    out = search(conn, pointed, "quokka", k=3, mode="hybrid+rerank", reranker=prefer, rerank_candidates=8)
    assert out[0].section_heading == "Filler 0" and out[0].rerank_rank == 1
    assert len(out) == 3
    assert out[0].vector_rank is not None or out[0].keyword_rank is not None  # first-stage ranks kept


def test_rerank_candidates_limit_what_the_reranker_sees(conn, pointed, stored):
    seen = []

    def spy(pairs):
        seen.append(len(pairs))
        return [0.0] * len(pairs)

    search(conn, pointed, "quokka", k=2, mode="vector+rerank", reranker=Reranker(score_fn=spy), rerank_candidates=4)
    assert seen == [4]


def test_rerank_mode_needs_a_reranker_and_plain_mode_refuses_one(conn, pointed, stored):
    with pytest.raises(ValueError, match="needs a reranker"):
        search(conn, pointed, "quokka", mode="hybrid+rerank")
    with pytest.raises(ValueError, match="does not use one"):
        search(conn, pointed, "quokka", mode="hybrid", reranker=Reranker(score_fn=lambda p: [0.0] * len(p)))


# ---------------------------------------------------------------- BM25 leg


def test_tokenize_lowercases_and_drops_stop_words():
    assert tokenize("How do I use the Quokka-API?") == ["use", "quokka", "api"]


def test_bm25_finds_word_matches_only_and_ranks_by_relevance(conn, stored):
    bm = Bm25Searcher(conn)
    results = bm.search("quokka", k=5)
    assert [r.section_heading for r in results] == ["Quokka"]
    assert results[0].keyword_rank == 1 and results[0].keyword_score == results[0].score > 0
    assert bm.search("the of and", k=5) == []  # only stop words
    assert bm.search("   ", k=5) == []


def test_bm25_respects_filters(conn, add_document, stored):
    from src.retrieval.filters import SearchFilters

    add_document(conn, "Other", "other/x.md", [("the quokka lives here too", None, "Other quokka")])
    bm = Bm25Searcher(conn)
    both = {r.section_heading for r in bm.search("quokka", k=5)}
    assert both == {"Quokka", "Other quokka"}
    only = bm.search("quokka", k=5, filters=SearchFilters(path_prefix="other/"))
    assert [r.section_heading for r in only] == ["Other quokka"]


def test_hybrid_bm25_mode_uses_the_bm25_leg(conn, pointed, stored):
    results = search(conn, pointed, "quokka", k=3, mode="hybrid-bm25", bm25=Bm25Searcher(conn))
    assert {"Zebras", "Quokka"} <= {r.section_heading for r in results}
    with pytest.raises(ValueError, match="Bm25Searcher"):
        search(conn, pointed, "quokka", mode="hybrid-bm25")
