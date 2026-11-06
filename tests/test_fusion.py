import pytest

from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import RetrievedChunk


def chunk(chunk_id, **fields):
    return RetrievedChunk(chunk_id, 1, "Title", "path.md", None, None, f"text {chunk_id}", 0.0, **fields)


def ids(results):
    return [c.chunk_id for c in results]


def test_chunk_in_both_lists_outranks_chunks_in_one_list():
    vector = [chunk(1), chunk(2), chunk(3)]
    keyword = [chunk(9), chunk(3)]  # chunk 3 is third in one list and second in the other
    fused = reciprocal_rank_fusion([vector, keyword])
    assert fused[0].chunk_id == 3
    assert set(ids(fused)) == {1, 2, 3, 9}


def test_scores_match_a_hand_calculation():
    # k = 60: score = sum of 1 / (60 + rank) over the lists the chunk appears in.
    vector = [chunk(1), chunk(2), chunk(3)]
    keyword = [chunk(3), chunk(1)]
    fused = {c.chunk_id: c for c in reciprocal_rank_fusion([vector, keyword], k=60)}
    assert fused[1].fused_score == pytest.approx(1 / 61 + 1 / 62)  # rank 1 and rank 2
    assert fused[3].fused_score == pytest.approx(1 / 63 + 1 / 61)  # rank 3 and rank 1
    assert fused[2].fused_score == pytest.approx(1 / 62)  # rank 2, one list only
    assert ids(reciprocal_rank_fusion([vector, keyword])) == [1, 3, 2]


def test_the_constant_k_changes_the_scores():
    lists = [[chunk(1), chunk(2)]]
    by_k0 = {c.chunk_id: c.fused_score for c in reciprocal_rank_fusion(lists, k=0)}
    assert by_k0[1] == pytest.approx(1.0) and by_k0[2] == pytest.approx(0.5)
    by_k10 = {c.chunk_id: c.fused_score for c in reciprocal_rank_fusion(lists, k=10)}
    assert by_k10[1] == pytest.approx(1 / 11) and by_k10[2] == pytest.approx(1 / 12)


def test_score_field_is_the_fused_score():
    fused = reciprocal_rank_fusion([[chunk(1)], [chunk(1)]])[0]
    assert fused.score == fused.fused_score == pytest.approx(2 / 61)


def test_chunks_are_de_duplicated_by_id():
    fused = reciprocal_rank_fusion([[chunk(1), chunk(2)], [chunk(2), chunk(1)], [chunk(1)]])
    assert sorted(ids(fused)) == [1, 2]


def test_a_chunk_repeated_inside_one_list_counts_once():
    fused = reciprocal_rank_fusion([[chunk(1), chunk(1), chunk(2)]])
    scores = {c.chunk_id: c.fused_score for c in fused}
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == pytest.approx(1 / 62)  # the duplicate did not push chunk 2 down


def test_ranks_from_every_list_are_kept():
    fused = {c.chunk_id: c for c in reciprocal_rank_fusion([[chunk(1), chunk(2)], [chunk(2)], [chunk(3), chunk(2)]])}
    assert fused[1].list_ranks == (1, None, None)
    assert fused[2].list_ranks == (2, 1, 2)
    assert fused[3].list_ranks == (None, None, 1)


def test_vector_and_keyword_details_from_the_inputs_are_merged():
    vector = [chunk(1, vector_rank=1, vector_score=0.9)]
    keyword = [chunk(1, keyword_rank=4, keyword_score=0.2), chunk(2, keyword_rank=1, keyword_score=0.5)]
    fused = {c.chunk_id: c for c in reciprocal_rank_fusion([vector, keyword])}
    assert (fused[1].vector_rank, fused[1].vector_score) == (1, 0.9)
    assert (fused[1].keyword_rank, fused[1].keyword_score) == (4, 0.2)
    assert fused[2].vector_rank is None and fused[2].keyword_rank == 1


def test_input_chunks_are_not_modified():
    original = chunk(1)
    reciprocal_rank_fusion([[original], [original]])
    assert original.fused_score is None and original.list_ranks is None and original.score == 0.0


def test_ties_are_broken_by_best_rank_then_chunk_id():
    # Chunk 7 is first in one list and chunk 5 is first in the other: equal scores and
    # equal best rank, so the lower chunk_id goes first whatever the list order.
    assert ids(reciprocal_rank_fusion([[chunk(7)], [chunk(5)]])) == [5, 7]
    assert ids(reciprocal_rank_fusion([[chunk(5)], [chunk(7)]])) == [5, 7]

    # A tie with two lists each: chunk 2 is rank 1 then 3, chunk 1 is rank 3 then 1.
    # Both sum 1/61 + 1/63 and both have best rank 1, so chunk_id decides.
    first = [chunk(2), chunk(8), chunk(1)]
    second = [chunk(1), chunk(9), chunk(2)]
    fused = reciprocal_rank_fusion([first, second])
    assert ids(fused)[:2] == [1, 2]
    assert fused[0].fused_score == fused[1].fused_score


def test_tie_prefers_the_better_single_rank_even_with_a_higher_id():
    # With k=0: chunk 9 is rank 2 in one list -> 1/2.
    # Chunk 1 is rank 6 in one list and rank 3 in the other -> 1/6 + 1/3 = 1/2.
    # Exactly tied, so the better single rank (2 beats 3) goes first, not the lower id.
    first = [chunk(21), chunk(9), chunk(22), chunk(23), chunk(24), chunk(1)]
    second = [chunk(25), chunk(26), chunk(1)]
    fused = {c.chunk_id: c for c in reciprocal_rank_fusion([first, second], k=0)}
    assert fused[9].fused_score == fused[1].fused_score == pytest.approx(0.5)
    order = ids(reciprocal_rank_fusion([first, second], k=0))
    assert order.index(9) < order.index(1)


def test_result_is_the_same_every_time():
    lists = [[chunk(i) for i in (4, 2, 9, 1)], [chunk(i) for i in (9, 3, 4)]]
    first = ids(reciprocal_rank_fusion(lists))
    for _ in range(5):
        assert ids(reciprocal_rank_fusion(lists)) == first


def test_limit_returns_only_the_top_results():
    lists = [[chunk(i) for i in range(1, 11)]]
    assert ids(reciprocal_rank_fusion(lists, limit=3)) == [1, 2, 3]
    assert len(reciprocal_rank_fusion(lists, limit=50)) == 10


def test_a_single_list_passes_through_in_order():
    original = [chunk(5), chunk(2), chunk(8), chunk(1)]
    assert ids(reciprocal_rank_fusion([original])) == [5, 2, 8, 1]


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert ids(reciprocal_rank_fusion([[], [chunk(1)]])) == [1]


def test_negative_k_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[chunk(1)]], k=-1)
