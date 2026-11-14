import math

import pytest

from eval.metrics import (
    aggregate,
    bootstrap_ci,
    content_words,
    count_text,
    judge_chunk,
    normalize,
    score_from_flags,
    score_ranking,
)
from eval.questions import (
    Expected,
    Question,
    difficulty_for,
    format_questions,
    parse_question,
    select_set,
    select_split,
)

DOC_A, DOC_B, DOC_ALT = "docs/a.md", "docs/b.md", "docs/alt.md"
CURLY_APOSTROPHE = chr(0x2019)
CURLY_QUOTES = chr(0x201C) + "hi" + chr(0x201D)


def expect(doc, *phrases):
    return Expected(doc=doc, evidence=tuple(phrases))


# ---------------------------------------------------------------- normalisation


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  Hello \n\t  WORLD  ") == "hello world"


def test_normalize_removes_markdown_emphasis_and_code_marks():
    assert normalize("a **bold** and `code` and ~~gone~~ and _it_") == "a bold and code and gone and it"


def test_normalize_makes_curly_quotes_and_dashes_plain():
    assert normalize(f"don{CURLY_APOSTROPHE}t {CURLY_QUOTES} a{chr(0x2013)}b") == "don't \"hi\" a-b"
    assert normalize("a" + chr(0x00A0) + "b") == "a b"


def test_normalize_applies_unicode_compatibility_forms():
    assert normalize(chr(0xFB01) + "le") == "file"  # the "fi" ligature


def test_underscores_are_dropped_on_both_sides_so_identifiers_still_match():
    assert normalize("max_age") == normalize("`max_age`") == "maxage"
    assert normalize("Defaults to `600`") in normalize("... Cache CORS responses. Defaults to `600`.")


def test_normalize_is_idempotent():
    text = "  **Some** `mixed` TEXT\n here_ "
    assert normalize(normalize(text)) == normalize(text)


def test_content_words_keeps_only_long_non_stop_words():
    assert content_words("How does the CACHE decide which entries to drop?") == {"cache", "decide", "entries", "drop"}
    assert content_words("a an of it is") == set()


# ---------------------------------------------------------------- judging chunks


def test_a_chunk_is_relevant_when_it_is_from_the_document_and_has_an_evidence_phrase():
    slots = ((expect(DOC_A, "refills at a steady rate"),),)
    assert judge_chunk(DOC_A, "A token bucket refills at a steady rate.", slots) == (0, True)
    assert judge_chunk(DOC_A, "A token bucket **refills** at a `steady` rate.", slots) == (0, True)  # emphasis is ignored
    # a link's URL in the middle of the phrase does break the match
    assert judge_chunk(DOC_A, "It refills at a [steady](http://x.example) rate.", slots) == (None, True)


def test_right_document_but_no_evidence_is_only_a_document_hit():
    slots = ((expect(DOC_A, "refills at a steady rate"),),)
    assert judge_chunk(DOC_A, "Something else entirely.", slots) == (None, True)


def test_evidence_belongs_to_its_own_document():
    slots = ((expect(DOC_A, "phrase one"),), (expect(DOC_B, "phrase two"),))
    assert judge_chunk(DOC_A, "this has phrase two", slots) == (None, True)  # B's phrase in A does not count
    assert judge_chunk(DOC_B, "this has phrase two", slots) == (1, True)


def test_an_unrelated_document_is_neither():
    slots = ((expect(DOC_A, "phrase"),),)
    assert judge_chunk("docs/other.md", "phrase", slots) == (None, False)


def test_an_also_acceptable_document_fills_the_same_slot():
    slots = ((expect(DOC_A, "first phrase"), expect(DOC_ALT, "alternative phrase")),)
    assert judge_chunk(DOC_ALT, "an alternative phrase here", slots) == (0, True)
    assert judge_chunk(DOC_ALT, "the first phrase here", slots) == (None, True)  # A's phrase does not count in ALT


# ----------------------------------------------------------- scoring a question


def test_hit_at_k_and_mrr_hand_computed():
    # relevant chunk at rank 4
    score = score_from_flags([None, None, None, 0, None], [False] * 5, n_slots=1)
    assert score.hit == {1: False, 3: False, 5: True}
    assert score.mrr10 == pytest.approx(0.25)
    assert score.first_relevant_rank == 4


def test_relevant_chunk_at_rank_one():
    score = score_from_flags([0, None, None], [True, False, False], n_slots=1)
    assert score.hit == {1: True, 3: True, 5: True} and score.mrr10 == 1.0


def test_mrr_stops_at_rank_10():
    matched = [None] * 10 + [0]  # the first relevant chunk is 11th
    score = score_from_flags(matched, [False] * 11, n_slots=1)
    assert score.mrr10 == 0.0 and score.first_relevant_rank is None and score.hit[5] is False
    at_ten = score_from_flags([None] * 9 + [0], [False] * 10, n_slots=1)
    assert at_ten.mrr10 == pytest.approx(0.1)


def test_nothing_relevant_scores_zero():
    score = score_from_flags([None] * 5, [False] * 5, n_slots=1)
    assert score.mrr10 == 0.0 and not any(score.hit.values()) and score.doc_hit5 is False


def test_doc_hit_is_the_lenient_measure():
    score = score_from_flags([None, None, None], [False, True, False], n_slots=1)
    assert score.doc_hit5 is True and score.hit[5] is False


def test_doc_hit_only_looks_at_the_top_five():
    score = score_from_flags([None] * 6, [False] * 5 + [True], n_slots=1)
    assert score.doc_hit5 is False


def test_multi_doc_coverage_and_hit_need_every_document():
    # slot 0 found at rank 1, slot 1 only at rank 4
    score = score_from_flags([0, None, None, 1, None], [True, False, False, True, False], n_slots=2)
    assert score.coverage == {1: 0.5, 3: 0.5, 5: 1.0}
    assert score.hit == {1: False, 3: False, 5: True}
    assert score.mrr10 == 1.0  # MRR is about the first relevant chunk of any slot


def test_multi_doc_one_document_found_twice_is_still_half_coverage():
    score = score_from_flags([0, 0, 0], [True] * 3, n_slots=2)
    assert score.coverage[5] == 0.5 and score.hit[5] is False


def test_multi_doc_never_hits_if_one_document_is_missing():
    score = score_from_flags([0, 0, None, None, None], [True] * 5, n_slots=2)
    assert not any(score.hit.values())


def test_scoring_a_question_without_slots_is_an_error():
    with pytest.raises(ValueError):
        score_from_flags([None], [False], n_slots=0)


def test_score_ranking_end_to_end_with_an_alternative_document():
    slots = ((expect(DOC_A, "phrase a"), expect(DOC_ALT, "phrase alt")),)
    ranking = [(DOC_B, "unrelated"), (DOC_A, "no evidence"), (DOC_ALT, "contains phrase alt"), (DOC_A, "phrase a")]
    score = score_ranking(ranking, slots)
    assert score.first_relevant_rank == 3 and score.mrr10 == pytest.approx(1 / 3)
    assert score.hit == {1: False, 3: True, 5: True} and score.doc_hit5 is True


# -------------------------------------------------------------------- aggregation


def test_aggregate_counts_and_means():
    scores = [
        score_from_flags([0, None], [True, False], 1),  # rank 1
        score_from_flags([None, None, 0], [False] * 3, 1),  # rank 3
        score_from_flags([None] * 5, [False] * 5, 1),  # miss
    ]
    result = aggregate(scores)
    assert (result["n"], result["hit1"], result["hit3"], result["hit5"]) == (3, 1, 2, 2)
    assert result["mrr10"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert result["doc_hit5"] == 1
    low, high = result["hit5_ci"]
    assert 0 <= low <= 2 / 3 <= high <= 1


def test_count_text_shows_counts_next_to_the_percentage():
    assert count_text(27, 34) == "27/34 (79.4%)"
    assert count_text(0, 0) == "0/0 (n/a)"


def test_bootstrap_is_reproducible_with_the_fixed_seed():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25]
    assert bootstrap_ci(values) == bootstrap_ci(values)
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values)  # the default seed is 7
    assert bootstrap_ci(values, seed=8) != bootstrap_ci(values)


def test_bootstrap_interval_contains_the_mean_and_is_wide_for_few_values():
    values = [1.0] * 5 + [0.0] * 5
    low, high = bootstrap_ci(values)
    assert low <= 0.5 <= high and high - low > 0.3


def test_bootstrap_of_identical_values_is_a_point():
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_bootstrap_of_nothing_is_nan():
    assert all(math.isnan(x) for x in bootstrap_ci([]))


# -------------------------------------------------------------------- questions


def raw_question(**overrides):
    raw = {
        "id": "q01",
        "category": "exact_term",
        "question": "What is it?",
        "expected": [{"doc": DOC_A, "evidence": ["phrase a"]}],
        "holdout": False,
    }
    raw.update(overrides)
    return raw


def test_parse_question_reads_the_new_fields():
    q = parse_question(
        raw_question(
            also_acceptable=[{"doc": DOC_ALT, "evidence": ["phrase alt"]}],
            headline=False,
            lexical_overlap=0.3,
            difficulty="high_overlap",
        )
    )
    assert isinstance(q, Question)
    assert q.also_acceptable == (Expected(DOC_ALT, ("phrase alt",)),)
    assert (q.headline, q.lexical_overlap, q.difficulty) == (False, 0.3, "high_overlap")


def test_defaults_for_optional_fields():
    q = parse_question(raw_question())
    assert (q.headline, q.also_acceptable, q.lexical_overlap, q.difficulty, q.filters) == (True, (), None, None, {})


def test_slots_single_document_with_alternatives_is_one_slot():
    q = parse_question(raw_question(also_acceptable=[{"doc": DOC_ALT, "evidence": ["x"]}]))
    assert len(q.slots) == 1 and [e.doc for e in q.slots[0]] == [DOC_A, DOC_ALT]
    assert q.acceptable_docs == (DOC_A, DOC_ALT)


def test_slots_multi_document_is_one_slot_per_document():
    q = parse_question(
        raw_question(category="multi_doc", expected=[{"doc": DOC_A, "evidence": ["x"]}, {"doc": DOC_B, "evidence": ["y"]}])
    )
    assert [[e.doc for e in slot] for slot in q.slots] == [[DOC_A], [DOC_B]]


def test_unanswerable_question_has_no_slots():
    q = parse_question(raw_question(category="unanswerable", expected=[]))
    assert q.slots == () and not q.answerable


@pytest.mark.parametrize("bad", [{}, {"id": "q01"}, "text", {**{"id": "q1"}, "expected": "nope"}])
def test_parse_question_rejects_malformed_items(bad):
    with pytest.raises(ValueError):
        parse_question(bad)


def test_select_split():
    questions = [parse_question(raw_question(id="q01", holdout=False)), parse_question(raw_question(id="q02", holdout=True))]
    assert [q.id for q in select_split(questions, "dev")] == ["q01"]
    assert [q.id for q in select_split(questions, "holdout")] == ["q02"]
    assert [q.id for q in select_split(questions, "all")] == ["q01", "q02"]
    with pytest.raises(ValueError):
        select_split(questions, "test")


@pytest.mark.parametrize("overlap, expected", [(0.0, "low_overlap"), (0.25, "low_overlap"), (0.26, "high_overlap"), (1.0, "high_overlap")])
def test_difficulty_threshold(overlap, expected):
    assert difficulty_for(overlap) == expected


def test_format_questions_round_trips_and_keeps_the_field_order():
    import json

    items = [
        raw_question(difficulty="high_overlap", lexical_overlap=0.5, notes="n", also_acceptable=[{"doc": DOC_ALT, "evidence": ["z"]}], headline=True),
        raw_question(id="q02", category="multi_doc", expected=[{"doc": DOC_A, "evidence": ["x"]}, {"doc": DOC_B, "evidence": ["y"]}], filters={"path_prefix": "docs/"}),
    ]
    text = format_questions(items)
    assert json.loads(text) == items
    first = [line.split('"')[1] for line in text.splitlines() if line.startswith('    "')][:8]
    assert first == ["id", "category", "question", "expected", "also_acceptable", "headline", "holdout", "lexical_overlap"]


def test_format_questions_rejects_unknown_fields():
    with pytest.raises(ValueError):
        format_questions([raw_question(surprise=1)])


def test_added_in_defaults_to_v1_and_is_read_when_present():
    assert parse_question(raw_question()).added_in == "v1"
    assert parse_question(raw_question(added_in="v2")).added_in == "v2"


def test_select_set_keeps_v1_questions_only_or_everything():
    questions = [parse_question(raw_question(id="q01")), parse_question(raw_question(id="q02", added_in="v2"))]
    assert [q.id for q in select_set(questions, "v1")] == ["q01"]
    assert [q.id for q in select_set(questions, "v2")] == ["q01", "q02"]
    with pytest.raises(ValueError):
        select_set(questions, "v3")


def test_format_questions_writes_added_in_right_after_the_category():
    text = format_questions([raw_question(added_in="v2")])
    keys = [line.split('"')[1] for line in text.splitlines() if line.startswith('    "')]
    assert keys[:4] == ["id", "category", "added_in", "question"]
