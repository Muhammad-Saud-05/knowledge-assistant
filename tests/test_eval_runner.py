import json
from datetime import datetime, timezone

import pytest

from eval import run_eval
from eval.questions import Expected, Question, format_questions
from eval.report import (
    default_pairs,
    set_comparison_table,
    difficulty_table,
    filtered_table,
    headline_questions,
    main_tables,
    missed_lists,
    paired_table,
    results_from_json,
    results_to_json,
    score_run,
    top1_tables,
    win_loss_lists,
)
from eval.run_eval import (
    EvalError,
    check_documents,
    check_split_allowed,
    parse_modes,
    result_filename,
    run_questions,
)
from eval.validate_questions import DocInfo
from src.retrieval.models import RetrievedChunk

A, B, C = "docs/a.md", "docs/b.md", "docs/c.md"
SECRET = "confidential-zebra-text"


def question(qid, category, expected=(), *, filters=None, headline=True, difficulty="high_overlap", holdout=False):
    return Question(
        id=qid, category=category, question=f"question {qid}", expected=tuple(expected), holdout=holdout,
        filters=filters or {}, headline=headline, difficulty=difficulty if expected else None,
        lexical_overlap=0.5 if expected else None,
    )


QUESTIONS = [
    question("q01", "exact_term", [Expected(A, ("alpha phrase",))]),
    question("q02", "paraphrase", [Expected(B, ("beta phrase",))], difficulty="low_overlap"),
    question("q03", "multi_doc", [Expected(A, ("alpha phrase",)), Expected(B, ("beta phrase",))]),
    question("q04", "unanswerable"),
    question("q05", "filtered", [Expected(A, ("alpha phrase",))], filters={"path_prefix": "docs/a"}, headline=False),
]


def chunk(cid, path, text, score=0.5, vector_rank=None, keyword_rank=None):
    return RetrievedChunk(cid, 1, "T", path, None, None, text, score, vector_rank=vector_rank, keyword_rank=keyword_rank)


GOOD_A = lambda cid=1, score=0.9: chunk(cid, A, f"This has the alpha phrase. {SECRET}", score)  # noqa: E731
GOOD_B = lambda cid=2, score=0.9: chunk(cid, B, f"This has the beta phrase. {SECRET}", score)  # noqa: E731
JUNK = lambda cid=9, path=C, score=0.1: chunk(cid, path, f"nothing useful {SECRET}", score)  # noqa: E731

# What each fake mode returns for each question (ranked best first).
SCRIPT = {
    "vector": {
        "q01": [GOOD_A(), JUNK()],
        "q02": [JUNK(10), JUNK(11), GOOD_B(12), JUNK(13)],  # relevant at rank 3
        "q03": [GOOD_A(), JUNK(14), JUNK(15)],  # only one of the two documents
        "q04": [JUNK(16, score=0.42)],
        "q05": [GOOD_A(17), JUNK(18, path=B)],  # the second result violates the filter
    },
    "keyword": {
        "q01": [JUNK(19), JUNK(20)],  # miss
        "q02": [GOOD_B()],
        "q03": [GOOD_A(), GOOD_B()],  # both documents, so hit@3 but not hit@1
        "q04": [JUNK(21, score=0.9)],
        "q05": [GOOD_A(22)],
    },
    "hybrid": {
        "q01": [GOOD_A()],
        "q02": [GOOD_B()],
        "q03": [GOOD_A(), GOOD_B()],
        "q04": [JUNK(23, score=0.03)],
        "q05": [GOOD_A(24)],
    },
}
CORPUS = {
    A: DocInfo("markdown", None, []),
    B: DocInfo("markdown", None, []),
    C: DocInfo("markdown", None, []),
}
MODES = ["vector", "keyword", "hybrid"]


def fake_search(q, mode):
    return SCRIPT[mode][q.id]


@pytest.fixture
def results():
    return run_questions(QUESTIONS, MODES, fake_search, CORPUS)


# ----------------------------------------------------------------- running


def test_run_records_ranked_judgements(results):
    run = results["vector"]["q02"]
    assert [e.slot for e in run.entries] == [None, None, 0, None]
    assert [e.doc_ok for e in run.entries] == [False, False, True, False]
    assert run.seconds >= 0 and run.top1_score == 0.1


def test_run_judges_multi_doc_slots_and_unanswerable(results):
    assert [e.slot for e in results["keyword"]["q03"].entries] == [0, 1]
    assert [e.slot for e in results["vector"]["q04"].entries] == [None]  # nothing is relevant to an unanswerable question
    assert results["vector"]["q04"].top1_score == 0.42


def test_run_checks_filters_against_the_stored_document(results):
    flags = [e.filter_ok for e in results["vector"]["q05"].entries]
    assert flags == [True, False]  # docs/b.md does not start with docs/a
    assert results["vector"]["q01"].entries[0].filter_ok is None  # no filter on that question


def test_run_keeps_only_the_top_ten():
    many = {"vector": {q.id: [JUNK(i) for i in range(30)] for q in QUESTIONS}}
    out = run_questions(QUESTIONS[:1], ["vector"], lambda q, m: many[m][q.id], CORPUS)
    assert len(out["vector"]["q01"].entries) == 10


def test_scores_match_hand_computation(results):
    q = {x.id: x for x in QUESTIONS}
    vector = lambda m, i: score_run(q[i], results[m][i])  # noqa: E731
    assert vector("vector", "q01").hit == {1: True, 3: True, 5: True}
    assert vector("vector", "q02").hit == {1: False, 3: True, 5: True} and vector("vector", "q02").mrr10 == pytest.approx(1 / 3)
    assert vector("vector", "q03").hit[5] is False and vector("vector", "q03").coverage[5] == 0.5
    assert vector("keyword", "q03").hit == {1: False, 3: True, 5: True}
    assert vector("keyword", "q01").hit[5] is False and vector("keyword", "q01").mrr10 == 0.0
    assert score_run(q["q04"], results["vector"]["q04"]) is None


# ----------------------------------------------------------------- tables


def test_main_table_covers_only_headline_questions(results):
    assert [q.id for q in headline_questions(QUESTIONS)] == ["q01", "q02", "q03"]
    text = main_tables(QUESTIONS, results, MODES)
    first_table = text.split("\n\n")[0]
    # hit@1: hybrid hits q01 and q02 but needs BOTH documents of q03 in the top 1 (2/3); vector only q01 (1/3)
    assert "| hybrid | 2/3 (66.7%) | 3/3 (100.0%) | 3/3 (100.0%) |" in first_table
    assert "| vector | 1/3 (33.3%) |" in first_table
    assert "multi_doc coverage@5" in text and "1.00 (n=1)" in text  # hybrid covers both documents
    assert "hit@5 95% CI" in text and "median ms/question" in text


def test_hit_at_5_counts_in_the_main_table(results):
    first_table = main_tables(QUESTIONS, results, MODES).split("\n\n")[0]
    row = {line.split("|")[1].strip(): line for line in first_table.splitlines() if line.startswith(("| vector", "| keyword", "| hybrid"))}
    assert "2/3 (66.7%)" in row["vector"] and "2/3 (66.7%)" in row["keyword"] and "3/3 (100.0%)" in row["hybrid"]


def test_difficulty_table_shows_n_per_bucket(results):
    text = difficulty_table(QUESTIONS, results, MODES)
    assert "| low_overlap | n=1 |" in text and "| high_overlap | n=2 |" in text


def test_paired_table_matches_hand_computed_differences(results):
    text = paired_table(QUESTIONS, results, default_pairs(MODES))
    assert "hybrid - vector" in text and "hybrid - keyword" in text
    # hit@5: hybrid hits all 3; vector misses q03 and keyword misses q01 -> +0.333 each
    lines = [l for l in text.splitlines() if "hit@5" in l]
    assert all("+0.333" in l and "1 better / 0 worse / 2 same" in l for l in lines)
    mrr_vs_vector = next(l for l in text.splitlines() if "hybrid - vector" in l and "MRR@10" in l)
    assert f"{(1 - 1) / 3 + (1 - 1 / 3) / 3 + 0:+.3f}" in mrr_vs_vector  # mean of (0, 2/3, 0) = +0.222


def test_default_pairs_compare_hybrid_with_each_other_mode():
    assert default_pairs(["vector", "keyword", "hybrid"]) == [("hybrid", "vector"), ("hybrid", "keyword")]
    assert default_pairs(["vector", "keyword"]) == [("keyword", "vector")]


def test_win_loss_lists(results):
    text = win_loss_lists(QUESTIONS, results, [("hybrid", "vector"), ("hybrid", "keyword")])
    assert "hybrid hits, vector misses (1): q03 (multi_doc)" in text
    assert "hybrid hits, keyword misses (1): q01 (exact_term)" in text


def test_missed_lists_show_only_id_and_expected_documents(results):
    text = missed_lists(QUESTIONS, results, MODES)
    assert "**vector: 1 of 3 missed at k=5**" in text and "- q03: a.md, b.md" in text
    assert "**hybrid: 0 of 3 missed at k=5**" in text
    assert SECRET not in text


def test_filtered_table_counts_filter_violations(results):
    text = filtered_table(QUESTIONS, results, MODES)
    row = next(l for l in text.splitlines() if l.startswith("| vector"))
    assert row.rstrip().endswith("| 1 |")  # one vector result violates the filter
    hybrid = next(l for l in text.splitlines() if l.startswith("| hybrid"))
    assert hybrid.rstrip().endswith("| 0 |")
    assert "path_prefix=docs/a" in text


def test_top1_table_puts_unanswerable_next_to_answerable(results):
    text = top1_tables(QUESTIONS, results, MODES)
    assert "(n=4)" in text and "(n=1)" in text  # four answerable questions, one unanswerable
    assert "| q04 | 0.420 | 0.900 | 0.030 |" in text


# ---------------------------------------------------------------- JSON


def test_json_round_trip_and_no_chunk_text(results):
    meta = {"split": "dev"}
    data = results_to_json(QUESTIONS, results, meta)
    text = json.dumps(data)
    assert SECRET not in text and "alpha phrase" not in text and "beta phrase" not in text
    assert '"text"' not in text
    back = results_from_json(json.loads(text))
    for mode in MODES:
        for q in QUESTIONS:
            assert back[mode][q.id].entries == results[mode][q.id].entries
    item = next(i for i in data["questions"] if i["id"] == "q04")
    assert item["answerable"] is False and item["runs"]["keyword"]["top1_score"] == 0.9


def test_json_stores_top1_scores_for_answerable_and_unanswerable(results):
    data = results_to_json(QUESTIONS, results, {})
    tops = {i["id"]: i["runs"]["hybrid"]["top1_score"] for i in data["questions"]}
    assert tops["q01"] == 0.9 and tops["q04"] == 0.03


# ----------------------------------------------------- guards and options


def test_missing_documents_fail_with_a_clear_message():
    with pytest.raises(EvalError) as error:
        check_documents(QUESTIONS, {A: CORPUS[A]})
    assert "docs/b.md (needed by q02, q03)" in str(error.value)
    check_documents(QUESTIONS, CORPUS)  # all present: no error


def test_acceptable_documents_must_exist_too():
    q = Question("q01", "exact_term", "x", (Expected(A, ("a",)),), False, also_acceptable=(Expected("docs/gone.md", ("b",)),))
    with pytest.raises(EvalError, match="docs/gone.md"):
        check_documents([q], CORPUS)


def test_parse_modes():
    assert parse_modes("vector, hybrid") == ["vector", "hybrid"]
    for bad in ("", "vector,fuzzy", "vector,vector"):
        with pytest.raises(EvalError):
            parse_modes(bad)


def test_the_holdout_split_is_locked_unless_confirmed():
    check_split_allowed("dev", False)
    for split in ("holdout", "all"):
        with pytest.raises(EvalError, match="confirm-holdout"):
            check_split_allowed(split, False)
        check_split_allowed(split, True)


def test_main_refuses_the_holdout_without_confirmation(capsys):
    assert run_eval.main(["--split", "holdout"]) == 2
    assert "--confirm-holdout" in capsys.readouterr().err
    assert run_eval.main(["--split", "all"]) == 2


def test_result_filename_has_a_utc_timestamp_and_the_modes():
    when = datetime(2026, 11, 14, 9, 30, 5, tzinfo=timezone.utc)
    assert result_filename(["vector", "hybrid+rerank"], "dev", when) == "eval_20261114T093005Z_vector-hybrid_rerank_dev.json"


def test_from_json_prints_the_tables_without_a_database(results, tmp_path, capsys):
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(
        format_questions(
            [
                {"id": q.id, "category": q.category, "question": q.question,
                 "expected": [{"doc": e.doc, "evidence": list(e.evidence)} for e in q.expected],
                 **({"filters": q.filters} if q.filters else {}),
                 "holdout": q.holdout, "headline": q.headline,
                 **({"lexical_overlap": 0.5, "difficulty": q.difficulty} if q.expected else {})}
                for q in QUESTIONS
            ]
        ),
        encoding="utf-8",
    )
    saved = tmp_path / "result.json"
    saved.write_text(json.dumps(results_to_json(QUESTIONS, results, {})), encoding="utf-8")

    assert run_eval.main(["--from-json", str(saved), "--questions", str(questions_file), "--split", "all", "--confirm-holdout"]) == 0
    out = capsys.readouterr().out
    assert "## Main results" in out and "## Paired differences" in out and "hybrid - vector" in out


# ------------------------------------------------------------- v1 against v2


def test_set_comparison_shows_v1_v2_and_the_additions_side_by_side(results):
    from dataclasses import replace

    marked = [replace(q, added_in="v2") if q.id == "q03" else q for q in QUESTIONS]  # q03 is an addition
    text = set_comparison_table(marked, results, MODES)
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith(("| vector", "| keyword", "| hybrid"))}
    # v1 = q01, q02 (hybrid hits both); v2 = q01..q03 (hybrid hits all three); additions only = q03
    assert "v1 (n=2) hit@5" in text and "v2 (v1 + additions) (n=3) hit@5" in text and "additions only (n=1) hit@5" in text
    assert "| 2/2 (100.0%) | 1.000 | 3/3 (100.0%) | 1.000 | 1/1 (100.0%) | 1.000 |" in rows["hybrid"]
    # vector hits q01 and q02 within the top 5 but misses the multi-document q03
    assert "| 2/2 (100.0%) | 0.667 | 2/3 (66.7%) | 0.778 | 0/1 (0.0%) | 1.000 |" in rows["vector"]


def test_set_comparison_is_part_of_the_report_only_when_there_are_additions(results):
    from dataclasses import replace

    from eval.report import render_report

    assert "Question set v1 next to v2" not in render_report(QUESTIONS, results, MODES)
    marked = [replace(q, added_in="v2") if q.id == "q03" else q for q in QUESTIONS]
    assert "Question set v1 next to v2" in render_report(marked, results, MODES)


def test_result_filename_marks_v2_but_keeps_v1_names():
    when = datetime(2026, 11, 14, 9, 30, 5, tzinfo=timezone.utc)
    assert result_filename(["vector"], "dev", when) == "eval_20261114T093005Z_vector_dev.json"
    assert result_filename(["vector"], "dev", when, "v1") == "eval_20261114T093005Z_vector_dev.json"
    assert result_filename(["vector", "hybrid"], "dev", when, "v2") == "eval_20261114T093005Z_v2_vector-hybrid_dev.json"
