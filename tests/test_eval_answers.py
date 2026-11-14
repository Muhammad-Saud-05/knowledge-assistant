import json

import pytest

from eval import run_answers as ra
from eval.questions import Expected, Question
from src.retrieval.models import RetrievedChunk
from src.synthesis.answer import NOT_FOUND, Answer, validate_answer

SECRET_LABEL = "evidence-phrase-that-must-not-leak"


def question(qid, category="paraphrase", answerable=True, holdout=False, headline=True):
    expected = (Expected("docs/a.md", (SECRET_LABEL,)),) if answerable else ()
    return Question(qid, category, f"question {qid}?", expected, holdout, headline=headline, difficulty="high_overlap" if answerable else None, lexical_overlap=0.5 if answerable else None)


def chunks(n=2):
    return [RetrievedChunk(i, 1, f"Title {i}", f"docs/{i}.md", None, f"Heading {i}", f"text of source {i} " * 40, 0.5) for i in range(1, n + 1)]


def answer(text, n_sources=2, first=1.5, total=4.0):
    return Answer("q", text, chunks(n_sources), validate_answer(text, n_sources), {"first_token": first, "total": total, "retrieval": 0.2})


def record(qid, text, answerable=True, **kw):
    return ra.make_record(question(qid, answerable=answerable), answer(text, **kw), "vector+rerank")


# --------------------------------------------------------------- the question set


def test_the_selection_is_the_dev_main_table_plus_the_dev_unanswerable_questions():
    qs = [
        question("q01", "exact_term"),
        question("q02", "paraphrase", holdout=True),  # holdout: never
        question("q03", "multi_doc"),
        question("q04", "filtered", headline=False),  # filtered: not in the main table
        question("q05", "unanswerable", answerable=False),
        question("q06", "unanswerable", answerable=False, holdout=True),
    ]
    # the fixture set is tiny, so bypass the v1/v2 set selector's view of added_in
    selected = [q.id for q in ra.select_questions(qs)]
    assert selected == ["q01", "q03", "q05"]


# ---------------------------------------------------------------------- records


def test_an_exact_refusal_is_recorded_as_one():
    r = record("q1", NOT_FOUND)
    assert r["exact_not_found"] and r["tolerant_not_found"] and not r["has_citation"]
    assert record("q1", "  " + NOT_FOUND + "\n")["exact_not_found"]  # surrounding whitespace only


def test_the_refusal_sentence_with_extra_text_is_not_an_exact_refusal():
    r = record("q1", NOT_FOUND + " But maybe it is in the docs [1].")
    assert not r["exact_not_found"] and r["has_citation"]


def test_citations_are_checked_against_the_sources_provided():
    ok = record("q1", "A claim [1]. Another [2].")
    assert ok["cited_numbers"] == [1, 2] and ok["has_citation"] and ok["all_citations_valid"] and ok["invalid_numbers"] == []
    bad = record("q1", "A claim [1]. Invented [7].")
    assert bad["has_citation"] and not bad["all_citations_valid"] and bad["invalid_numbers"] == [7]
    none = record("q1", "A claim with no citation.")
    assert not none["has_citation"] and none["all_citations_valid"]  # vacuously: the summary requires a citation too


def test_a_record_keeps_timings_and_source_previews_of_200_characters():
    r = record("q1", "A claim [1].", first=2.5, total=9.0)
    assert (r["first_token_seconds"], r["total_seconds"], r["retrieval_seconds"]) == (2.5, 9.0, 0.2)
    assert r["sources_provided"] == 2 and r["sources"][0]["number"] == 1
    assert r["sources"][0]["title"] == "Title 1" and r["sources"][0]["heading"] == "Heading 1"
    assert 0 < len(r["sources"][0]["preview"]) <= ra.PREVIEW_CHARS
    json.dumps(r)  # serialisable


# --------------------------------------------------------------------- summary


def make_set():
    return [
        record("q01", "Right [1]."),  # answerable, answered, cited
        record("q02", "Right but no citation."),  # answerable, answered, no citation
        record("q03", "Bad cite [9]."),  # answerable, answered, invalid citation
        record("q04", NOT_FOUND),  # answerable, wrongly refused
        record("q05", NOT_FOUND, answerable=False),  # unanswerable, correctly refused
        record("q06", "Made up [1].", answerable=False),  # unanswerable, answered with a citation
        record("q07", "Made up, no cite.", answerable=False),  # unanswerable, answered without one
    ]


def test_summary_counts():
    s = ra.summarize(make_set())
    assert s["n"] == 7
    assert s["unanswerable"] == {"n": 3, "refused": 1, "answered": 2, "answered_with_citation": 1, "answered_without_citation": 1}
    assert s["answerable"] == {"n": 4, "wrongly_refused": 1, "answered": 3}
    a = s["answered"]
    assert a["n"] == 5 and a["with_citation"] == 3 and a["all_citations_valid"] == 2  # q01 and q06; q03 has an invalid number
    assert (a["answerable_n"], a["answerable_with_citation"], a["answerable_all_valid"]) == (3, 2, 1)
    assert (a["unanswerable_n"], a["unanswerable_with_citation"], a["unanswerable_all_valid"]) == (2, 1, 1)
    assert a["invalid_numbers"] == 1
    assert s["first_token"][2] == 7 and s["total"][0] == 4.0


def test_rendered_summary_shows_counts_with_n_not_only_percentages():
    text = ra.render_summary(ra.summarize(make_set()))
    assert "1/3 (33.3%)" in text  # refused of the unanswerable
    assert "1/4 (25.0%)" in text  # wrongly refused of the answerable
    assert "3/5 (60.0%)" in text and "2/5 (40.0%)" in text
    assert "median" in text and "citation numbers outside the sources provided: 1" in text


def test_summary_handles_a_group_with_no_answers():
    s = ra.summarize([record("q1", NOT_FOUND)])
    text = ra.render_summary(s)
    assert "0/0" in text


# ----------------------------------------------------------- the grading sample


def records_for_sampling():
    rs = {f"q{i:02d}": record(f"q{i:02d}", "An answer [1].") for i in range(1, 31)}
    rs["q31"] = record("q31", NOT_FOUND, answerable=False)  # refused: not for grading
    rs["q32"] = record("q32", "Invented [1].", answerable=False)  # answered: must be graded
    rs["q33"] = record("q33", "Invented too.", answerable=False)
    return {i: dict(r, id=i, answerable=r["answerable"]) for i, r in rs.items()}


def test_the_sample_is_seeded_and_contains_every_answered_unanswerable_question():
    records = records_for_sampling()
    a = ra.grading_sample(records)
    b = ra.grading_sample(records)
    assert [r["id"] for r in a] == [r["id"] for r in b]  # deterministic
    ids = [r["id"] for r in a]
    assert ids == sorted(ids)
    answerable = [r for r in a if r["answerable"]]
    assert len(answerable) == 12 and len({r["id"] for r in answerable}) == 12
    assert {"q32", "q33"} <= set(ids) and "q31" not in ids  # the refused unanswerable one is not included
    assert len(a) == 14
    # a different seed would choose differently
    other = ra.grading_sample(records, seed=8)
    assert [r["id"] for r in other if r["answerable"]] != [r["id"] for r in answerable]


def test_to_grade_has_the_answer_the_sources_and_a_blank_grade_line_and_no_labels():
    sample = ra.grading_sample(records_for_sampling())
    text = ra.render_to_grade(sample)
    assert text.count("Grade: correct / partly / wrong / unsupported") == len(sample)
    assert "An answer [1]." in text and "[1] Title 1, Heading 1" in text and "[2] Title 2, Heading 2" in text
    assert "> text of source 1" in text
    assert SECRET_LABEL not in text  # evidence phrases and expected documents are never shown
    assert "graded by the assistant" in text


# ----------------------------------------------------------- running and resuming


def test_run_saves_each_answer_as_it_finishes_and_resumes_after_a_stop(tmp_path):
    qs = [question(f"q{i}") for i in range(1, 5)]
    calls = []

    def flaky(text):
        calls.append(text)
        if len(calls) == 3:
            raise KeyboardInterrupt  # the run is stopped during the third question
        return answer("Answer [1].")

    with pytest.raises(KeyboardInterrupt):
        ra.run_questions(qs, flaky, "vector+rerank", tmp_path)
    saved = ra.load_records(tmp_path / ra.ANSWERS_FILE)
    assert sorted(saved) == ["q1", "q2"]  # saved incrementally, nothing lost

    calls.clear()
    records = ra.run_questions(qs, lambda text: calls.append(text) or answer("Answer [1]."), "vector+rerank", tmp_path)
    assert calls == ["question q3?", "question q4?"]  # only the remaining ones were run
    assert sorted(records) == ["q1", "q2", "q3", "q4"]
    lines = (tmp_path / ra.ANSWERS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4


def test_a_torn_last_line_is_ignored_when_resuming(tmp_path):
    path = tmp_path / ra.ANSWERS_FILE
    ra.append_record(path, record("q1", "A [1]."))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "q2", "answ')  # interrupted mid-write
    assert sorted(ra.load_records(path)) == ["q1"]


def test_main_summary_only_works_from_saved_answers_and_writes_to_grade(tmp_path, capsys, monkeypatch):
    qs = [question("q01", "exact_term"), question("q02", "unanswerable", answerable=False)]
    monkeypatch.setattr(ra, "load_questions", lambda path=None: qs)
    monkeypatch.setattr(ra, "select_questions", lambda questions: qs)
    for q, text in ((qs[0], "An answer [1]."), (qs[1], NOT_FOUND)):
        ra.append_record(tmp_path / ra.ANSWERS_FILE, ra.make_record(q, answer(text), "vector+rerank"))
    assert ra.main(["--out", str(tmp_path), "--summary-only"]) == 0
    out = capsys.readouterr().out
    assert "correctly refused" in out and "1/1 (100.0%)" in out
    assert (tmp_path / ra.GRADE_FILE).exists()
