import copy
import json
from datetime import date

import pytest

from eval.questions import CATEGORIES, EXPECTED_COUNTS, format_questions
from eval.validate_questions import (
    DocInfo,
    doc_passes_filters,
    lexical_overlaps,
    to_search_filters,
    update_overlap,
    validate,
)
from eval.questions import parse_question

# A tiny fake corpus and a valid 40-question file built from it, so each rule can be broken on purpose.


def make_corpus():
    corpus = {}
    for i in range(24):
        corpus[f"data/corpus/docs/doc{i:02d}.md"] = DocInfo(
            "markdown", date(2026, 11, 14), [f"Some filler. Evidence number {i} appears here. More filler about topic{chr(97 + i % 26)}."]
        )
    corpus["data/corpus/docs/doc00.md"].source_type = "pdf"
    corpus["sample_corpus/notes.md"] = DocInfo(
        "markdown", date(2026, 9, 14), ["Notes with sample evidence marker for the sample document."]
    )
    corpus["data/corpus/docs/alt.md"] = DocInfo("markdown", date(2026, 11, 14), ["An alternative passage with alternative evidence marker."])
    return corpus


def entry(i):
    return {"doc": f"data/corpus/docs/doc{i:02d}.md", "evidence": [f"Evidence number {i} appears here"]}


def make_questions():
    items = []

    def add(category, **fields):
        position = sum(1 for q in items if q["category"] == category) + 1
        qid = f"q{len(items) + 1:02d}"
        item = {
            "id": qid,
            "category": category,
            "question": f"Zebra question number {len(items) + 1} about quokka{chr(96 + position)}?",
            "expected": [],
            "holdout": position % 4 == 0,
        }
        item.update(fields)
        items.append(item)

    for i in range(10):
        add("exact_term", expected=[entry(i) if i else {"doc": "sample_corpus/notes.md", "evidence": ["sample evidence marker"]}])
    for i in range(10, 22):
        add("paraphrase", expected=[entry(i)])
    for i in range(8):
        add("multi_doc", expected=[entry(i + 1), entry(i + 11)])
    for _ in range(6):
        add("unanswerable")
    for i in (1, 2, 3, 4):
        add("filtered", expected=[entry(i)], filters={"path_prefix": "data/corpus/docs/"}, headline=False)
    return items


@pytest.fixture
def corpus():
    return make_corpus()


@pytest.fixture
def items():
    return make_questions()


def problems(items, corpus, **kwargs):
    return validate(copy.deepcopy(items), corpus, require_overlap=kwargs.pop("require_overlap", False))


def find(items, qid):
    return next(q for q in items if q["id"] == qid)


def test_a_valid_file_has_no_problems(items, corpus):
    assert problems(items, corpus) == []
    assert Counter_categories(items) == EXPECTED_COUNTS


def Counter_categories(items):
    return {c: sum(q["category"] == c for q in items) for c in CATEGORIES}


# ------------------------------------------------------------------------ schema


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda i: i[0].pop("question"), "missing field 'question'"),
        (lambda i: i[0].update(id="x1"), "id must look like q01"),
        (lambda i: i[0].update(category="mystery"), "category must be one of"),
        (lambda i: i[0].update(holdout="yes"), "holdout must be true or false"),
        (lambda i: i[0].update(headline="no"), "headline must be true or false"),
        (lambda i: i[0].update(surprise=1), "unknown field"),
        (lambda i: i[0].update(lexical_overlap=2), "between 0 and 1"),
        (lambda i: i[0].update(difficulty="medium"), "difficulty must be one of"),
        (lambda i: i[0]["expected"][0].update(evidence=[]), "1 to 3 phrases"),
        (lambda i: i[0]["expected"][0].update(evidence=["a", "b", "c", "d"]), "1 to 3 phrases"),
        (lambda i: i[0]["expected"][0].update(evidence=["one two three four five six seven eight nine ten eleven"]), "max 10"),
        (lambda i: i[0].update(also_acceptable="nope"), "also_acceptable must be a list"),
        (lambda i: i[0].update(filters={"colour": "red"}), "unknown filter field"),
        (lambda i: i[0].update(filters={"date_from": "01/02/2026"}), "must be YYYY-MM-DD"),
    ],
)
def test_schema_problems_are_caught(items, corpus, mutate, message):
    mutate(items)
    assert any(message in p for p in problems(items, corpus)), problems(items, corpus)


def test_a_non_object_item_is_reported(items, corpus):
    items.append("not an object")
    assert any("not a JSON object" in p for p in problems(items, corpus))


# ------------------------------------------------------------------ file structure


def test_duplicate_ids_and_wrong_counts(items, corpus):
    items[1]["id"] = items[0]["id"]
    assert any("duplicate id q01" in p for p in problems(items, corpus))
    extra = copy.deepcopy(items[0]) | {"id": "q41", "question": "A different extra question entirely?"}
    items.append(extra)
    found = problems(items, corpus)
    assert any("category exact_term: expected 10" in p for p in found)
    assert any("in total, found 41" in p for p in found)


def test_duplicate_question_text(items, corpus):
    items[1]["question"] = items[0]["question"].upper()
    assert any("duplicates q01" in p for p in problems(items, corpus))


def test_unanswerable_must_have_no_expected_and_no_overlap_fields(items, corpus):
    unanswerable = next(q for q in items if q["category"] == "unanswerable")
    unanswerable["expected"] = [entry(5)]
    assert any("must have an empty 'expected'" in p for p in problems(items, corpus))
    unanswerable["expected"] = []
    unanswerable["lexical_overlap"] = 0.3
    assert any("no lexical_overlap or difficulty" in p for p in problems(items, corpus))


def test_expected_document_rules(items, corpus):
    find(items, "q02")["expected"] = []
    assert any("q02: no expected document" in p for p in problems(items, corpus))
    items = make_questions()
    find(items, "q02")["expected"] = [entry(1), entry(2)]
    assert any("q02: a exact_term question needs exactly one" in p for p in problems(items, corpus))
    items = make_questions()
    multi = next(q for q in items if q["category"] == "multi_doc")
    multi["expected"] = [entry(3), entry(3)]
    found = problems(items, corpus)
    assert any("needs at least two different documents" in p for p in found)
    assert any("listed twice" in p for p in found)


def test_headline_rules(items, corpus):
    filtered = next(q for q in items if q["category"] == "filtered")
    del filtered["headline"]
    assert any('must have "headline": false' in p for p in problems(items, corpus))
    items = make_questions()
    find(items, "q02")["headline"] = False
    assert any('only filtered questions may have "headline": false' in p for p in problems(items, corpus))


# ------------------------------------------------------------------ database checks


def test_unknown_document(items, corpus):
    find(items, "q03")["expected"][0]["doc"] = "data/corpus/docs/missing.md"
    assert any("document not in the database" in p for p in problems(items, corpus))


def test_evidence_must_exist_in_the_stored_text(items, corpus):
    find(items, "q03")["expected"][0]["evidence"] = ["this phrase is nowhere"]
    assert any("q03: evidence not found in the stored text" in p for p in problems(items, corpus))


def test_evidence_matching_ignores_case_and_markdown(items, corpus):
    find(items, "q03")["expected"][0]["evidence"] = ["**EVIDENCE** `number` 2 APPEARS here"]
    assert problems(items, corpus) == []


def test_paraphrase_overlap_rule(items, corpus):
    q = next(x for x in items if x["category"] == "paraphrase")
    q["question"] = "Where does the evidence number appear here?"
    q["expected"][0]["evidence"] = ["Evidence number 10 appears here"]
    assert any("paraphrase shares" in p for p in problems(items, corpus))


def test_paraphrase_rule_includes_also_acceptable_evidence(items, corpus):
    q = next(x for x in items if x["category"] == "paraphrase")
    q["question"] = "Tell me about the alternative marker passage"
    q["also_acceptable"] = [{"doc": "data/corpus/docs/alt.md", "evidence": ["alternative evidence marker"]}]
    assert any("paraphrase shares" in p and "alternative" in p for p in problems(items, corpus))


# ------------------------------------------------------------ also_acceptable rules


def test_also_acceptable_evidence_must_exist_in_each_acceptable_document(items, corpus):
    q = find(items, "q03")
    q["also_acceptable"] = [{"doc": "data/corpus/docs/alt.md", "evidence": ["alternative evidence marker"]}]
    assert problems(items, corpus) == []
    q["also_acceptable"] = [{"doc": "data/corpus/docs/alt.md", "evidence": ["not in the alternative"]}]
    assert any("also_acceptable evidence not found in the stored text of data/corpus/docs/alt.md" in p for p in problems(items, corpus))
    q["also_acceptable"] = [{"doc": "data/corpus/docs/nowhere.md", "evidence": ["x"]}]
    assert any("document not in the database" in p for p in problems(items, corpus))


def test_also_acceptable_is_for_single_document_questions_only(items, corpus):
    multi = next(q for q in items if q["category"] == "multi_doc")
    multi["also_acceptable"] = [{"doc": "data/corpus/docs/alt.md", "evidence": ["alternative evidence marker"]}]
    assert any("only for single-document questions" in p for p in problems(items, corpus))


def test_also_acceptable_must_not_repeat_a_document(items, corpus):
    q = find(items, "q03")
    q["also_acceptable"] = [{"doc": q["expected"][0]["doc"], "evidence": q["expected"][0]["evidence"]}]
    assert any("also_acceptable repeats a document" in p for p in problems(items, corpus))


def test_alternative_documents_do_not_count_toward_the_per_document_limit(items, corpus):
    for qid in ("q03", "q04", "q05", "q06", "q07"):
        find(items, qid)["also_acceptable"] = [{"doc": "data/corpus/docs/alt.md", "evidence": ["alternative evidence marker"]}]
    assert problems(items, corpus) == []


# ----------------------------------------------------------------------- filters


def test_filters_must_parse_and_the_expected_document_must_satisfy_them(items, corpus):
    filtered = next(q for q in items if q["category"] == "filtered")
    filtered["filters"] = {"date_from": "2026-12-01"}
    assert any("does not satisfy the filters" in p for p in problems(items, corpus))
    filtered["filters"] = {"date_from": "2026-12-01", "date_to": "2026-01-01"}
    assert any("filters do not parse" in p for p in problems(items, corpus))
    filtered["filters"] = {}
    assert any("needs a 'filters' field" in p for p in problems(items, corpus))


def test_an_alternative_document_must_also_satisfy_the_filters(items, corpus):
    filtered = next(q for q in items if q["category"] == "filtered")
    filtered["filters"] = {"path_prefix": "data/corpus/docs/doc"}
    filtered["also_acceptable"] = [{"doc": "data/corpus/docs/alt.md", "evidence": ["alternative evidence marker"]}]
    assert any("alt.md does not satisfy the filters" in p for p in problems(items, corpus))


def test_doc_passes_filters_matches_the_database_rules():
    info = DocInfo("pdf", date(2026, 10, 2), [])
    assert doc_passes_filters("a/b.pdf", info, to_search_filters({"source_types": ["pdf"]}))
    assert not doc_passes_filters("a/b.pdf", info, to_search_filters({"source_types": ["markdown"]}))
    assert doc_passes_filters("a/b.pdf", info, to_search_filters({"date_from": "2026-10-02", "date_to": "2026-10-02"}))
    assert not doc_passes_filters("a/b.pdf", info, to_search_filters({"date_from": "2026-10-03"}))
    assert not doc_passes_filters("a/b.pdf", DocInfo("pdf", None, []), to_search_filters({"date_to": "2030-01-01"}))
    assert doc_passes_filters("a/b.pdf", info, to_search_filters({"path_prefix": "a/"}))
    assert not doc_passes_filters("a/b.pdf", info, to_search_filters({"path_prefix": "b/"}))


# -------------------------------------------------------- holdout and spread


def test_holdout_share_is_checked_per_category(items, corpus):
    for q in items:
        if q["category"] == "exact_term":
            q["holdout"] = q["id"] in ("q01", "q02", "q03", "q04", "q05", "q06")
    assert any("category exact_term: 6/10 holdout" in p for p in problems(items, corpus))
    for q in items:
        if q["category"] == "exact_term":
            q["holdout"] = False
    assert any("category exact_term: 0/10 holdout" in p for p in problems(items, corpus))


def test_too_many_questions_on_one_document(items, corpus):
    for q in items:
        if q["category"] == "paraphrase":
            q["expected"] = [entry(10)]
    assert any("is used by" in p and "(max 3)" in p for p in problems(items, corpus))


def test_too_few_distinct_documents(items, corpus):
    # 42 document uses packed onto 14 documents, 3 each: within the per-document limit, but under 15 documents.
    paths = ["sample_corpus/notes.md"] + [f"data/corpus/docs/doc{i:02d}.md" for i in range(1, 14)]

    def entry_for(path):
        return {"doc": path, "evidence": ["sample evidence marker" if path.startswith("sample") else f"Evidence number {int(path[-5:-3])} appears here"]}

    uses = 0
    for q in items:
        if q["category"] == "unanswerable":
            continue
        count = 2 if q["category"] == "multi_doc" else 1
        q["expected"] = [entry_for(paths[(uses + k) % 14]) for k in range(count)]
        uses += count
    found = problems(items, corpus)
    assert any("only 14 different documents (want at least 15)" in p for p in found), found
    assert not any("(max 3)" in p for p in found), found


def test_a_sample_corpus_document_is_required(items, corpus):
    find(items, "q01")["expected"] = [entry(1)]
    assert any("no question uses a document from sample_corpus/" in p for p in problems(items, corpus))


# ----------------------------------------------------- overlap and difficulty


def test_lexical_overlap_hand_computed():
    corpus = {"d.md": DocInfo("markdown", None, ["The quick brown fox jumps over the lazy dog near the river."])}
    q = parse_question(
        {"id": "q01", "category": "exact_term", "question": "Which brown fox jumps near mountains?", "holdout": False,
         "expected": [{"doc": "d.md", "evidence": ["quick brown fox"]}]}
    )
    # content words: brown, jumps, near, mountains (which is a stop word); in the chunk: brown, jumps, near
    assert lexical_overlaps([q], corpus) == {"q01": pytest.approx(0.75)}


def test_overlap_takes_the_best_chunk_and_the_best_acceptable_document():
    corpus = {
        "d.md": DocInfo("markdown", None, ["alpha beta evidence here", "gamma evidence here"]),
        "alt.md": DocInfo("markdown", None, ["alpha beta gamma delta evidence"]),
    }
    q = parse_question(
        {"id": "q01", "category": "exact_term", "question": "alpha beta gamma delta", "holdout": False,
         "expected": [{"doc": "d.md", "evidence": ["evidence here"]}],
         "also_acceptable": [{"doc": "alt.md", "evidence": ["delta evidence"]}]}
    )
    assert lexical_overlaps([q], corpus)["q01"] == pytest.approx(1.0)  # from the alternative document


def test_multi_doc_overlap_is_the_mean_over_documents():
    corpus = {
        "a.md": DocInfo("markdown", None, ["alpha beta evidence"]),
        "b.md": DocInfo("markdown", None, ["gamma evidence"]),
    }
    q = parse_question(
        {"id": "q01", "category": "multi_doc", "question": "alpha beta gamma delta", "holdout": False,
         "expected": [{"doc": "a.md", "evidence": ["alpha beta"]}, {"doc": "b.md", "evidence": ["gamma evidence"]}]}
    )
    assert lexical_overlaps([q], corpus)["q01"] == pytest.approx((0.5 + 0.25) / 2)


def test_unanswerable_questions_have_no_overlap():
    q = parse_question({"id": "q01", "category": "unanswerable", "question": "anything at all", "holdout": False, "expected": []})
    assert lexical_overlaps([q], {}) == {"q01": None}


def test_stored_overlap_must_be_present_and_current(items, corpus):
    found = validate(copy.deepcopy(items), corpus, require_overlap=True)
    assert any("lexical_overlap/difficulty missing" in p for p in found)

    fresh = {q.id: q for q in map(parse_question, items)}
    computed = lexical_overlaps(list(fresh.values()), corpus)
    for raw in items:
        value = computed[raw["id"]]
        if value is not None:
            raw["lexical_overlap"] = round(value, 2)
            raw["difficulty"] = "low_overlap" if round(value, 2) <= 0.25 else "high_overlap"
    assert validate(copy.deepcopy(items), corpus, require_overlap=True) == []

    answerable = next(q for q in items if "lexical_overlap" in q)
    answerable["lexical_overlap"] = round(answerable["lexical_overlap"] + 0.3, 2) % 1
    assert any("stale" in p for p in validate(copy.deepcopy(items), corpus, require_overlap=True))

    answerable["lexical_overlap"] = round(computed[answerable["id"]], 2)
    answerable["difficulty"] = "high_overlap" if answerable["difficulty"] == "low_overlap" else "low_overlap"
    assert any("does not match lexical_overlap" in p for p in validate(copy.deepcopy(items), corpus, require_overlap=True))


def test_update_overlap_writes_the_fields_and_the_file_then_validates(items, corpus, tmp_path):
    path = tmp_path / "questions.json"
    changed = update_overlap(items, corpus, path)
    assert changed == sum(1 for q in items if q["category"] != "unanswerable")
    written = json.loads(path.read_text(encoding="utf-8"))
    assert validate(written, corpus, require_overlap=True) == []
    assert all("lexical_overlap" not in q for q in written if q["category"] == "unanswerable")
    assert path.read_text(encoding="utf-8") == format_questions(written)  # stable layout
    assert update_overlap(written, corpus, path) == 0  # nothing left to change
