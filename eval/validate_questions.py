"""Check the evaluation question file against the database.

Usage: python -m eval.validate_questions [--file PATH] [--tables]
Exit status 0 when everything passes, 1 when anything is wrong (each problem is listed).
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.retrieval.filters import SearchFilters

from .metrics import content_words, normalize
from .questions import (
    CATEGORIES,
    EXPECTED_COUNTS,
    FILTER_FIELDS,
    ID_PATTERN,
    MAX_EVIDENCE_WORDS,
    QUESTIONS_FILE,
    Question,
    parse_question,
)

MAX_PARAPHRASE_OVERLAP = 1  # shared content words between a paraphrase question and its evidence
HOLDOUT_SHARE = (0.15, 0.35)  # allowed share of holdout questions per category ("about 25%")
MIN_DISTINCT_DOCS = 15
MAX_QUESTIONS_PER_DOC = 3
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class DocInfo:
    source_type: str
    doc_date: date | None
    chunks: list[str]  # the stored chunk texts


Corpus = dict[str, DocInfo]


def load_corpus(conn) -> Corpus:
    corpus: Corpus = {}
    for doc_id, path, source_type, doc_date in conn.execute(
        "SELECT id, path, source_type, doc_date FROM documents"
    ).fetchall():
        corpus[path] = DocInfo(source_type, doc_date, [])
        rows = conn.execute("SELECT text FROM chunks WHERE document_id = %s ORDER BY chunk_index", (doc_id,))
        corpus[path].chunks = [r[0] for r in rows.fetchall()]
    return corpus


# ------------------------------------------------------------------- checks


def _schema_errors(raw: object, position: int) -> list[str]:
    label = raw.get("id", f"item {position}") if isinstance(raw, dict) else f"item {position}"
    if not isinstance(raw, dict):
        return [f"{label}: not a JSON object"]
    errors = []

    def problem(message: str) -> None:
        errors.append(f"{label}: {message}")

    for key in ("id", "category", "question", "expected", "holdout"):
        if key not in raw:
            problem(f"missing field '{key}'")
    unknown = set(raw) - {"id", "category", "question", "expected", "holdout", "filters", "notes"}
    if unknown:
        problem(f"unknown field(s): {', '.join(sorted(unknown))}")
    if "id" in raw and not (isinstance(raw["id"], str) and ID_PATTERN.match(raw["id"])):
        problem("id must look like q01")
    if "category" in raw and raw["category"] not in CATEGORIES:
        problem(f"category must be one of {', '.join(CATEGORIES)}")
    if "question" in raw and not (isinstance(raw["question"], str) and raw["question"].strip()):
        problem("question must be a non-empty string")
    if "holdout" in raw and not isinstance(raw["holdout"], bool):
        problem("holdout must be true or false")
    if "notes" in raw and not isinstance(raw["notes"], str):
        problem("notes must be a string")

    expected = raw.get("expected")
    if "expected" in raw:
        if not isinstance(expected, list):
            problem("expected must be a list")
        else:
            for item in expected:
                if not isinstance(item, dict) or not isinstance(item.get("doc"), str):
                    problem("each expected entry needs a 'doc' path")
                    continue
                evidence = item.get("evidence")
                if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
                    problem(f"{item['doc']}: evidence must be a list of 1 to 3 phrases")
                    continue
                for phrase in evidence:
                    if not isinstance(phrase, str) or not phrase.strip():
                        problem(f"{item['doc']}: evidence phrases must be non-empty strings")
                    elif len(phrase.split()) > MAX_EVIDENCE_WORDS:
                        problem(f"evidence has {len(phrase.split())} words (max {MAX_EVIDENCE_WORDS}): {phrase!r}")

    filters = raw.get("filters", {})
    if filters is not None:
        if not isinstance(filters, dict):
            problem("filters must be an object")
        else:
            bad = set(filters) - set(FILTER_FIELDS)
            if bad:
                problem(f"unknown filter field(s): {', '.join(sorted(bad))}")
            for name in ("date_from", "date_to"):
                value = filters.get(name)
                if value is not None and not (isinstance(value, str) and ISO_DATE.match(value)):
                    problem(f"{name} must be YYYY-MM-DD")
    return errors


def to_search_filters(filters: dict) -> SearchFilters | None:
    if not filters:
        return None
    return SearchFilters(
        source_types=filters.get("source_types"),
        date_from=date.fromisoformat(filters["date_from"]) if filters.get("date_from") else None,
        date_to=date.fromisoformat(filters["date_to"]) if filters.get("date_to") else None,
        path_prefix=filters.get("path_prefix"),
    )


def doc_passes_filters(path: str, info: DocInfo, filters: SearchFilters) -> bool:
    """The same rule the database applies (see src/retrieval/filters.py)."""
    if filters.source_types is not None and info.source_type not in filters.source_types:
        return False
    if filters.date_from is not None and (info.doc_date is None or info.doc_date < filters.date_from):
        return False
    if filters.date_to is not None and (info.doc_date is None or info.doc_date > filters.date_to):
        return False
    if filters.path_prefix and not path.startswith(filters.path_prefix):
        return False
    return True


def validate(raw_items: list, corpus: Corpus) -> list[str]:
    """Every problem found in the question file; an empty list means it is valid."""
    errors: list[str] = []
    for position, raw in enumerate(raw_items, start=1):
        errors += _schema_errors(raw, position)
    if errors:
        return errors  # the checks below assume a well-formed file

    questions: list[Question] = [parse_question(raw) for raw in raw_items]

    # ids, categories, duplicates
    ids = Counter(q.id for q in questions)
    errors += [f"duplicate id {i} ({n} times)" for i, n in ids.items() if n > 1]
    counts = Counter(q.category for q in questions)
    for category, want in EXPECTED_COUNTS.items():
        if counts.get(category, 0) != want:
            errors.append(f"category {category}: expected {want} questions, found {counts.get(category, 0)}")
    if len(questions) != sum(EXPECTED_COUNTS.values()):
        errors.append(f"expected {sum(EXPECTED_COUNTS.values())} questions in total, found {len(questions)}")
    seen: dict[str, str] = {}
    for q in questions:
        key = normalize(q.question)
        if key in seen:
            errors.append(f"{q.id}: question text duplicates {seen[key]}")
        seen[key] = q.id

    per_doc: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        docs = q.expected_docs
        if q.category == "unanswerable":
            if q.expected:
                errors.append(f"{q.id}: an unanswerable question must have an empty 'expected'")
            continue
        if not q.expected:
            errors.append(f"{q.id}: no expected document")
            continue
        distinct = set(docs)
        if q.category == "multi_doc" and len(distinct) < 2:
            errors.append(f"{q.id}: a multi_doc question needs at least two different documents")
        if q.category != "multi_doc" and len(docs) != 1:
            errors.append(f"{q.id}: a {q.category} question needs exactly one expected document")
        if len(distinct) != len(docs):
            errors.append(f"{q.id}: the same document is listed twice")

        for expected in q.expected:
            per_doc[expected.doc].add(q.id)
            info = corpus.get(expected.doc)
            if info is None:
                errors.append(f"{q.id}: document not in the database: {expected.doc}")
                continue
            texts = [normalize(chunk) for chunk in info.chunks]
            for phrase in expected.evidence:
                if not any(normalize(phrase) in text for text in texts):
                    errors.append(f"{q.id}: evidence not found in the stored text of {expected.doc}: {phrase!r}")

        if q.category == "paraphrase":
            evidence_words = content_words(" ".join(p for e in q.expected for p in e.evidence))
            shared = sorted(content_words(q.question) & evidence_words)
            if len(shared) > MAX_PARAPHRASE_OVERLAP:
                errors.append(f"{q.id}: paraphrase shares {len(shared)} content words with its evidence: {shared}")

        if q.category == "filtered" and not q.filters:
            errors.append(f"{q.id}: a filtered question needs a 'filters' field")
        if q.filters:
            try:
                search_filters = to_search_filters(q.filters)
                search_filters.validate()
            except (ValueError, TypeError) as error:
                errors.append(f"{q.id}: filters do not parse: {error}")
                continue
            for expected in q.expected:
                info = corpus.get(expected.doc)
                if info is not None and not doc_passes_filters(expected.doc, info, search_filters):
                    errors.append(f"{q.id}: expected document {expected.doc} does not satisfy the filters")

    # holdout shares
    for category in CATEGORIES:
        members = [q for q in questions if q.category == category]
        if not members:
            continue
        holdouts = sum(q.holdout for q in members)
        share = holdouts / len(members)
        if holdouts < 1 or not HOLDOUT_SHARE[0] <= share <= HOLDOUT_SHARE[1]:
            errors.append(
                f"category {category}: {holdouts}/{len(members)} holdout questions "
                f"({share:.0%}); want about 25% and at least one"
            )

    # spread over documents
    if len(per_doc) < MIN_DISTINCT_DOCS:
        errors.append(f"questions touch only {len(per_doc)} different documents (want at least {MIN_DISTINCT_DOCS})")
    for doc, question_ids in sorted(per_doc.items()):
        if len(question_ids) > MAX_QUESTIONS_PER_DOC:
            errors.append(f"{doc} is used by {len(question_ids)} questions (max {MAX_QUESTIONS_PER_DOC})")
    if not any(doc.startswith("sample_corpus/") for doc in per_doc):
        errors.append("no question uses a document from sample_corpus/")
    return errors


# ----------------------------------------------------------------- reporting


def lexical_overlaps(questions: list[Question], corpus: Corpus) -> dict[str, float | None]:
    """For each answerable question: the share of its content words that appear in its best
    expected chunk (averaged over its expected documents). Information only, never enforced."""
    result: dict[str, float | None] = {}
    for q in questions:
        words = content_words(q.question)
        if not q.expected or not words:
            result[q.id] = None
            continue
        per_doc = []
        for expected in q.expected:
            info = corpus.get(expected.doc)
            best = 0.0
            for chunk in info.chunks if info else []:
                text = normalize(chunk)
                if any(normalize(p) in text for p in expected.evidence):
                    best = max(best, len(words & content_words(chunk)) / len(words))
            per_doc.append(best)
        result[q.id] = sum(per_doc) / len(per_doc)
    return result


def short_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def print_tables(questions: list[Question], overlaps: dict[str, float | None]) -> None:
    for category in CATEGORIES:
        members = [q for q in questions if q.category == category]
        print(f"\n### {category} ({len(members)})\n")
        extra = " filters |" if category == "filtered" else ""
        print(f"| id | H | question | expected document(s) |{extra} overlap |")
        print(f"|---|---|---|---|{'---|' if extra else ''}---|")
        for q in members:
            docs = ", ".join(short_name(d) for d in q.expected_docs) or "(none: unanswerable)"
            overlap = overlaps.get(q.id)
            overlap_text = f"{overlap:.2f}" if overlap is not None else "-"
            filters_text = f" {json.dumps(q.filters)} |" if category == "filtered" else ""
            print(f"| {q.id} | {'H' if q.holdout else ''} | {q.question} | {docs} |{filters_text} {overlap_text} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the evaluation question file.")
    parser.add_argument("--file", type=Path, default=QUESTIONS_FILE, help="question file (default: eval/questions.json)")
    parser.add_argument("--tables", action="store_true", help="also print the questions grouped by category")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        raw_items = json.loads(args.file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"Cannot read {args.file}: {error}", file=sys.stderr)
        return 1
    if not isinstance(raw_items, list):
        print("The question file must contain a JSON list.", file=sys.stderr)
        return 1

    from src.db.connection import get_connection

    with get_connection() as conn:
        corpus = load_corpus(conn)
    errors = validate(raw_items, corpus)

    if errors:
        print(f"FAILED: {len(errors)} problem(s) in {args.file}\n")
        for error in errors:
            print(f"  - {error}")
        return 1

    questions = [parse_question(raw) for raw in raw_items]
    holdouts = sum(q.holdout for q in questions)
    docs = {d for q in questions for d in q.expected_docs}
    print(f"OK: {len(questions)} questions ({len(questions) - holdouts} dev, {holdouts} holdout) over {len(docs)} documents.")
    print("Checked: schema, ids, category counts, document paths, evidence phrases found in stored text,")
    print("paraphrase word overlap, holdout shares, filters, spread over documents, duplicate questions.")
    overlaps = lexical_overlaps(questions, corpus)
    by_category = defaultdict(list)
    for q in questions:
        if overlaps[q.id] is not None:
            by_category[q.category].append(overlaps[q.id])
    print("\nLexical overlap (share of the question's content words found in its best expected chunk), mean per category:")
    for category, values in by_category.items():
        print(f"  {category:<12} {sum(values) / len(values):.2f}   (min {min(values):.2f}, max {max(values):.2f})")
    if args.tables:
        print_tables(questions, overlaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
