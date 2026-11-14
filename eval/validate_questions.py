"""Check the evaluation question file against the database.

Usage: python -m eval.validate_questions [--file PATH] [--tables] [--ids q01,q02] [--update-overlap]
Exit status 0 when everything passes, 1 when anything is wrong (each problem is listed).

--update-overlap recomputes each answerable question's lexical_overlap and difficulty from the
stored text and writes them into the file. The validator then checks the stored numbers are
current, so editing a question without updating them is caught.
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
    DIFFICULTIES,
    EXPECTED_COUNTS,
    FILTER_FIELDS,
    ID_PATTERN,
    KNOWN_FIELDS,
    LOW_OVERLAP_MAX,
    MAX_EVIDENCE_WORDS,
    QUESTIONS_FILE,
    Question,
    difficulty_for,
    format_questions,
    parse_question,
)

MAX_PARAPHRASE_OVERLAP = 1  # shared content words between a paraphrase question and its evidence
HOLDOUT_SHARE = (0.15, 0.35)  # allowed share of holdout questions per category ("about 25%")
MIN_DISTINCT_DOCS = 15
MAX_QUESTIONS_PER_DOC = 3
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SINGLE_DOC_CATEGORIES = ("exact_term", "paraphrase", "filtered")


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


def _entry_errors(items: object, name: str, label: str) -> list[str]:
    """Schema problems in an 'expected' or 'also_acceptable' list."""
    if not isinstance(items, list):
        return [f"{label}: {name} must be a list"]
    errors = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("doc"), str):
            errors.append(f"{label}: each {name} entry needs a 'doc' path")
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
            errors.append(f"{label}: {item['doc']}: evidence must be a list of 1 to 3 phrases")
            continue
        for phrase in evidence:
            if not isinstance(phrase, str) or not phrase.strip():
                errors.append(f"{label}: {item['doc']}: evidence phrases must be non-empty strings")
            elif len(phrase.split()) > MAX_EVIDENCE_WORDS:
                errors.append(f"{label}: evidence has {len(phrase.split())} words (max {MAX_EVIDENCE_WORDS}): {phrase!r}")
    return errors


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
    unknown = set(raw) - KNOWN_FIELDS
    if unknown:
        problem(f"unknown field(s): {', '.join(sorted(unknown))}")
    if "id" in raw and not (isinstance(raw["id"], str) and ID_PATTERN.match(raw["id"])):
        problem("id must look like q01")
    if "category" in raw and raw["category"] not in CATEGORIES:
        problem(f"category must be one of {', '.join(CATEGORIES)}")
    if "question" in raw and not (isinstance(raw["question"], str) and raw["question"].strip()):
        problem("question must be a non-empty string")
    for flag in ("holdout", "headline"):
        if flag in raw and not isinstance(raw[flag], bool):
            problem(f"{flag} must be true or false")
    if "notes" in raw and not isinstance(raw["notes"], str):
        problem("notes must be a string")
    if "lexical_overlap" in raw and not (
        isinstance(raw["lexical_overlap"], (int, float)) and 0 <= raw["lexical_overlap"] <= 1
    ):
        problem("lexical_overlap must be a number between 0 and 1")
    if "difficulty" in raw and raw["difficulty"] not in DIFFICULTIES:
        problem(f"difficulty must be one of {', '.join(DIFFICULTIES)}")
    if "expected" in raw:
        errors += _entry_errors(raw["expected"], "expected", label)
    if "also_acceptable" in raw:
        errors += _entry_errors(raw["also_acceptable"], "also_acceptable", label)

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


def lexical_overlaps(questions: list[Question], corpus: Corpus) -> dict[str, float | None]:
    """For each answerable question: the share of its content words that appear in its best
    relevant chunk. For a single-document question the best chunk may come from the expected or
    any also_acceptable document; a multi-document question averages the best chunk of each of
    its documents. None for unanswerable questions."""
    result: dict[str, float | None] = {}
    for q in questions:
        words = content_words(q.question)
        if not q.expected or not words:
            result[q.id] = None
            continue
        per_slot = []
        for slot in q.slots:
            best = 0.0
            for item in slot:
                info = corpus.get(item.doc)
                for chunk in info.chunks if info else []:
                    text = normalize(chunk)
                    if any(normalize(p) in text for p in item.evidence):
                        best = max(best, len(words & content_words(chunk)) / len(words))
            per_slot.append(best)
        result[q.id] = sum(per_slot) / len(per_slot)
    return result


def validate(raw_items: list, corpus: Corpus, require_overlap: bool = True) -> list[str]:
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
        # headline: only filtered questions are kept out of the main results table
        if q.category == "filtered" and q.headline:
            errors.append(f"{q.id}: a filtered question must have \"headline\": false")
        if q.category != "filtered" and not q.headline:
            errors.append(f"{q.id}: only filtered questions may have \"headline\": false")

        docs = q.expected_docs
        if q.category == "unanswerable":
            if q.expected or q.also_acceptable:
                errors.append(f"{q.id}: an unanswerable question must have an empty 'expected'")
            if q.lexical_overlap is not None or q.difficulty is not None:
                errors.append(f"{q.id}: an unanswerable question has no lexical_overlap or difficulty")
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
        if q.also_acceptable and q.category not in SINGLE_DOC_CATEGORIES:
            errors.append(f"{q.id}: also_acceptable is only for single-document questions")
        alt_docs = [e.doc for e in q.also_acceptable]
        if len(set(alt_docs)) != len(alt_docs) or set(alt_docs) & distinct:
            errors.append(f"{q.id}: also_acceptable repeats a document")

        for expected in (*q.expected, *q.also_acceptable):
            is_alternative = expected in q.also_acceptable
            if not is_alternative:
                per_doc[expected.doc].add(q.id)
            info = corpus.get(expected.doc)
            if info is None:
                errors.append(f"{q.id}: document not in the database: {expected.doc}")
                continue
            texts = [normalize(chunk) for chunk in info.chunks]
            for phrase in expected.evidence:
                if not any(normalize(phrase) in text for text in texts):
                    kind = "also_acceptable evidence" if is_alternative else "evidence"
                    errors.append(f"{q.id}: {kind} not found in the stored text of {expected.doc}: {phrase!r}")

        if q.category == "paraphrase":
            evidence_words = content_words(" ".join(p for slot in q.slots for e in slot for p in e.evidence))
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
            for item in (*q.expected, *q.also_acceptable):
                info = corpus.get(item.doc)
                if info is not None and not doc_passes_filters(item.doc, info, search_filters):
                    errors.append(f"{q.id}: document {item.doc} does not satisfy the filters")

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

    # stored lexical overlap and difficulty must be current
    if require_overlap and not errors:
        computed = lexical_overlaps(questions, corpus)
        for q in questions:
            if computed[q.id] is None:
                continue
            fresh = computed[q.id]
            if q.lexical_overlap is None or q.difficulty is None:
                errors.append(f"{q.id}: lexical_overlap/difficulty missing (run --update-overlap)")
            elif abs(q.lexical_overlap - round(fresh, 2)) > 1e-9:  # stored values are rounded to 2 decimals
                errors.append(f"{q.id}: stored lexical_overlap {q.lexical_overlap} is stale, now {fresh:.2f} (run --update-overlap)")
            elif q.difficulty != difficulty_for(q.lexical_overlap):
                errors.append(f"{q.id}: difficulty {q.difficulty!r} does not match lexical_overlap {q.lexical_overlap}")
    return errors


# ----------------------------------------------------------------- reporting


def short_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def print_tables(questions: list[Question], overlaps: dict[str, float | None]) -> None:
    for category in CATEGORIES:
        members = [q for q in questions if q.category == category]
        if not members:
            continue
        print(f"\n### {category} ({len(members)})\n")
        extra = " filters |" if category == "filtered" else ""
        print(f"| id | H | question | expected document(s) |{extra} overlap |")
        print(f"|---|---|---|---|{'---|' if extra else ''}---|")
        for q in members:
            docs = ", ".join(short_name(d) for d in q.expected_docs) or "(none: unanswerable)"
            if q.also_acceptable:
                docs += " (also: " + ", ".join(short_name(e.doc) for e in q.also_acceptable) + ")"
            overlap = overlaps.get(q.id)
            overlap_text = "-" if overlap is None else f"{overlap:.2f} {'low' if overlap <= LOW_OVERLAP_MAX else 'high'}"
            filters_text = f" {json.dumps(q.filters)} |" if category == "filtered" else ""
            print(f"| {q.id} | {'H' if q.holdout else ''} | {q.question} | {docs} |{filters_text} {overlap_text} |")


def update_overlap(raw_items: list, corpus: Corpus, path: Path) -> int:
    questions = [parse_question(raw) for raw in raw_items]
    computed = lexical_overlaps(questions, corpus)
    changed = 0
    for raw in raw_items:
        overlap = computed[raw["id"]]
        if overlap is None:
            raw.pop("lexical_overlap", None)
            raw.pop("difficulty", None)
            continue
        rounded = round(overlap, 2)
        if raw.get("lexical_overlap") != rounded or raw.get("difficulty") != difficulty_for(rounded):
            changed += 1
        raw["lexical_overlap"] = rounded
        raw["difficulty"] = difficulty_for(rounded)
    path.write_text(format_questions(raw_items), encoding="utf-8", newline="\n")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the evaluation question file.")
    parser.add_argument("--file", type=Path, default=QUESTIONS_FILE, help="question file (default: eval/questions.json)")
    parser.add_argument("--tables", action="store_true", help="also print the questions grouped by category")
    parser.add_argument("--ids", help="with --tables, only these question ids (comma separated)")
    parser.add_argument("--update-overlap", action="store_true", help="write lexical_overlap and difficulty into the file")
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

    if args.update_overlap:
        errors = validate(raw_items, corpus, require_overlap=False)
        if errors:
            print("Fix these problems first:\n")
            for error in errors:
                print(f"  - {error}")
            return 1
        changed = update_overlap(raw_items, corpus, args.file)
        print(f"Updated lexical_overlap and difficulty ({changed} question(s) changed).")
        raw_items = json.loads(args.file.read_text(encoding="utf-8"))

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
    print("Checked: schema, ids, category counts, document paths, evidence phrases (including also_acceptable)")
    print("found in stored text, paraphrase word overlap, holdout shares, filters, headline flags, spread over")
    print("documents, duplicate questions, and that the stored lexical overlap and difficulty are current.")

    overlaps = {q.id: q.lexical_overlap for q in questions}
    by_category = defaultdict(list)
    by_difficulty = Counter()
    for q in questions:
        if overlaps[q.id] is not None:
            by_category[q.category].append(overlaps[q.id])
            by_difficulty[q.difficulty] += 1
    print("\nLexical overlap (share of the question's content words found in its best relevant chunk):")
    for category, values in by_category.items():
        print(f"  {category:<12} mean {sum(values) / len(values):.2f}   (min {min(values):.2f}, max {max(values):.2f})")
    print(f"  difficulty buckets (low_overlap = overlap <= {LOW_OVERLAP_MAX}): "
          + ", ".join(f"{name} {by_difficulty[name]}" for name in DIFFICULTIES))
    if args.tables:
        wanted = set(args.ids.split(",")) if args.ids else None
        shown = [q for q in questions if wanted is None or q.id in wanted]
        print_tables(shown, overlaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
