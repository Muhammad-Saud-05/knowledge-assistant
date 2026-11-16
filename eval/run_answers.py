"""Answer-level check: run the full pipeline (retrieval, prompt, language model) on the development
questions and record what the answers look like. A measurement, not an evaluation of correctness:
whether answers are right is judged by hand on a sample (see to_grade.md).

Usage:
    python -m eval.run_answers                    # run (or resume) and print the summary
    python -m eval.run_answers --summary-only     # rebuild the summary and to_grade.md from saved answers

Questions: the development main-table questions (exact_term, paraphrase, multi_doc) and the
development unanswerable questions, one at a time. The holdout is never touched. Every answer is
saved to eval/answers/answers.jsonl as soon as it is finished, so a stopped run resumes where it
stopped. That folder is git-ignored because it holds document text.
"""

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

from src.synthesis.answer import NOT_FOUND, Answer, parse_citations, validate_answer

from .metrics import median, percentile
from .questions import QUESTIONS_FILE, Question, load_questions, select_set, select_split

ANSWERS_DIR = Path("eval/answers")
ANSWERS_FILE = "answers.jsonl"
META_FILE = "meta.json"
GRADE_FILE = "to_grade.md"
MAIN_CATEGORIES = ("exact_term", "paraphrase", "multi_doc")
PREVIEW_CHARS = 200
GRADE_SAMPLE = 12
GRADE_SEED = 7
K = 5

AnswerFn = Callable[[str], Answer]


# ------------------------------------------------------------------ questions


def select_questions(questions: list[Question]) -> list[Question]:
    """Development main-table questions followed by development unanswerable questions."""
    dev = select_split(select_set(questions, "v2"), "dev")
    main = [q for q in dev if q.answerable and q.headline and q.category in MAIN_CATEGORIES]
    unanswerable = [q for q in dev if not q.answerable]
    return main + unanswerable


# -------------------------------------------------------------------- records


def preview(text: str) -> str:
    return " ".join(text.split())[:PREVIEW_CHARS]


def make_record(question: Question, answer: Answer, mode: str) -> dict:
    """What is kept for one question. The answer and the source previews are document-derived text,
    so this goes only into the git-ignored folder."""
    cited = sorted(parse_citations(answer.text))
    validation = answer.validation
    return {
        "id": question.id,
        "category": question.category,
        "answerable": question.answerable,
        "question": question.question,
        "mode": mode,
        "answer": answer.text,
        "exact_not_found": answer.text.strip() == NOT_FOUND,
        "tolerant_not_found": validation.is_not_found,
        "cited_numbers": cited,
        "has_citation": bool(cited),
        "all_citations_valid": not validation.cited_invalid,
        "invalid_numbers": list(validation.cited_invalid),
        "sources_provided": len(answer.chunks),
        "first_token_seconds": answer.timings.get("first_token"),
        "total_seconds": answer.timings.get("total"),
        "retrieval_seconds": answer.timings.get("retrieval"),
        "sources": [
            {"number": n, "title": c.title, "heading": c.section_heading, "page": c.page_number, "preview": preview(c.text)}
            for n, c in enumerate(answer.chunks, start=1)
        ],
    }


def load_records(path: Path) -> dict[str, dict]:
    """Saved records by question id (a torn last line from an interrupted run is ignored)."""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        records[record["id"]] = record
    return records


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def run_questions(
    questions: list[Question], answer_fn: AnswerFn, mode: str, out_dir: Path, log: Callable[[str], None] = lambda _: None
) -> dict[str, dict]:
    """Answer every question not yet saved, one at a time, saving each record as it is finished."""
    path = out_dir / ANSWERS_FILE
    records = load_records(path)
    for number, question in enumerate(questions, start=1):
        if question.id in records:
            log(f"[{number}/{len(questions)}] {question.id}: already done")
            continue
        record = make_record(question, answer_fn(question.question), mode)
        append_record(path, record)
        records[question.id] = record
        log(f"[{number}/{len(questions)}] {question.id}: first token {record['first_token_seconds']}, total {record['total_seconds']}")
    return records


# -------------------------------------------------------------------- summary


def _share(count: int, n: int) -> str:
    return f"{count}/{n} ({100 * count / n:.1f}%)" if n else "0/0"


def summarize(records: list[dict]) -> dict:
    un = [r for r in records if not r["answerable"]]
    an = [r for r in records if r["answerable"]]
    un_answered = [r for r in un if not r["exact_not_found"]]
    an_answered = [r for r in an if not r["exact_not_found"]]
    answered = un_answered + an_answered

    def cited(rs):
        return sum(1 for r in rs if r["has_citation"])

    def all_valid(rs):  # at least one citation, and every cited number is among the sources provided
        return sum(1 for r in rs if r["has_citation"] and r["all_citations_valid"])

    firsts = [r["first_token_seconds"] for r in records if r["first_token_seconds"] is not None]
    totals = [r["total_seconds"] for r in records if r["total_seconds"] is not None]
    return {
        "n": len(records),
        "unanswerable": {
            "n": len(un),
            "refused": sum(1 for r in un if r["exact_not_found"]),
            "answered": len(un_answered),
            "answered_with_citation": cited(un_answered),
            "answered_without_citation": len(un_answered) - cited(un_answered),
        },
        "answerable": {
            "n": len(an),
            "wrongly_refused": sum(1 for r in an if r["exact_not_found"]),
            "answered": len(an_answered),
        },
        "answered": {
            "n": len(answered),
            "with_citation": cited(answered),
            "all_citations_valid": all_valid(answered),
            "answerable_n": len(an_answered),
            "answerable_with_citation": cited(an_answered),
            "answerable_all_valid": all_valid(an_answered),
            "unanswerable_n": len(un_answered),
            "unanswerable_with_citation": cited(un_answered),
            "unanswerable_all_valid": all_valid(un_answered),
            "invalid_numbers": sum(len(r["invalid_numbers"]) for r in answered),
        },
        "tolerant_differs": sum(1 for r in records if r["tolerant_not_found"] != r["exact_not_found"]),
        "first_token": (median(firsts), percentile(firsts, 0.95), len(firsts)) if firsts else None,
        "total": (median(totals), percentile(totals, 0.95), len(totals)) if totals else None,
    }


def render_summary(s: dict) -> str:
    un, an, ans = s["unanswerable"], s["answerable"], s["answered"]
    rows = [
        ("Unanswerable questions: correctly refused (the exact not-found sentence)", _share(un["refused"], un["n"])),
        ("Unanswerable questions: answered with at least one citation anyway", _share(un["answered_with_citation"], un["n"])),
        ("Unanswerable questions: answered without any citation", _share(un["answered_without_citation"], un["n"])),
        ("Answerable questions: wrongly refused", _share(an["wrongly_refused"], an["n"])),
        ("Answerable questions: answered", _share(an["answered"], an["n"])),
        ("Answers (not refused) with at least one citation, all 39 questions", _share(ans["with_citation"], ans["n"])),
        ("Answers with at least one citation and every cited number among the sources, all", _share(ans["all_citations_valid"], ans["n"])),
        ("  of the answerable questions' answers: with at least one citation", _share(ans["answerable_with_citation"], ans["answerable_n"])),
        ("  of the answerable questions' answers: all citations valid", _share(ans["answerable_all_valid"], ans["answerable_n"])),
        ("  of the unanswerable questions' answers: with at least one citation", _share(ans["unanswerable_with_citation"], ans["unanswerable_n"])),
        ("  of the unanswerable questions' answers: all citations valid", _share(ans["unanswerable_all_valid"], ans["unanswerable_n"])),
    ]
    lines = ["| measure | count / n |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    timing = []
    for label, key in (("time to first token", "first_token"), ("total time", "total")):
        if s[key]:
            med, p95, n = s[key]
            timing.append(f"- {label}: median {med:.1f} s, 95th percentile {p95:.1f} s (n={n})")
    extra = [f"- citation numbers outside the sources provided: {ans['invalid_numbers']}"]
    if s["tolerant_differs"]:
        extra.append(f"- answers where the app's tolerant not-found check and the exact check disagree: {s['tolerant_differs']}")
    return "\n".join(lines + [""] + timing + extra) + "\n"


# ------------------------------------------------------------------ recompute


def recompute(records: list[dict]) -> dict:
    """Re-judge SAVED answers with the current refusal and citation checks (nothing is regenerated).

    The old figures use the fields saved when the answers were made (exact not-found sentence; any cited
    number counted as a citation). The new ones run the current validate_answer on the saved text and the
    saved number of sources.
    """
    rows = []
    for r in records:
        new = validate_answer(r["answer"], r["sources_provided"])
        rows.append(
            {
                "id": r["id"],
                "answerable": r["answerable"],
                "old_refused": r["exact_not_found"],
                "new_refused": new.refused,
                "old_uncited": not r["exact_not_found"] and not r["has_citation"],
                "new_uncited": new.uncited,
                "old_invalid": bool(r["invalid_numbers"]),
                "new_invalid": bool(new.invalid_citations),
            }
        )
    un = [x for x in rows if not x["answerable"]]
    an = [x for x in rows if x["answerable"]]

    def count(group, key):
        return sum(1 for x in group if x[key])

    return {
        "n_unanswerable": len(un),
        "n_answerable": len(an),
        "correctly_refused": (count(un, "old_refused"), count(un, "new_refused")),
        "wrongly_refused": (count(an, "old_refused"), count(an, "new_refused")),
        "uncited_answerable": (count(an, "old_uncited"), count(an, "new_uncited")),
        "uncited_unanswerable": (count(un, "old_uncited"), count(un, "new_uncited")),
        "invalid_answers": (count(rows, "old_invalid"), count(rows, "new_invalid")),
        "n": len(rows),
        "changed_refusal": sorted(x["id"] for x in rows if x["old_refused"] != x["new_refused"]),
    }


def render_recompute(s: dict) -> str:
    def row(label, pair, n):
        return f"| {label} | {_share(pair[0], n)} | {_share(pair[1], n)} |"

    lines = [
        "| measure | old (exact sentence; any number counts as a citation) | new (shared refusal check; valid citations only) |",
        "|---|---|---|",
        row("Unanswerable questions correctly refused", s["correctly_refused"], s["n_unanswerable"]),
        row("Answerable questions wrongly refused", s["wrongly_refused"], s["n_answerable"]),
        row("Answerable questions: answer not a refusal and cites no source", s["uncited_answerable"], s["n_answerable"]),
        row("Unanswerable questions: answer not a refusal and cites no source", s["uncited_unanswerable"], s["n_unanswerable"]),
        row("Answers with at least one invalid citation, all questions", s["invalid_answers"], s["n"]),
    ]
    changed = ", ".join(s["changed_refusal"]) or "none"
    return "\n".join(lines) + f"\n\nQuestions whose refusal status changed: {changed}\n"


# ------------------------------------------------------------------ to_grade


def grading_sample(records: dict[str, dict], seed: int = GRADE_SEED, n: int = GRADE_SAMPLE) -> list[dict]:
    """n answers of answerable questions chosen at random (seeded), plus every unanswerable question
    that was answered instead of refused. In question-id order."""
    answerable_ids = sorted(i for i, r in records.items() if r["answerable"])
    chosen = set(random.Random(seed).sample(answerable_ids, min(n, len(answerable_ids))))
    unrefused = {i for i, r in records.items() if not r["answerable"] and not r["exact_not_found"]}
    return [records[i] for i in sorted(chosen | unrefused)]


def render_to_grade(sample: list[dict]) -> str:
    lines = [
        "# Answers to grade by hand",
        "",
        f"{len(sample)} answers: {GRADE_SAMPLE} answerable questions chosen at random (seed {GRADE_SEED}) and every",
        "unanswerable question that was answered instead of refused. Grades: correct / partly / wrong / unsupported.",
        "Nothing here has been graded by the assistant.",
        "",
    ]
    for r in sample:
        lines += [f"## {r['id']}", "", f"**Question:** {r['question']}", "", "**Answer:**", "", r["answer"].strip() or "(empty)", "", "**Sources provided:**", ""]
        for s in r["sources"]:
            place = ", ".join(x for x in (s["title"] or "Untitled", f"p. {s['page']}" if s["page"] is not None else None, s["heading"]) if x)
            lines.append(f"[{s['number']}] {place}")
            lines.append(f"> {s['preview']}")
            lines.append("")
        lines += ["Grade: correct / partly / wrong / unsupported", "", "---", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------- main


def build_pipeline():
    """The default pipeline, as the application would run it."""
    from src.db.connection import get_connection
    from src.ingestion.embedder import Embedder
    from src.llm.ollama import OllamaLLM
    from src.retrieval.defaults import (
        SPLIT_MODE,
        decomposition_enabled,
        load_reranker,
        make_decomposer,
        reranker_enabled,
        resolve_mode,
    )
    from src.synthesis.answer import answer_question_stream

    reranking = reranker_enabled()
    mode = resolve_mode(None, reranking, reranking, decomposition_enabled())
    conn = get_connection()
    embedder = Embedder()
    embedder.embed_query("warm up")
    reranker = load_reranker() if mode.endswith("+rerank") or mode == SPLIT_MODE else None
    decomposer = make_decomposer() if mode == SPLIT_MODE else None
    llm = OllamaLLM()
    llm.warm_up()

    def answer(question: str) -> Answer:
        stream = answer_question_stream(question, conn, embedder, llm, k=K, mode=mode, reranker=reranker, decomposer=decomposer)
        while True:
            try:
                next(stream)
            except StopIteration as done:
                return done.value

    return answer, mode, llm.model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer-level check on the development questions.")
    parser.add_argument("--out", type=Path, default=ANSWERS_DIR, help="folder for answers (git-ignored)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--summary-only", action="store_true", help="do not run anything; rebuild the summary and to_grade.md")
    parser.add_argument("--recompute", action="store_true", help="re-judge the saved answers with the current checks and print old against new (nothing is run)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = select_questions(load_questions(args.questions))
    if args.recompute:
        wanted = {q.id for q in questions}
        saved = [r for i, r in load_records(args.out / ANSWERS_FILE).items() if i in wanted]
        if not saved:
            print("No saved answers to recompute.", file=sys.stderr)
            return 1
        print(render_recompute(recompute(saved)))
        return 0
    if not args.summary_only:
        answer_fn, mode, model = build_pipeline()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / META_FILE).write_text(json.dumps({"mode": mode, "model": model, "k": K, "questions": len(questions)}, indent=1), encoding="utf-8")
        records = run_questions(questions, answer_fn, mode, args.out, log=lambda m: print(m, flush=True))
    else:
        records = load_records(args.out / ANSWERS_FILE)

    wanted = {q.id for q in questions}
    done = {i: r for i, r in records.items() if i in wanted}
    if len(done) < len(wanted):
        print(f"Only {len(done)} of {len(wanted)} questions have answers; the summary is incomplete.", file=sys.stderr)
    if not done:
        return 1
    print(render_summary(summarize(list(done.values()))))
    (args.out / GRADE_FILE).write_text(render_to_grade(grading_sample(done)), encoding="utf-8")
    print(f"Wrote {args.out / GRADE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
