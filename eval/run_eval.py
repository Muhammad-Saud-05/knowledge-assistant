"""Run the evaluation questions through retrieval and report how well each mode does.

Usage:
    python -m eval.run_eval --modes vector,keyword,hybrid --split dev --k 5 --candidates 30 --out eval/results

The default split is "dev". The holdout questions are for one final check with a frozen
configuration, so "holdout" and "all" are refused unless --confirm-holdout is given.
Results are saved as JSON (ranked chunk ids and scores, never chunk text).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.retrieval.models import RetrievedChunk

from .metrics import judge_chunk
from .questions import QUESTIONS_FILE, Question, load_questions, select_split
from .report import QuestionRun, RankEntry, Results, render_report, results_from_json, results_to_json
from .validate_questions import Corpus, doc_passes_filters, load_corpus, to_search_filters

KNOWN_MODES = ("vector", "keyword", "hybrid")
DEPTH = 10  # MRR@10 needs the top ten
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

SearchFn = Callable[[Question, str], list[RetrievedChunk]]


class EvalError(Exception):
    """A problem to show the user as a plain message."""


def parse_modes(text: str, known: tuple[str, ...] = KNOWN_MODES) -> list[str]:
    modes = [m.strip() for m in text.split(",") if m.strip()]
    if not modes:
        raise EvalError("give at least one mode, for example --modes vector,keyword,hybrid")
    unknown = [m for m in modes if m not in known]
    if unknown:
        raise EvalError(f"unknown mode(s) {unknown}; known modes: {', '.join(known)}")
    if len(set(modes)) != len(modes):
        raise EvalError("each mode may be listed only once")
    return modes


def check_split_allowed(split: str, confirmed: bool) -> None:
    if split != "dev" and not confirmed:
        raise EvalError(
            f"the {split!r} split includes the holdout questions, which are reserved for one final check "
            "with a frozen configuration. Use --split dev, or pass --confirm-holdout when you mean it."
        )


def check_documents(questions: list[Question], corpus: Corpus) -> None:
    """Every document a question needs must be in the database, or the results would be meaningless."""
    missing: dict[str, list[str]] = {}
    for q in questions:
        for doc in q.acceptable_docs:
            if doc not in corpus:
                missing.setdefault(doc, []).append(q.id)
    if missing:
        lines = [f"  {doc} (needed by {', '.join(ids)})" for doc, ids in sorted(missing.items())]
        raise EvalError("these documents are missing from the database:\n" + "\n".join(lines))


def run_questions(
    questions: list[Question], modes: list[str], search_fn: SearchFn, corpus: Corpus | None = None
) -> Results:
    """Run every question in every mode, timing each search and judging each returned chunk."""
    results: Results = {mode: {} for mode in modes}
    for q in questions:
        filters = to_search_filters(q.filters) if q.filters else None
        for mode in modes:
            started = perf_counter()
            chunks = search_fn(q, mode)
            seconds = perf_counter() - started
            entries = []
            for chunk in chunks[:DEPTH]:
                slot, doc_ok = judge_chunk(chunk.path, chunk.text, q.slots) if q.answerable else (None, False)
                filter_ok = None
                if filters is not None and corpus is not None and chunk.path in corpus:
                    filter_ok = doc_passes_filters(chunk.path, corpus[chunk.path], filters)
                entries.append(
                    RankEntry(
                        chunk_id=chunk.chunk_id, path=chunk.path, slot=slot, doc_ok=doc_ok, score=float(chunk.score),
                        vector_rank=chunk.vector_rank, keyword_rank=chunk.keyword_rank, filter_ok=filter_ok,
                    )
                )
            results[mode][q.id] = QuestionRun(q.id, mode, entries, seconds)
    return results


def build_search_fn(conn, embedder, candidates: int) -> SearchFn:
    from src.retrieval.hybrid import search

    def run(question: Question, mode: str) -> list[RetrievedChunk]:
        filters = to_search_filters(question.filters) if question.filters else None
        return search(conn, embedder, question.question, k=DEPTH, mode=mode, filters=filters, candidates=candidates)

    return run


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def result_filename(modes: list[str], split: str, when: datetime) -> str:
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    label = "-".join(m.replace("+", "_") for m in modes)
    return f"eval_{stamp}_{label}_{split}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the question set.")
    parser.add_argument("--modes", default="vector,keyword,hybrid", help="comma-separated: vector, keyword, hybrid")
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="dev", help="which questions (default dev)")
    parser.add_argument("--confirm-holdout", action="store_true", help="allow the holdout split (final check only)")
    parser.add_argument("--k", type=int, default=5, help="cutoff for the win/loss and missed-question lists (default 5)")
    parser.add_argument("--candidates", type=int, default=30, help="candidates per method for hybrid (default 30)")
    parser.add_argument("--out", type=Path, default=Path("eval/results"), help="folder for the result JSON")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE, help="question file")
    parser.add_argument("--markdown", type=Path, help="also write the tables to this file")
    parser.add_argument("--from-json", type=Path, help="print the tables from a saved result file (no database needed)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        modes = parse_modes(args.modes)
        check_split_allowed(args.split, args.confirm_holdout)
        questions = select_split(load_questions(args.questions), args.split)

        if args.from_json:
            data = json.loads(args.from_json.read_text(encoding="utf-8"))
            results = results_from_json(data)
            ids = set(next(iter(results.values())))
            questions = [q for q in questions if q.id in ids]
            modes = list(results)
            report = render_report(questions, results, modes, args.k)
            print(report)
            return 0

        from src.db.connection import get_connection
        from src.ingestion.embedder import Embedder

        with get_connection() as conn:
            corpus = load_corpus(conn)
            check_documents(questions, corpus)
            documents = len(corpus)
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

            embedder = Embedder()
            if any(m in ("vector", "hybrid") for m in modes):
                embedder.embed_query("warm up")  # load the model before anything is timed
            search_fn = build_search_fn(conn, embedder, args.candidates)
            results = run_questions(questions, modes, search_fn, corpus)
    except EvalError as error:
        print(f"Cannot run the evaluation: {error}", file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    meta = {
        "created_utc": now.isoformat(timespec="seconds"),
        "split": args.split,
        "modes": modes,
        "k": args.k,
        "candidates": args.candidates,
        "depth": DEPTH,
        "embedding_model": EMBEDDING_MODEL,
        "documents": documents,
        "chunks": chunks,
        "questions_file": str(args.questions.name),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "note": "ranked chunk ids and scores only; no chunk text",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / result_filename(modes, args.split, now)
    out_path.write_text(json.dumps(results_to_json(questions, results, meta), indent=1), encoding="utf-8")

    header = (
        f"# Evaluation: {len(questions)} questions ({args.split} split), modes {', '.join(modes)}\n\n"
        f"Database: {documents} documents, {chunks} chunks. Embedding model {EMBEDDING_MODEL}. "
        f"Hybrid candidates per method: {args.candidates}. Top {DEPTH} judged per question.\n"
        f"Questions file {args.questions.name} (sha256 {meta['questions_sha256'][:12]}), commit {meta['git_commit']}.\n"
    )
    report = render_report(questions, results, modes, args.k)
    print(header + "\n" + report)
    print(f"Saved: {out_path}")
    if args.markdown:
        args.markdown.write_text(header + "\n" + report, encoding="utf-8")
        print(f"Tables written to {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
