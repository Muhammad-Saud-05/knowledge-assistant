"""Recorded rankings and the tables computed from them. Pure: no database, no model.

A RankEntry holds what is needed to score and audit one retrieved chunk (ids, document path,
which slot it filled, scores) but never the chunk text, so saved result files contain no
document content.
"""

from dataclasses import asdict, dataclass

from .metrics import (
    QuestionScore,
    aggregate,
    count_text,
    median,
    paired_bootstrap_ci,
    percentile,
    score_from_flags,
)
from .questions import DIFFICULTIES, Question

HEADLINE_CATEGORIES = ("exact_term", "paraphrase", "multi_doc")


@dataclass
class RankEntry:
    chunk_id: int
    path: str
    slot: int | None  # which required slot the chunk filled (None: not relevant)
    doc_ok: bool  # the chunk comes from an acceptable document
    score: float  # the score of the method that produced it (cosine, ts_rank_cd or fused)
    vector_rank: int | None
    keyword_rank: int | None
    filter_ok: bool | None  # None when the question has no filter


@dataclass
class QuestionRun:
    qid: str
    mode: str
    entries: list[RankEntry]
    seconds: float

    @property
    def top1_score(self) -> float | None:
        return self.entries[0].score if self.entries else None


Results = dict[str, dict[str, QuestionRun]]  # mode -> question id -> run


def score_run(question: Question, run: QuestionRun) -> QuestionScore | None:
    if not question.answerable:
        return None
    return score_from_flags([e.slot for e in run.entries], [e.doc_ok for e in run.entries], len(question.slots))


# ------------------------------------------------------------------ JSON form


def results_to_json(questions: list[Question], results: Results, meta: dict) -> dict:
    """A JSON-ready dict: per question, per mode, the ranked ids and scores. No chunk text."""
    modes = list(results)
    out_questions = []
    for q in questions:
        runs = {}
        for mode in modes:
            run = results[mode][q.id]
            runs[mode] = {
                "seconds": round(run.seconds, 4),
                "top1_score": run.top1_score,
                "entries": [asdict(e) for e in run.entries],
            }
        out_questions.append(
            {
                "id": q.id,
                "category": q.category,
                "difficulty": q.difficulty,
                "headline": q.headline,
                "holdout": q.holdout,
                "answerable": q.answerable,
                "expected_docs": list(q.expected_docs),
                "runs": runs,
            }
        )
    return {**meta, "questions": out_questions}


def results_from_json(data: dict) -> Results:
    results: Results = {}
    for item in data["questions"]:
        for mode, run in item["runs"].items():
            entries = [RankEntry(**e) for e in run["entries"]]
            results.setdefault(mode, {})[item["id"]] = QuestionRun(item["id"], mode, entries, run["seconds"])
    return results


# --------------------------------------------------------------- table helpers


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def ci_text(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:.2f}, {interval[1]:.2f}]"


def short(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def headline_questions(questions: list[Question]) -> list[Question]:
    """Answerable questions that count in the main results table."""
    return [q for q in questions if q.answerable and q.headline and q.category in HEADLINE_CATEGORIES]


def _scores(questions: list[Question], results: Results, mode: str) -> list[QuestionScore]:
    return [score_run(q, results[mode][q.id]) for q in questions]


def _hit_values(scores: list[QuestionScore], k: int = 5) -> list[float]:
    return [1.0 if s.hit[k] else 0.0 for s in scores]


# ---------------------------------------------------------------------- tables


def metrics_table(title_rows: list[tuple[str, list[Question]]], results: Results, modes: list[str]) -> str:
    """One row per (group, mode): counts with percentages for hit@1/3/5, MRR@10, doc_hit@5."""
    rows = []
    for label, qs in title_rows:
        if not qs:
            continue
        for mode in modes:
            scores = _scores(qs, results, mode)
            agg = aggregate(scores)
            n = agg["n"]
            rows.append(
                [label, f"n={len(qs)}", mode, count_text(agg["hit1"], n), count_text(agg["hit3"], n),
                 count_text(agg["hit5"], n), f"{agg['mrr10']:.3f}", count_text(agg["doc_hit5"], n)]
            )
    return md_table(["group", "n", "mode", "hit@1", "hit@3", "hit@5", "MRR@10", "doc_hit@5"], rows)


def main_tables(questions: list[Question], results: Results, modes: list[str]) -> str:
    qs = headline_questions(questions)
    multi = [q for q in qs if q.category == "multi_doc"]
    rows, ci_rows = [], []
    for mode in modes:
        scores = _scores(qs, results, mode)
        agg = aggregate(scores)
        n = agg["n"]
        coverage = [s.coverage[5] for s in _scores(multi, results, mode)]
        times = [results[mode][q.id].seconds * 1000 for q in qs]
        rows.append(
            [mode, count_text(agg["hit1"], n), count_text(agg["hit3"], n), count_text(agg["hit5"], n),
             f"{agg['mrr10']:.3f}", count_text(agg["doc_hit5"], n),
             f"{sum(coverage) / len(coverage):.2f} (n={len(coverage)})" if coverage else "-"]
        )
        ci_rows.append(
            [mode, ci_text(agg["hit5_ci"]), ci_text(agg["mrr10_ci"]), f"{median(times):.0f}", f"{percentile(times, 0.95):.0f}"]
        )
    first = md_table(["mode", "hit@1", "hit@3", "hit@5", "MRR@10", "doc_hit@5", "multi_doc coverage@5"], rows)
    second = md_table(
        ["mode", "hit@5 95% CI", "MRR@10 95% CI", "median ms/question", "p95 ms/question"], ci_rows
    )
    return first + "\n\n" + second


def set_comparison_table(questions: list[Question], results: Results, modes: list[str]) -> str:
    """hit@5 and MRR@10 for the frozen v1 questions next to the same measures on v2 (v1 plus the
    later additions) and on the additions alone. Main-table questions only."""
    qs = headline_questions(questions)
    groups = [
        ("v1", [q for q in qs if q.added_in == "v1"]),
        ("v2 (v1 + additions)", qs),
        ("additions only", [q for q in qs if q.added_in != "v1"]),
    ]
    rows = []
    for mode in modes:
        row = [mode]
        for _, group in groups:
            if not group:
                row += ["-", "-"]
                continue
            agg = aggregate(_scores(group, results, mode))
            row += [count_text(agg["hit5"], agg["n"]), f"{agg['mrr10']:.3f}"]
        rows.append(row)
    headers = ["mode"]
    for label, group in groups:
        headers += [f"{label} (n={len(group)}) hit@5", f"{label} MRR@10"]
    return md_table(headers, rows)


def category_table(questions: list[Question], results: Results, modes: list[str]) -> str:
    qs = headline_questions(questions)
    groups = [(c, [q for q in qs if q.category == c]) for c in HEADLINE_CATEGORIES]
    return metrics_table(groups, results, modes)


def difficulty_table(questions: list[Question], results: Results, modes: list[str]) -> str:
    qs = headline_questions(questions)
    groups = [(d, [q for q in qs if q.difficulty == d]) for d in DIFFICULTIES]
    return metrics_table(groups, results, modes)


def filtered_table(questions: list[Question], results: Results, modes: list[str]) -> str:
    filtered = [q for q in questions if q.category == "filtered"]
    if not filtered:
        return "(no filtered questions in this split)"
    rows = []
    for mode in modes:
        scores = _scores(filtered, results, mode)
        agg = aggregate(scores)
        returned = [len(results[mode][q.id].entries) for q in filtered]
        violations = sum(1 for q in filtered for e in results[mode][q.id].entries if e.filter_ok is False)
        rows.append(
            [mode, f"n={len(filtered)}", count_text(agg["hit5"], agg["n"]), f"{agg['mrr10']:.3f}",
             f"{min(returned)}-{max(returned)}", str(violations)]
        )
    summary = md_table(
        ["mode", "n", "hit@5", "MRR@10", "results returned (min-max)", "results that violate the filter"], rows
    )
    detail_rows = []
    for q in filtered:
        row = [q.id, json_filters(q)]
        for mode in modes:
            score = score_run(q, results[mode][q.id])
            row.append(f"rank {score.first_relevant_rank}" if score.first_relevant_rank else "miss")
        detail_rows.append(row)
    detail = md_table(["id", "filter"] + modes, detail_rows)
    return summary + "\n\n" + detail


def json_filters(q: Question) -> str:
    return ", ".join(f"{k}={v}" for k, v in q.filters.items())


def default_pairs(modes: list[str]) -> list[tuple[str, str]]:
    """(A, B) pairs to compare as A minus B: hybrid against every other mode."""
    if "hybrid" not in modes:
        return [(modes[i], modes[j]) for i in range(len(modes)) for j in range(i)]
    return [("hybrid", m) for m in modes if m != "hybrid"]


def paired_table(questions: list[Question], results: Results, pairs: list[tuple[str, str]], groups=None) -> str:
    qs = headline_questions(questions)
    rows = []
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        sets = [("all headline", qs)] + (groups or [])
        for label, group in sets:
            if not group:
                continue
            sa, sb = _scores(group, results, a), _scores(group, results, b)
            for metric, va, vb in (
                ("hit@5", _hit_values(sa), _hit_values(sb)),
                ("MRR@10", [s.mrr10 for s in sa], [s.mrr10 for s in sb]),
            ):
                diff, low, high = paired_bootstrap_ci(va, vb)
                better = sum(x > y for x, y in zip(va, vb))
                worse = sum(x < y for x, y in zip(va, vb))
                verdict = "clear" if low > 0 or high < 0 else "not clear"
                rows.append(
                    [f"{a} - {b}", label, f"n={len(group)}", metric, f"{diff:+.3f}", ci_text((low, high)),
                     f"{better} better / {worse} worse / {len(group) - better - worse} same", verdict]
                )
    return md_table(
        ["comparison", "questions", "n", "metric", "mean difference", "paired 95% CI", "questions", "interval excludes 0?"],
        rows,
    )


def win_loss_lists(questions: list[Question], results: Results, pairs: list[tuple[str, str]], k: int = 5) -> str:
    qs = headline_questions(questions)
    by_id = {q.id: q for q in qs}
    lines = []
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        a_wins, b_wins, both, neither = [], [], [], []
        for q in qs:
            ha = score_run(q, results[a][q.id]).hit[k]
            hb = score_run(q, results[b][q.id]).hit[k]
            tag = f"{q.id} ({q.category})"
            (a_wins if ha and not hb else b_wins if hb and not ha else both if ha else neither).append(tag)
        lines.append(f"**{a} vs {b} at hit@{k}**\n")
        lines.append(f"- {a} hits, {b} misses ({len(a_wins)}): {', '.join(a_wins) or 'none'}")
        lines.append(f"- {b} hits, {a} misses ({len(b_wins)}): {', '.join(b_wins) or 'none'}")
        lines.append(f"- both hit: {len(both)}; both miss: {len(neither)} ({', '.join(neither) or 'none'})\n")
    return "\n".join(lines)


def missed_lists(questions: list[Question], results: Results, modes: list[str], k: int = 5) -> str:
    qs = headline_questions(questions)
    lines = []
    for mode in modes:
        missed = [q for q in qs if not score_run(q, results[mode][q.id]).hit[k]]
        lines.append(f"**{mode}: {len(missed)} of {len(qs)} missed at k={k}**\n")
        for q in missed:
            lines.append(f"- {q.id}: " + ", ".join(short(d) for d in q.expected_docs))
        lines.append("")
    return "\n".join(lines)


def top1_tables(questions: list[Question], results: Results, modes: list[str]) -> str:
    """Top-1 score of answerable versus unanswerable questions, per mode. Summary numbers only:
    no threshold is chosen here."""
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    if not unanswerable:
        return "(no unanswerable questions in this split)"

    def stats(group: list[Question], mode: str) -> str:
        values = [results[mode][q.id].top1_score for q in group if results[mode][q.id].top1_score is not None]
        if not values:
            return "-"
        return f"{min(values):.3f} / {median(values):.3f} / {max(values):.3f}"

    rows = [[mode, f"{stats(answerable, mode)} (n={len(answerable)})", f"{stats(unanswerable, mode)} (n={len(unanswerable)})"] for mode in modes]
    summary = md_table(["mode (score)", "answerable: min / median / max", "unanswerable: min / median / max"], rows)
    detail_rows = [[q.id] + [f"{results[m][q.id].top1_score:.3f}" if results[m][q.id].top1_score is not None else "-" for m in modes] for q in unanswerable]
    detail = md_table(["unanswerable id"] + [f"{m} top-1" for m in modes], detail_rows)
    return summary + "\n\n" + detail


def timing_note(modes: list[str]) -> str:
    return (
        "Scores are not comparable across modes: vector = cosine similarity, keyword = ts_rank_cd, "
        "hybrid = reciprocal-rank-fusion score (a ranking value, not a similarity)."
    )


def render_report(questions: list[Question], results: Results, modes: list[str], k: int = 5) -> str:
    pairs = default_pairs(modes)
    headline = headline_questions(questions)
    by_difficulty = [(f"{d}", [q for q in headline if q.difficulty == d]) for d in DIFFICULTIES]
    by_category = [(c, [q for q in headline if q.category == c]) for c in HEADLINE_CATEGORIES]
    sections = [
        f"## Main results (exact_term, paraphrase, multi_doc; n={len(headline)} questions)\n\n{main_tables(questions, results, modes)}",
        *(
            [f"## Question set v1 next to v2 (main-table questions)\n\n{set_comparison_table(questions, results, modes)}"]
            if any(q.added_in != "v1" for q in questions)
            else []
        ),
        f"## By category\n\n{category_table(questions, results, modes)}",
        f"## By difficulty bucket (lexical overlap of the question with its best passage)\n\n{difficulty_table(questions, results, modes)}",
        f"## Paired differences (same questions resampled for both modes)\n\n{paired_table(questions, results, pairs)}\n\n"
        f"### Paired differences by category\n\n{paired_table(questions, results, pairs, groups=by_category)}\n\n"
        f"### Paired differences by difficulty bucket\n\n{paired_table(questions, results, pairs, groups=by_difficulty)}",
        f"## Wins and losses per question\n\n{win_loss_lists(questions, results, pairs, k)}",
        f"## Questions each mode missed at k={k}\n\n{missed_lists(questions, results, modes, k)}",
        f"## Filtered questions: filter-correctness check\n\n{filtered_table(questions, results, modes)}",
        f"## Top-1 scores: answerable versus unanswerable questions\n\n{timing_note(modes)}\n\n{top1_tables(questions, results, modes)}",
    ]
    return "\n\n".join(sections) + "\n"
