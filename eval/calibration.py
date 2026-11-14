"""Calibration analysis for a possible "not found" cutoff. Analysis only: nothing here is used by the app.

For each mode, the top-1 score of every question in a saved result file is split into answerable and
unanswerable questions. A threshold would refuse a question when its top-1 score is below the
threshold. For each threshold the table shows the share of unanswerable questions that would be
refused (good) against the share of answerable questions that would be refused by mistake (bad).

Usage:
    python -m eval.calibration eval/results/<file>.json --modes hybrid,vector+rerank
"""

import argparse
import json
import sys
from pathlib import Path

from .metrics import median
from .report import md_table


def auroc(positives: list[float], negatives: list[float]) -> float:
    """Chance that a random answerable question has a higher top-1 score than a random unanswerable
    one (ties count half). 0.5 means no separation at all; 1.0 means perfect separation."""
    if not positives or not negatives:
        raise ValueError("both groups are needed")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def split_scores(data: dict, mode: str) -> tuple[list[float], list[float]]:
    """(answerable top-1 scores, unanswerable top-1 scores) for a mode."""
    answerable, unanswerable = [], []
    for q in data["questions"]:
        score = q["runs"][mode]["top1_score"]
        if score is None:
            continue
        (answerable if q["answerable"] else unanswerable).append(float(score))
    return answerable, unanswerable


def threshold_rows(answerable: list[float], unanswerable: list[float], thresholds: list[float], digits: int) -> list[list[str]]:
    rows = []
    for t in thresholds:
        refused_un = sum(s < t for s in unanswerable)
        refused_an = sum(s < t for s in answerable)
        rows.append(
            [
                f"{t:.{digits}f}",
                f"{refused_un}/{len(unanswerable)} ({100 * refused_un / len(unanswerable):.0f}%)",
                f"{refused_an}/{len(answerable)} ({100 * refused_an / len(answerable):.0f}%)",
            ]
        )
    return rows


def evenly_spaced(low: float, high: float, steps: int) -> list[float]:
    return [low + (high - low) * i / steps for i in range(steps + 1)]


def render(data: dict, modes: list[str]) -> str:
    parts = []
    summary = []
    for mode in modes:
        a, u = split_scores(data, mode)
        summary.append(
            [
                mode,
                f"{min(a):.3f} / {median(a):.3f} / {max(a):.3f} (n={len(a)})",
                f"{min(u):.3f} / {median(u):.3f} / {max(u):.3f} (n={len(u)})",
                f"{auroc(a, u):.2f}",
            ]
        )
    parts.append(
        md_table(["mode", "answerable: min / median / max", "unanswerable: min / median / max", "AUROC (answerable above unanswerable)"], summary)
    )
    for mode in modes:
        a, u = split_scores(data, mode)
        low, high = min(a + u), max(a + u)
        digits = 4 if high < 1 else 1
        parts.append(
            f"**{mode}**: threshold = refuse when the top-1 score is below it\n\n"
            + md_table(
                ["threshold", "unanswerable refused (correct)", "answerable refused (wrong)"],
                threshold_rows(a, u, evenly_spaced(low, high, 10), digits),
            )
        )
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top-1 score distributions and a threshold table (analysis only).")
    parser.add_argument("result", type=Path, help="a saved result file from eval.run_eval")
    parser.add_argument("--modes", required=True, help="comma-separated modes present in the file")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = json.loads(args.result.read_text(encoding="utf-8"))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    missing = [m for m in modes if m not in data["questions"][0]["runs"]]
    if missing:
        print(f"modes not in the file: {missing}", file=sys.stderr)
        return 2
    print(render(data, modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
