"""Retrieval metrics. Pure functions: no database, no model.

A retrieved chunk is RELEVANT to a question if it comes from an expected document (or, for a
single-document question, an also_acceptable one) AND its text contains at least one of that
document's evidence phrases (compared after normalize()). That is a deliberately strict,
mechanical judgement: it only credits the passages the author pointed at or found by reading,
so another passage that also answers the question may still count as a miss.
"""

import math
import random
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from .questions import Expected

# ---------------------------------------------------------------- normalisation

# Characters that normalisation turns into plain ones (written as code points so the
# source stays plain ASCII): curly quotes, no-break space, en and em dashes.
_PUNCTUATION = {
    0x2018: "'",
    0x2019: "'",
    0x201C: '"',
    0x201D: '"',
    0x00A0: " ",
    0x2013: "-",
    0x2014: "-",
}
_MARKDOWN_CHARACTERS = re.compile(r"[*_`~]")  # emphasis, code and strike-through marks


def normalize(text: str) -> str:
    """The one place where text is made comparable.

    Lowercase, standard Unicode forms (so ligatures and look-alike characters match), curly
    quotes and dashes made plain, markdown emphasis characters (* _ ` ~) removed, and all runs
    of whitespace collapsed to one space. Underscores go too, so `max_age` and "max age"
    compare equal; this only ever makes a match slightly more lenient.
    """
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    text = _MARKDOWN_CHARACTERS.sub("", text).lower()
    return re.sub(r"\s+", " ", text).strip()


STOP_WORDS = frozenset(
    """
    about above after again against also always among another because been before being below between
    both cannot could does doing done down during each either else even ever every from further have
    having here herself himself into itself just least less like made make many might more most much
    must myself never none once only other ought over please really said same shall should since some
    such than that their theirs them themselves then there these they this those through today under
    until upon very want wants well were what when where whether which while whom whose will with within
    without would your yours yourself yourselves
    """.split()
)


def content_words(text: str) -> set[str]:
    """Lowercase words of 4 or more letters that are not stop words (used to measure lexical overlap)."""
    return {w for w in re.findall(r"[a-z]{4,}", normalize(text)) if w not in STOP_WORDS}


# -------------------------------------------------------------- judging chunks

Slots = Sequence[Sequence[Expected]]


def judge_chunk(path: str, text: str, slots: Slots) -> tuple[int | None, bool]:
    """Judge one retrieved chunk against what the question needs.

    `slots` is Question.slots: a multi-document question has one slot per expected document;
    a single-document question has one slot holding its expected document and any
    also_acceptable documents. A chunk fills a slot if it comes from one of the slot's documents
    and contains one of THAT document's evidence phrases.

    Returns (index of the slot the chunk fills, or None; whether the chunk comes from any
    acceptable document at all).
    """
    in_doc = False
    haystack = None
    for index, slot in enumerate(slots):
        for item in slot:
            if item.doc != path:
                continue
            in_doc = True
            haystack = normalize(text) if haystack is None else haystack
            if any(normalize(phrase) in haystack for phrase in item.evidence):
                return index, True
    return None, in_doc


# ------------------------------------------------------------ scoring one question

CUTOFFS = (1, 3, 5)
MRR_DEPTH = 10


@dataclass(frozen=True)
class QuestionScore:
    hit: dict[int, bool]  # k -> a relevant chunk in the top k (a multi-document question needs EVERY slot filled)
    coverage: dict[int, float]  # k -> share of the slots filled by a relevant chunk in the top k
    mrr10: float
    doc_hit5: bool  # any top-5 chunk from an acceptable document (the lenient measure)
    first_relevant_rank: int | None  # 1-based, within the top 10


def score_from_flags(matched: list[int | None], doc_ok: list[bool], n_slots: int) -> QuestionScore:
    """Score a ranked list given, for each rank, which slot it filled (or None) and whether it
    came from an acceptable document."""
    if n_slots < 1:
        raise ValueError("an unanswerable question has no retrieval score")
    coverage = {k: len({m for m in matched[:k] if m is not None}) / n_slots for k in CUTOFFS}
    hit = {k: coverage[k] >= 1.0 for k in CUTOFFS}
    first = next((i + 1 for i, m in enumerate(matched[:MRR_DEPTH]) if m is not None), None)
    return QuestionScore(
        hit=hit,
        coverage=coverage,
        mrr10=1.0 / first if first else 0.0,
        doc_hit5=any(doc_ok[:5]),
        first_relevant_rank=first,
    )


def score_ranking(chunks: list[tuple[str, str]], slots: Slots) -> QuestionScore:
    """Score a ranked list of (document path, chunk text) pairs."""
    judged = [judge_chunk(path, text, slots) for path, text in chunks]
    return score_from_flags([m for m, _ in judged], [d for _, d in judged], len(slots))


# ----------------------------------------------------------------- aggregation


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: list[float], resamples: int = 1000, seed: int = 7, level: float = 0.95) -> tuple[float, float]:
    """A percentile bootstrap confidence interval for the mean of `values`.

    Questions are resampled with replacement; the seed is fixed so the interval is
    reproducible. With few questions the interval is wide, and that is the honest picture.
    """
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    tail = (1 - level) / 2
    low = means[max(0, math.floor(tail * resamples))]
    high = means[min(resamples - 1, math.ceil((1 - tail) * resamples) - 1)]
    return (low, high)


def aggregate(scores: list[QuestionScore]) -> dict:
    """Summary numbers for a set of answerable questions."""
    n = len(scores)
    hit5 = [1.0 if s.hit[5] else 0.0 for s in scores]
    mrr = [s.mrr10 for s in scores]
    result = {
        "n": n,
        "hit1": sum(s.hit[1] for s in scores),
        "hit3": sum(s.hit[3] for s in scores),
        "hit5": sum(s.hit[5] for s in scores),
        "doc_hit5": sum(s.doc_hit5 for s in scores),
        "mrr10": mean(mrr),
        "hit5_ci": bootstrap_ci(hit5),
        "mrr10_ci": bootstrap_ci(mrr),
    }
    return result


def count_text(hits: int, n: int) -> str:
    """'27/34 (79.4%)': always show the counts next to a percentage."""
    return f"{hits}/{n} ({100 * hits / n:.1f}%)" if n else "0/0 (n/a)"
