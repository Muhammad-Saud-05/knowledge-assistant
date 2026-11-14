"""The evaluation questions: their shape, loading them, and the dev/holdout split."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

QUESTIONS_FILE = Path(__file__).with_name("questions.json")

CATEGORIES = ("exact_term", "paraphrase", "multi_doc", "unanswerable", "filtered")
EXPECTED_COUNTS = {"exact_term": 10, "paraphrase": 12, "multi_doc": 8, "unanswerable": 6, "filtered": 4}
FILTER_FIELDS = ("source_types", "date_from", "date_to", "path_prefix")
ID_PATTERN = re.compile(r"^q\d{2}$")
MAX_EVIDENCE_WORDS = 10
LOW_OVERLAP_MAX = 0.25  # a question whose overlap is at most this is "low_overlap"; above it, "high_overlap"
DIFFICULTIES = ("low_overlap", "high_overlap")
KNOWN_FIELDS = {
    "id", "category", "question", "filters", "expected", "also_acceptable",
    "headline", "holdout", "lexical_overlap", "difficulty", "notes",
}
# Order in which fields are written to the question file.
FIELD_ORDER = (
    "id", "category", "question", "filters", "expected", "also_acceptable",
    "headline", "holdout", "lexical_overlap", "difficulty", "notes",
)


@dataclass(frozen=True)
class Expected:
    """One document that should be found, and short phrases copied from it that prove a chunk is the right one."""

    doc: str  # exactly as stored in documents.path
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    expected: tuple[Expected, ...]
    holdout: bool
    filters: dict = field(default_factory=dict)
    notes: str = ""
    # Other passages that genuinely answer a single-document question (found by reading the text).
    also_acceptable: tuple[Expected, ...] = ()
    # Whether the question counts in the main results table (filtered questions have their own table).
    headline: bool = True
    lexical_overlap: float | None = None
    difficulty: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.expected)

    @property
    def expected_docs(self) -> tuple[str, ...]:
        return tuple(e.doc for e in self.expected)

    @property
    def slots(self) -> tuple[tuple[Expected, ...], ...]:
        """What must be found. A multi-document question has one slot per expected document.
        A single-document question has one slot that its expected document or any
        also_acceptable document can fill."""
        if len(self.expected) > 1:
            return tuple((e,) for e in self.expected)
        return ((*self.expected, *self.also_acceptable),) if self.expected else ()

    @property
    def acceptable_docs(self) -> tuple[str, ...]:
        return tuple(e.doc for slot in self.slots for e in slot)


def _parse_expected(items: object, what: str) -> tuple[Expected, ...]:
    if not isinstance(items, list):
        raise ValueError(f"'{what}' must be a list")
    parsed = []
    for item in items:
        if not isinstance(item, dict) or "doc" not in item or not isinstance(item.get("evidence"), list):
            raise ValueError(f"each '{what}' entry needs 'doc' and a list 'evidence'")
        parsed.append(Expected(doc=item["doc"], evidence=tuple(item["evidence"])))
    return tuple(parsed)


def parse_question(raw: dict) -> Question:
    """Turn one JSON item into a Question. Raises ValueError with a clear message if it is malformed."""
    if not isinstance(raw, dict):
        raise ValueError("a question must be a JSON object")
    missing = [key for key in ("id", "category", "question", "expected", "holdout") if key not in raw]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")
    return Question(
        id=raw["id"],
        category=raw["category"],
        question=raw["question"],
        expected=_parse_expected(raw["expected"], "expected"),
        holdout=raw["holdout"],
        filters=raw.get("filters") or {},
        notes=raw.get("notes", ""),
        also_acceptable=_parse_expected(raw.get("also_acceptable", []), "also_acceptable"),
        headline=raw.get("headline", True),
        lexical_overlap=raw.get("lexical_overlap"),
        difficulty=raw.get("difficulty"),
    )


def load_questions(path: Path | str = QUESTIONS_FILE) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("the question file must contain a JSON list")
    return [parse_question(item) for item in raw]


def select_split(questions: list[Question], split: str) -> list[Question]:
    """'dev' = everything except the holdout, 'holdout' = only the holdout, 'all' = both."""
    if split == "all":
        return list(questions)
    if split == "dev":
        return [q for q in questions if not q.holdout]
    if split == "holdout":
        return [q for q in questions if q.holdout]
    raise ValueError(f"unknown split {split!r}; use all, dev or holdout")


def difficulty_for(overlap: float) -> str:
    return "low_overlap" if overlap <= LOW_OVERLAP_MAX else "high_overlap"


# ---------------------------------------------------------- writing the file


def _entries(items: list[dict], indent: str) -> str:
    """The 'expected' style: one entry on the same line, several entries one per line."""
    rendered = [json.dumps(item, ensure_ascii=False) for item in items]
    if len(rendered) <= 1:
        return "[" + ", ".join(rendered) + "]"
    return "[\n" + ",\n".join(f"{indent}  {r}" for r in rendered) + f"\n{indent}]"


def format_questions(items: list[dict]) -> str:
    """The question file text, in the same hand-readable layout the file has always had."""
    blocks = []
    for item in items:
        unknown = set(item) - KNOWN_FIELDS
        if unknown:
            raise ValueError(f"{item.get('id')}: unknown field(s) {sorted(unknown)}")
        lines = []
        for key in FIELD_ORDER:
            if key not in item:
                continue
            value = item[key]
            if key in ("expected", "also_acceptable"):
                text = _entries(value, "    ")
            else:
                text = json.dumps(value, ensure_ascii=False)
            lines.append(f'    "{key}": {text}')
        blocks.append("  {\n" + ",\n".join(lines) + "\n  }")
    return "[\n" + ",\n".join(blocks) + "\n]\n"
