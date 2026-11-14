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

    @property
    def answerable(self) -> bool:
        return bool(self.expected)

    @property
    def expected_docs(self) -> tuple[str, ...]:
        return tuple(e.doc for e in self.expected)


def parse_question(raw: dict) -> Question:
    """Turn one JSON item into a Question. Raises ValueError with a clear message if it is malformed."""
    if not isinstance(raw, dict):
        raise ValueError("a question must be a JSON object")
    missing = [key for key in ("id", "category", "question", "expected", "holdout") if key not in raw]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")
    if not isinstance(raw["expected"], list):
        raise ValueError("'expected' must be a list")
    expected = []
    for item in raw["expected"]:
        if not isinstance(item, dict) or "doc" not in item or not isinstance(item.get("evidence"), list):
            raise ValueError("each 'expected' entry needs 'doc' and a list 'evidence'")
        expected.append(Expected(doc=item["doc"], evidence=tuple(item["evidence"])))
    return Question(
        id=raw["id"],
        category=raw["category"],
        question=raw["question"],
        expected=tuple(expected),
        holdout=raw["holdout"],
        filters=raw.get("filters") or {},
        notes=raw.get("notes", ""),
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
