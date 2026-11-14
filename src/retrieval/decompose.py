"""Split a question that asks for several things into sub-questions, and merge their results.

A question such as "which URL do I call, and how should I declare the function?" needs passages from
different places. Searching with the whole question can return passages for only one of the asks, so the
question is split, each sub-question is searched and reranked on its own, and the lists are merged.

Two ways to split (both pre-registered in docs/eval_report.md, Phase 2b, with their exact rules):
  RuleDecomposer  a fixed text rule: question marks, semicolons, and "and" before a question word
  LLMDecomposer   the local language model, with any failure falling back to the original question

A decomposer's decompose(question) always returns a non-empty list; a list of one means "not split".
"""

import json
import re
from dataclasses import dataclass

from .filters import SearchFilters
from .models import RetrievedChunk

MAX_SUB_QUESTIONS = 3
MIN_WORDS = 3
MAX_SUB_QUESTION_CHARS = 300

STARTERS = ("how", "what", "which", "why", "when", "where", "who", "can", "could", "should", "do", "does", "is", "are", "will", "would")
_AFTER_QUESTION_MARK = re.compile(r"(?<=\?)\s+(?=\S)")
_SEMICOLON = re.compile(r"\s*;\s*")
_AND_BEFORE_STARTER = re.compile(r",?\s+and\s+(?=(?:" + "|".join(STARTERS) + r")\b)", re.IGNORECASE)


def split_by_rule(question: str) -> list[str]:
    """The parts of a question that clearly contains two or more asks; [question] if it does not."""
    text = " ".join(question.split())
    parts = _AFTER_QUESTION_MARK.split(text)
    parts = [q for part in parts for q in _SEMICOLON.split(part)]
    parts = [q for part in parts for q in _AND_BEFORE_STARTER.split(part)]
    ends_with_question = text.endswith("?")
    cleaned = []
    for part in parts:
        part = part.strip(" ,;")
        if part.lower().startswith("and "):
            part = part[4:].strip(" ,;")
        if ends_with_question and part and not part.endswith("?"):
            part += "?"
        if len(part.rstrip("?").split()) >= MIN_WORDS:
            cleaned.append(part)
    if len(cleaned) < 2:
        return [question]
    return cleaned[:MAX_SUB_QUESTIONS]


class RuleDecomposer:
    name = "rule"

    def decompose(self, question: str) -> list[str]:
        return split_by_rule(question)


LLM_SYSTEM_PROMPT = (
    "You split a user's question into standalone sub-questions. If the question asks for one thing, "
    "return it unchanged as the only sub-question. If it asks for two or more different things, return "
    "at most 3 sub-questions, each of which can be understood on its own (repeat any names it needs). "
    'Reply with JSON only, in the form {"sub_questions": ["...", "..."]}.'
)


def parse_sub_questions(reply: str, original: str) -> list[str]:
    """The sub-questions in a model reply, or [original] if the reply is not exactly what was asked for.

    Valid: a JSON object whose "sub_questions" is a list of 1 to 3 non-empty strings of at most 300
    characters. One sub-question means "not split", so the original question is used unchanged.
    """
    try:
        data = json.loads(reply)
        items = data["sub_questions"]
    except (ValueError, TypeError, KeyError):
        return [original]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_SUB_QUESTIONS:
        return [original]
    if not all(isinstance(s, str) and s.strip() and len(s) <= MAX_SUB_QUESTION_CHARS for s in items):
        return [original]
    if len(items) == 1:
        return [original]
    return [" ".join(s.split()) for s in items]


class LLMDecomposer:
    """Asks a language model (anything with generate(system, user)) to split the question.

    Give it a model created with temperature 0 and JSON output. Every failure, including the model
    being unreachable, falls back to the original question, so retrieval never fails because of it.
    """

    name = "llm"

    def __init__(self, llm):
        self.llm = llm

    def decompose(self, question: str) -> list[str]:
        try:
            reply = self.llm.generate(LLM_SYSTEM_PROMPT, question)
        except Exception:  # noqa: BLE001 - any failure means "do not split"
            return [question]
        return parse_sub_questions(reply, question)


def merge_round_robin(lists: list[list[RetrievedChunk]], limit: int) -> list[RetrievedChunk]:
    """Take the best chunk of each list in turn (first list first), skipping chunks already taken,
    until `limit` chunks are collected or every list is used up."""
    merged: list[RetrievedChunk] = []
    seen: set[int] = set()
    positions = [0] * len(lists)
    while len(merged) < limit and any(pos < len(lst) for pos, lst in zip(positions, lists)):
        for i, lst in enumerate(lists):
            while positions[i] < len(lst) and lst[positions[i]].chunk_id in seen:
                positions[i] += 1
            if positions[i] < len(lst):
                chunk = lst[positions[i]]
                positions[i] += 1
                seen.add(chunk.chunk_id)
                merged.append(chunk)
                if len(merged) == limit:
                    break
    return merged


@dataclass
class Decomposed:
    chunks: list[RetrievedChunk]
    sub_questions: list[str]
    decomposition_seconds: float


def search_decomposed(
    conn,
    embedder,
    question: str,
    k: int,
    reranker,
    decomposer,
    filters: SearchFilters | None = None,
    rerank_candidates: int = 30,
) -> Decomposed:
    """Configuration C applied to each sub-question, then merged round-robin.

    For a question that is not split this is exactly vector search with reranking (C). Reranker scores
    of different sub-questions are never compared: only the order within each list is used.
    """
    from time import perf_counter

    from .hybrid import search

    started = perf_counter()
    sub_questions = decomposer.decompose(question)
    seconds = perf_counter() - started
    lists = [
        search(
            conn, embedder, sub, k=k, mode="vector+rerank", filters=filters,
            reranker=reranker, rerank_candidates=rerank_candidates,
        )
        for sub in sub_questions
    ]
    return Decomposed(merge_round_robin(lists, k), sub_questions, seconds)
