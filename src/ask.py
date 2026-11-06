"""Ask a question about your documents and get a cited answer.

Usage: python -m src.ask "how does a token bucket work" [--k 5] [--mode hybrid] [--model NAME]
           [--show-prompt] [--type pdf] [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--path-prefix PREFIX]
Ollama must be running, and documents must already be ingested.
"""

import argparse
import sys
import time
from collections.abc import Iterator

import psycopg

from .cli_filters import add_filter_arguments, filters_from_args
from .db.connection import get_connection
from .ingestion.embedder import Embedder
from .llm.base import LLMError
from .llm.ollama import NUM_CTX, OllamaLLM
from .retrieval.hybrid import MODES
from .synthesis.answer import Answer, answer_question_stream

SOURCE_PREVIEW_CHARS = 120
# Rough size of a token in characters, for the "prompt may not fit" warning.
CHARS_PER_TOKEN = 3.2
# Room to leave in the context window for the model's own answer.
ANSWER_TOKEN_ROOM = 400


class _PromptWatcher:
    """Wraps the model: shows the exact prompt sent, and warns if it may not fit."""

    def __init__(self, llm, show_prompt: bool):
        self._llm = llm
        self._show_prompt = show_prompt

    def generate(self, system: str, user: str) -> str:
        self._inspect(system, user)
        return self._llm.generate(system, user)

    def stream(self, system: str, user: str) -> Iterator[str]:
        self._inspect(system, user)
        yield from self._llm.stream(system, user)

    def _inspect(self, system: str, user: str) -> None:
        if self._show_prompt:
            print("=" * 30 + " SYSTEM PROMPT " + "=" * 30)
            print(system)
            print("=" * 31 + " USER MESSAGE " + "=" * 31)
            print(user)
            print("=" * 76 + "\n")
        estimated = int((len(system) + len(user)) / CHARS_PER_TOKEN)
        if estimated > NUM_CTX - ANSWER_TOKEN_ROOM:
            print(
                f"Warning: the prompt is about {estimated} tokens, which may not fit the model's "
                f"{NUM_CTX}-token window, so the start of the sources could be cut off. "
                "Try a smaller --k.\n",
                file=sys.stderr,
            )


def _describe(chunk) -> str:
    text = chunk.title or "Untitled"
    if chunk.page_number is not None:
        text += f", p. {chunk.page_number}"
    if chunk.section_heading:
        text += f" ({chunk.section_heading})"
    return text


def _found_by(chunk, mode: str) -> str:
    """Where a chunk ranked in the search methods that were used."""

    def rank(value: int | None) -> str:
        return f"#{value}" if value is not None else "-"

    parts = []
    if mode in ("vector", "hybrid"):
        parts.append(f"vector {rank(chunk.vector_rank)}")
    if mode in ("keyword", "hybrid"):
        parts.append(f"keyword {rank(chunk.keyword_rank)}")
    return ", ".join(parts)


def _print_report(answer: Answer, model_load_seconds: float, mode: str) -> None:
    validation = answer.validation
    print("\n")

    if validation.is_not_found:
        print("The model says your documents do not contain the answer.")
    else:
        print("Sources")
        if answer.cited_chunks:
            for number, chunk in answer.cited_chunks:
                preview = " ".join(chunk.text.split())[:SOURCE_PREVIEW_CHARS]
                print(f"[{number}] {_describe(chunk)} - found by {_found_by(chunk, mode)}")
                print(f"    {preview}")
        else:
            print("(none cited)")

    if validation.cited_invalid:
        numbers = ", ".join(str(n) for n in validation.cited_invalid)
        print(
            f"\nWarning: the answer cites source number(s) {numbers}, but only "
            f"{len(answer.chunks)} sources were retrieved. Those citations are not valid."
        )
    if validation.has_no_citations:
        print(
            "\nWarning: the answer has no valid citations, so it cannot be checked against "
            "your documents. Treat it with caution."
        )

    first = answer.timings.get("first_token")
    first_text = f"{first:.1f}s" if first is not None else "n/a"
    print(f"\nTime: first token {first_text}, total {answer.timings['total']:.1f}s")
    print(
        f"      (retrieval {answer.timings['retrieval']:.1f}s; the embedding model took "
        f"{model_load_seconds:.1f}s to load, which happens once per run and is not counted above)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about your ingested documents.")
    parser.add_argument("question", help="your question, in quotes")
    parser.add_argument("--k", type=int, default=5, help="how many sources to retrieve (default 5)")
    parser.add_argument(
        "--mode", choices=MODES, default="hybrid", help="how to search: vector, keyword or hybrid (default)"
    )
    parser.add_argument("--model", help="Ollama model name (default: OLLAMA_MODEL or llama3.2:3b)")
    parser.add_argument(
        "--show-prompt", action="store_true", help="print the exact prompt sent to the model"
    )
    add_filter_arguments(parser)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")
    if not args.question.strip():
        parser.error("the question must not be empty")
    filters = filters_from_args(parser, args)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        conn = get_connection()
    except (psycopg.OperationalError, RuntimeError) as error:
        # Do not echo connection errors: they can mention the host and user.
        reason = str(error) if isinstance(error, RuntimeError) else "is the database running?"
        print(f"Could not connect to the database ({reason})", file=sys.stderr)
        return 2

    llm = _PromptWatcher(OllamaLLM(model=args.model), args.show_prompt)
    embedder = Embedder()
    # Load the embedding model now, so the timings below measure the question,
    # not the one-time model load.
    load_started = time.perf_counter()
    _ = embedder.model
    model_load_seconds = time.perf_counter() - load_started
    try:
        with conn:
            stream = answer_question_stream(
                args.question, conn, embedder, llm, k=args.k, mode=args.mode, filters=filters
            )
            while True:
                try:
                    print(next(stream), end="", flush=True)
                except StopIteration as done:
                    answer = done.value
                    break
    except LLMError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1

    _print_report(answer, model_load_seconds, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
