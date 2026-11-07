"""POST /api/ask: answer a question, streamed as Server-Sent Events.

The reply is a stream of "events", each one a few lines of text:

    event: sources          the retrieved chunks, numbered 1..n (always first)
    data: [{"number": 1, "title": ..., "text": ..., ...}, ...]

    event: token            one piece of the answer, as it is generated
    data: {"text": "A bucket"}

    event: done             the final result: full text, citation checks, timings
    event: error            a short, friendly message; the stream then ends
    event: status           {"state": "queued"} while waiting for another answer to finish

Answers are generated one at a time (a CPU can only run one language model generation
well), so a second request waits its turn. Blocking work (database, embedding model, language
model) runs in worker threads so the server stays responsive for /api/health and the rest.
"""

import json
import logging
import time
from collections.abc import AsyncIterator

import anyio.to_thread
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from ..llm.base import LLMError
from ..retrieval.hybrid import search
from ..retrieval.models import RetrievedChunk
from ..synthesis.answer import NOT_FOUND, Answer, build_prompt, finish_answer
from .deps import get_embedder, get_gate, get_llm, get_pool
from .observability import request_id_of
from .schemas import AskRequest

router = APIRouter()
logger = logging.getLogger("kb.api")

SEARCH_FAILED = "I couldn't search your documents. Is the database running?"
GENERIC_FAILURE = "Something went wrong while answering. Please try again."
_DONE = object()


def sse(event: str, data) -> str:
    """One Server-Sent Event. JSON never contains a raw newline, so it cannot break the stream."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def source_payload(number: int, chunk: RetrievedChunk) -> dict:
    return {
        "number": number,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "path": chunk.path,
        "page_number": chunk.page_number,
        "section_heading": chunk.section_heading,
        "text": chunk.text,
        "vector_rank": chunk.vector_rank,
        "keyword_rank": chunk.keyword_rank,
    }


def done_payload(answer: Answer, queue_wait: float) -> dict:
    validation = answer.validation
    return {
        "text": answer.text,
        "validation": {
            "cited_valid": validation.cited_valid,
            "cited_invalid": validation.cited_invalid,
            "has_no_citations": validation.has_no_citations,
            "is_not_found": validation.is_not_found,
        },
        "timings": {**answer.timings, "queue_wait": queue_wait},
    }


def _retrieve(pool, embedder, body: AskRequest) -> list[RetrievedChunk]:
    filters = body.filters.to_search_filters() if body.filters else None
    with pool.connection() as conn:
        return search(conn, embedder, body.question, k=body.k, mode=body.mode, filters=filters)


def _next_piece(iterator):
    """Advance the language model's stream by one piece (blocking; runs in a thread)."""
    try:
        return next(iterator)
    except StopIteration:
        return _DONE


async def ask_event_stream(
    body: AskRequest, pool, embedder, llm, gate, request_id: str
) -> AsyncIterator[str]:
    queue_wait = 0.0
    if gate.locked():  # another answer is being generated: tell the client it is waiting
        yield sse("status", {"state": "queued"})
        waiting_since = time.perf_counter()
        await gate.acquire()
        queue_wait = time.perf_counter() - waiting_since
    else:
        await gate.acquire()

    pieces_iterator = None
    try:
        started = time.perf_counter()

        try:
            chunks = await run_in_threadpool(_retrieve, pool, embedder, body)
        except Exception as error:  # noqa: BLE001
            logger.error("ask id=%s search failed type=%s", request_id, type(error).__name__)
            yield sse("error", {"message": SEARCH_FAILED})
            return
        retrieval_seconds = time.perf_counter() - started
        yield sse("sources", [source_payload(n, c) for n, c in enumerate(chunks, start=1)])

        if not chunks:  # nothing to base an answer on: do not call the language model
            answer = finish_answer(body.question, chunks, NOT_FOUND, started, retrieval_seconds, None)
            _log_outcome(request_id, body, chunks, answer, queue_wait, "no-sources")
            yield sse("done", done_payload(answer, queue_wait))
            return

        system, user = build_prompt(body.question, chunks)
        pieces: list[str] = []
        first_token: float | None = None
        try:
            pieces_iterator = llm.stream(system, user)
            while True:
                piece = await anyio.to_thread.run_sync(_next_piece, pieces_iterator)
                if piece is _DONE:
                    break
                if first_token is None:
                    first_token = time.perf_counter() - started
                pieces.append(piece)
                yield sse("token", {"text": piece})
        except LLMError as error:  # Ollama not running, model missing, ... (messages are friendly)
            logger.error("ask id=%s model error type=%s", request_id, type(error).__name__)
            yield sse("error", {"message": str(error)})
            return
        except Exception as error:  # noqa: BLE001
            logger.error("ask id=%s unexpected error type=%s", request_id, type(error).__name__)
            yield sse("error", {"message": GENERIC_FAILURE})
            return

        answer = finish_answer(body.question, chunks, "".join(pieces), started, retrieval_seconds, first_token)
        _log_outcome(request_id, body, chunks, answer, queue_wait, "answered")
        yield sse("done", done_payload(answer, queue_wait))
    finally:
        # Runs on success, on error, and when the client disconnects (the stream is
        # cancelled or closed): stop the language model's generator, free the turn.
        close = getattr(pieces_iterator, "close", None)
        if close is not None:
            close()
        gate.release()


def _log_outcome(request_id: str, body: AskRequest, chunks, answer: Answer, queue_wait: float, outcome: str) -> None:
    """Ids and timings only: never the question or answer text."""
    timings = answer.timings
    first = timings.get("first_token")
    logger.info(
        "ask id=%s outcome=%s mode=%s k=%d chunks=%s retrieval=%.2fs first_token=%s total=%.2fs queue_wait=%.2fs",
        request_id, outcome, body.mode, body.k, [c.chunk_id for c in chunks],
        timings["retrieval"], f"{first:.2f}s" if first is not None else "n/a", timings["total"], queue_wait,
    )


@router.post("/ask")
async def ask(
    body: AskRequest,
    request: Request,
    pool=Depends(get_pool),
    embedder=Depends(get_embedder),
    llm=Depends(get_llm),
    gate=Depends(get_gate),
) -> StreamingResponse:
    """Answer a question from the stored documents, streamed as Server-Sent Events."""
    stream = ask_event_stream(body, pool, embedder, llm, gate, request_id_of(request))
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
