import asyncio
import json
import threading
import time

import pytest

from src.api.ask import ask_event_stream
from src.api.deps import get_llm, get_pool
from src.api.schemas import AskRequest
from src.llm.base import LLMUnavailableError, ModelNotFoundError
from src.synthesis.answer import NOT_FOUND

LONG_TEXT = "A token bucket holds tokens and refills at a steady rate, allowing short bursts of requests. " * 3
CHUNKS = [
    (LONG_TEXT.strip(), None, "Token bucket"),
    ("Fixed window counters reset at the end of every window.", None, "Fixed window"),
    ("Flashcards support spaced repetition and active recall.", 3, "Cards"),
]
QUESTION = CHUNKS[0][0]  # worded exactly like the first chunk, so it is the nearest match


def parse_sse(text):
    """The events of a Server-Sent Events body as (name, parsed data) pairs."""
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: "), f"bad event block: {block!r}"
        data_lines = [line[len("data: "):] for line in lines[1:] if line.startswith("data: ")]
        events.append((lines[0][len("event: "):], json.loads("\n".join(data_lines))))
    return events


def names(events):
    return [name for name, _ in events]


@pytest.fixture
def stored(conn, add_document):
    add_document(conn, "Rate Limiting Notes", "notes/rate.md", CHUNKS)


def ask(client, **body):
    body.setdefault("question", QUESTION)
    return client.post("/api/ask", json=body)


# ------------------------------------------------------------- normal answer


def test_events_arrive_in_order_sources_then_tokens_then_done(client, stored, fake_llm):
    fake_llm.reply = "A token bucket refills at a steady rate [1]."
    response = ask(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    sequence = names(events)
    assert sequence[0] == "sources" and sequence[-1] == "done"
    assert sequence.count("sources") == 1 and sequence.count("done") == 1
    assert set(sequence[1:-1]) == {"token"} and len(sequence) > 3
    assert "error" not in sequence


def test_sources_are_numbered_and_carry_the_full_chunk_text(client, stored):
    sources = parse_sse(ask(client).text)[0][1]

    assert [s["number"] for s in sources] == list(range(1, len(sources) + 1))
    assert len(sources) == 3
    first = sources[0]
    assert first["text"] == CHUNKS[0][0] and len(first["text"]) > 150  # the whole text, not a preview
    assert (first["title"], first["path"], first["section_heading"]) == ("Rate Limiting Notes", "notes/rate.md", "Token bucket")
    assert first["page_number"] is None and first["vector_rank"] == 1
    assert {"keyword_rank", "chunk_id"} <= set(first)
    by_heading = {s["section_heading"]: s for s in sources}
    assert by_heading["Cards"]["page_number"] == 3


def test_tokens_are_json_text_pieces_that_add_up_to_the_answer(client, stored, fake_llm):
    fake_llm.reply = 'Line one\nline "two" with unicode caf\u00e9 [1].'
    events = parse_sse(ask(client).text)

    tokens = [data for name, data in events if name == "token"]
    assert all(set(t) == {"text"} and isinstance(t["text"], str) for t in tokens)
    joined = "".join(t["text"] for t in tokens)
    assert joined == fake_llm.reply == events[-1][1]["text"]  # newlines and quotes survive intact


def test_done_event_carries_validation_and_timings(client, stored, fake_llm):
    fake_llm.reply = "A token bucket refills steadily [1][2]."
    done = parse_sse(ask(client).text)[-1][1]

    assert done["validation"] == {
        "cited_valid": [1, 2],
        "cited_invalid": [],
        "has_no_citations": False,
        "is_not_found": False,
    }
    assert set(done["timings"]) == {"retrieval", "first_token", "total", "queue_wait"}
    assert all(done["timings"][key] >= 0 for key in done["timings"])
    assert done["timings"]["first_token"] <= done["timings"]["total"]


def test_invalid_and_missing_citations_are_reported(client, stored, fake_llm):
    fake_llm.reply = "A claim [9]."
    validation = parse_sse(ask(client).text)[-1][1]["validation"]
    assert validation["cited_invalid"] == [9] and validation["has_no_citations"] is True

    fake_llm.reply = "A claim with no citation at all."
    validation = parse_sse(ask(client).text)[-1][1]["validation"]
    assert validation["has_no_citations"] is True and validation["cited_valid"] == []


def test_the_not_found_sentence_from_the_model_is_flagged(client, stored, fake_llm):
    fake_llm.reply = NOT_FOUND
    done = parse_sse(ask(client).text)[-1][1]
    assert done["validation"]["is_not_found"] is True and done["text"] == NOT_FOUND


def test_the_prompt_sent_to_the_model_contains_the_numbered_sources_and_the_question(client, stored, fake_llm):
    ask(client)
    system, user = fake_llm.calls[0]
    assert "[1] (Rate Limiting Notes" in user and QUESTION in user
    assert "untrusted" in system.lower()


# ------------------------------------------------------------ retrieval cases


def test_empty_retrieval_gives_not_found_without_calling_the_model(client, fake_llm):
    events = parse_sse(ask(client).text)  # no documents stored

    assert names(events) == ["sources", "done"]
    assert events[0][1] == []
    assert events[1][1]["text"] == NOT_FOUND and events[1][1]["validation"]["is_not_found"] is True
    assert fake_llm.calls == []


def test_filters_that_match_nothing_also_skip_the_model(client, stored, fake_llm):
    response = ask(client, filters={"source_types": ["pdf"]})  # only markdown is stored
    events = parse_sse(response.text)
    assert names(events) == ["sources", "done"] and events[0][1] == []
    assert fake_llm.calls == []


def test_filters_narrow_the_sources(client, stored):
    sources = parse_sse(ask(client, filters={"path_prefix": "notes/"}).text)[0][1]
    assert len(sources) == 3
    assert parse_sse(ask(client, filters={"path_prefix": "elsewhere/"}).text)[0][1] == []
    assert parse_sse(ask(client, filters={"date_from": "2030-01-01"}).text)[0][1] == []


def test_k_limits_the_number_of_sources(client, stored):
    assert len(parse_sse(ask(client, k=1).text)[0][1]) == 1
    assert len(parse_sse(ask(client, k=2).text)[0][1]) == 2


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_every_mode_works(client, stored, mode):
    events = parse_sse(ask(client, mode=mode).text)
    assert events[0][0] == "sources" and events[-1][0] == "done"
    assert events[0][1], mode


# ----------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   \n "},
        {"question": "x" * 1001},
        {"question": "ok", "k": 0},
        {"question": "ok", "k": 11},
        {"question": "ok", "k": "many"},
        {"question": "ok", "mode": "fuzzy"},
        {"question": "ok", "filters": {"date_from": "2026-13-40"}},
        {"question": "ok", "filters": {"date_from": "yesterday"}},
        {"question": "ok", "filters": {"source_types": ["docx"]}},
        {"question": "ok", "filters": {"source_types": []}},
        {"question": "ok", "filters": {"date_from": "2026-05-02", "date_to": "2026-05-01"}},
        {"question": "ok", "filters": {"path_prefix": "x" * 501}},
        {"question": 42},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, stored, fake_llm, payload):
    assert client.post("/api/ask", json=payload).status_code == 422
    assert fake_llm.calls == []


def test_a_body_that_is_not_json_is_rejected(client):
    response = client.post("/api/ask", content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


@pytest.mark.parametrize("extra", [{"k": 1}, {"k": 10}, {"filters": {"date_from": "2026-01-01", "date_to": "2026-01-01"}}])
def test_boundary_values_are_accepted(client, stored, extra):
    assert ask(client, **extra).status_code == 200


def test_a_question_of_exactly_1000_characters_is_accepted(client, stored):
    assert ask(client, question="q" * 1000).status_code == 200


# ----------------------------------------------------------------- failures


def test_ollama_not_running_gives_an_error_event(client, stored, fake_llm):
    fake_llm.error = LLMUnavailableError("Could not reach Ollama at http://127.0.0.1:11434. Start Ollama (open the Ollama app) and try again.")
    response = ask(client)

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert names(events) == ["sources", "error"]
    assert "Start Ollama" in events[-1][1]["message"]
    assert "Traceback" not in response.text and "File \"" not in response.text


def test_missing_model_gives_an_error_event_that_says_how_to_fix_it(client, stored, fake_llm):
    fake_llm.error = ModelNotFoundError("Model 'llama3.2:3b' is not installed in Ollama. Run: ollama pull llama3.2:3b")
    events = parse_sse(ask(client).text)
    assert events[-1][0] == "error" and "ollama pull" in events[-1][1]["message"]


def test_unexpected_errors_become_a_generic_message_without_details(client, stored, fake_llm):
    fake_llm.error = RuntimeError("secret detail at C:\\private\\path")
    response = ask(client)
    events = parse_sse(response.text)
    assert events[-1][0] == "error"
    assert "secret detail" not in response.text and "private" not in response.text


def test_an_error_in_the_middle_of_the_answer_ends_the_stream_with_an_error_event(client, stored, fake_llm):
    class FailsHalfway:
        model = "fake"

        def stream(self, system, user):
            yield "Part one. "
            yield "Part two. "
            raise LLMUnavailableError("Ollama stopped. Start Ollama and try again.")

    client.app.dependency_overrides[get_llm] = lambda: FailsHalfway()
    events = parse_sse(ask(client).text)
    assert names(events) == ["sources", "token", "token", "error"]


def test_a_database_failure_during_search_gives_an_error_event(client, app, stored):
    class BrokenPool:
        def connection(self):
            raise RuntimeError("connection refused on 127.0.0.1:5433")

    app.dependency_overrides[get_pool] = lambda: BrokenPool()
    response = ask(client)

    events = parse_sse(response.text)
    assert names(events) == ["error"]
    assert "database" in events[0][1]["message"].lower()
    assert "5433" not in response.text and "refused" not in response.text


# ------------------------------------------------- disconnects and one at a time


class EndlessLLM:
    """Produces pieces forever, and notes when it is stopped."""

    model = "fake"

    def __init__(self):
        self.produced = 0
        self.closed = False
        self._generators = []

    def _pieces(self):
        try:
            while True:
                self.produced += 1
                time.sleep(0.01)
                yield f"piece{self.produced} "
        finally:
            self.closed = True

    def stream(self, system, user):
        # Keep a reference, so Python's automatic clean-up cannot stop the generator for us:
        # only an explicit close() by the server can set `closed`.
        generator = self._pieces()
        self._generators.append(generator)
        return generator


def run_stream(pool, embedder, llm, gate):
    return ask_event_stream(AskRequest(question=QUESTION), pool, embedder, llm, gate, "test-request")


def test_closing_the_stream_stops_generation_and_frees_the_turn(stored, pool, embedder):
    async def scenario():
        gate, llm = asyncio.Semaphore(1), EndlessLLM()
        stream = run_stream(pool, embedder, llm, gate)
        received = []
        async for event in stream:
            received.append(event)
            if sum(e.startswith("event: token") for e in received) >= 3:
                break
        await stream.aclose()  # what happens when the client goes away
        return gate, llm, received

    gate, llm, received = asyncio.run(scenario())
    assert llm.closed, "the model's generator was not stopped"
    assert llm.produced < 10, "generation continued after the client left"
    assert not gate.locked(), "the turn was not given back"
    assert received[0].startswith("event: sources")


def test_cancelling_the_request_stops_generation_and_frees_the_turn(stored, pool, embedder):
    async def scenario():
        gate, llm = asyncio.Semaphore(1), EndlessLLM()
        seen_tokens = asyncio.Event()

        async def consume():
            async for event in run_stream(pool, embedder, llm, gate):
                if event.startswith("event: token"):
                    seen_tokens.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(seen_tokens.wait(), timeout=20)
        task.cancel()  # what the server does when it notices the disconnect
        with pytest.raises(asyncio.CancelledError):
            await task
        return gate, llm

    gate, llm = asyncio.run(scenario())
    assert llm.closed and not gate.locked()
    produced_at_cancel = llm.produced
    time.sleep(0.2)
    assert llm.produced == produced_at_cancel  # nothing is generating any more


class BlockingLLM:
    """The first call waits until released; records how many calls run at the same time."""

    model = "fake"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def stream(self, system, user):
        with self.lock:
            self.calls += 1
            first = self.calls == 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if first:
                self.entered.set()
                assert self.release.wait(timeout=20)
            yield "An answer [1]."
        finally:
            with self.lock:
                self.active -= 1

    def status(self):
        raise AssertionError("not used")


def test_a_second_request_waits_its_turn_and_other_endpoints_stay_responsive(client, app, stored):
    llm = BlockingLLM()
    app.dependency_overrides[get_llm] = lambda: llm
    bodies = {}

    def request(label):
        bodies[label] = ask(client).text

    first = threading.Thread(target=request, args=("first",))
    first.start()
    assert llm.entered.wait(timeout=20), "the first answer never started"

    second = threading.Thread(target=request, args=("second",))
    second.start()
    time.sleep(0.6)  # the second request has arrived and must be waiting

    assert llm.calls == 1, "the second request ran in parallel instead of waiting"
    # While an answer is being generated, the server still answers other requests.
    started = time.perf_counter()
    assert client.get("/api/documents").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert time.perf_counter() - started < 5

    llm.release.set()
    first.join(timeout=30)
    second.join(timeout=30)

    assert llm.max_active == 1
    first_events, second_events = parse_sse(bodies["first"]), parse_sse(bodies["second"])
    assert "status" not in names(first_events)
    assert second_events[0] == ("status", {"state": "queued"})
    assert names(first_events)[-1] == names(second_events)[-1] == "done"
    assert second_events[-1][1]["timings"]["queue_wait"] > 0.3


# -------------------------------------------------------------------- logging


def test_logs_from_an_ask_call_contain_none_of_the_question_answer_or_document_words(client, conn, add_document, fake_llm, caplog):
    import logging

    add_document(conn, "Safari", "notes/zoo.md", [("The okapi wanders near the quasar lake every morning.", None, "Animals")])
    fake_llm.reply = "The narwhal answer mentions flamingo [1]."
    question = "Where does the zanzibar okapi wander near the quasar?"

    with caplog.at_level(logging.DEBUG):
        response = ask(client, question=question)
    assert parse_sse(response.text)[-1][0] == "done"

    log_text = caplog.text + " ".join(r.getMessage() for r in caplog.records)
    for word in ("zanzibar", "okapi", "quasar", "narwhal", "flamingo", "wander", "Safari"):
        assert word not in log_text, f"{word!r} leaked into the logs"
    assert "ask id=" in log_text and "outcome=answered" in log_text  # but the useful facts are logged
    assert "chunks=[" in log_text


def test_logs_from_an_upload_contain_no_file_content(client, caplog):
    import logging

    secret_note = b"# Note\n\n## Section\n\n" + b"confidential-platypus " * 40
    with caplog.at_level(logging.DEBUG):
        client.post("/api/ingest", files={"file": ("my note (private).md", secret_note, "text/markdown")})

    log_text = caplog.text
    assert "confidential" not in log_text and "platypus" not in log_text
    assert "ingest id=" in log_text and "my note _private_.md" in log_text  # only the sanitized name
    assert "(private)" not in log_text
