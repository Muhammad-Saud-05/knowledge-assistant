"""Question splitting (configuration L): available by name, and an opt-in default (DECOMPOSITION_ENABLED)."""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings, load_settings
from src.api.deps import get_decomposer, get_reranker
from src.api.main import create_app
from src.retrieval.decompose import LLMDecomposer
from src.retrieval.defaults import PLAIN_MODE, RERANK_MODE, SPLIT_MODE, parse_enabled, resolve_mode
from src.retrieval.rerank import Reranker

DOCS = [
    ("The Ollama chat endpoint is called at http://localhost:11434/api/chat", None, "Ollama chat"),
    ("Declare the endpoint function with plain def so FastAPI runs it in a thread pool", None, "Plain def"),
    ("Fixed window counters reset at the end of every window", None, "Fixed window"),
    ("Flashcards support spaced repetition and active recall", None, "Cards"),
]
PARTS = ["ollama chat endpoint url", "declare endpoint function plain def"]
WHOLE = "which url for the chat endpoint and how do I declare the function"


def parse_sse(text):
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.split("\n")
        data = [line[len("data: "):] for line in lines[1:] if line.startswith("data: ")]
        events.append((lines[0][len("event: "):], json.loads("\n".join(data))))
    return events


class Scripted:
    def __init__(self, parts=None):
        self.parts, self.questions = parts, []

    def decompose(self, question):
        self.questions.append(question)
        return self.parts or [question]


def word_overlap_reranker():
    return Reranker(score_fn=lambda pairs: [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in pairs])


@pytest.fixture
def stored(conn, add_document):
    add_document(conn, "Notes", "notes/all.md", DOCS)


def make_client(app, reranker, decomposer, split_by_default=False):
    app.state.settings = dataclasses.replace(app.state.settings, decomposition_enabled=split_by_default)
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_decomposer] = lambda: decomposer
    return TestClient(app, base_url="http://localhost")


def ask(client, **body):
    body.setdefault("question", WHOLE)
    return client.post("/api/ask", json=body)


def chunk_ids(events):
    return [s["chunk_id"] for s in events[0][1]]


# ------------------------------------------------------------ settings and mode


def test_splitting_is_not_the_default_but_can_be_switched_on(monkeypatch):
    monkeypatch.delenv("DECOMPOSITION_ENABLED", raising=False)
    assert Settings().decomposition_enabled is False
    assert load_settings().decomposition_enabled is False
    monkeypatch.setenv("DECOMPOSITION_ENABLED", "true")
    assert load_settings().decomposition_enabled is True
    monkeypatch.setenv("DECOMPOSITION_ENABLED", "maybe")
    with pytest.raises(ValueError, match="DECOMPOSITION_ENABLED"):
        load_settings()
    assert parse_enabled("", "X", default=False) is False and parse_enabled("on", "X", default=False) is True


def test_resolve_mode_default_is_c_unless_splitting_is_the_default():
    assert resolve_mode(None, True, True) == RERANK_MODE == "vector+rerank"  # a decomposer alone changes nothing
    assert resolve_mode("default", True, True, split_by_default=False) == RERANK_MODE
    assert resolve_mode("default", True, True, split_by_default=True) == SPLIT_MODE == "rerank-split-llm"
    assert resolve_mode("default", True, False, split_by_default=True) == RERANK_MODE
    assert resolve_mode("default", False, True, split_by_default=True) == PLAIN_MODE  # no reranker: no splitting
    assert resolve_mode("default", False, False) == PLAIN_MODE


def test_the_split_mode_can_always_be_asked_for_by_name():
    assert resolve_mode(SPLIT_MODE, True, True, split_by_default=False) == SPLIT_MODE
    with pytest.raises(ValueError, match="not available"):
        resolve_mode(SPLIT_MODE, True, False)
    with pytest.raises(ValueError, match="reranking is switched off"):
        resolve_mode(SPLIT_MODE, False, True)


# ------------------------------------------------------------------ the API


def test_default_search_does_not_split_even_when_a_splitter_exists(app, stored):
    spy = Scripted(PARTS)
    with make_client(app, word_overlap_reranker(), spy) as client:
        default = parse_sse(ask(client).text)
        c_mode = parse_sse(ask(client, mode="vector+rerank").text)
    assert spy.questions == []  # the model was never asked to split
    assert chunk_ids(default) == chunk_ids(c_mode)


def test_the_split_mode_by_name_splits_and_the_sources_are_the_merged_list(app, stored, fake_llm):
    fake_llm.reply = "Call /api/chat [1]. Use plain def [2]."
    decomposer = Scripted(PARTS)
    with make_client(app, word_overlap_reranker(), decomposer) as client:
        events = parse_sse(ask(client, mode="rerank-split-llm").text)
    assert decomposer.questions == [WHOLE]
    sources = events[0][1]
    assert events[0][0] == "sources"
    assert [s["section_heading"] for s in sources][:2] == ["Ollama chat", "Plain def"]  # best of each list first
    assert [s["number"] for s in sources] == list(range(1, len(sources) + 1))
    assert len({s["chunk_id"] for s in sources}) == len(sources) == 4  # de-duplicated; only 4 chunks are stored
    assert all(s["rerank_rank"] is not None and s["rerank_score"] is not None for s in sources)


def test_citations_validate_against_the_merged_list(app, stored, fake_llm):
    fake_llm.reply = "Call /api/chat [1]. Use plain def [2]. Another claim [9]."
    with make_client(app, word_overlap_reranker(), Scripted(PARTS)) as client:
        events = parse_sse(ask(client, mode="rerank-split-llm").text)
    prompt = fake_llm.calls[0][1]
    assert prompt.index("[1]") < prompt.index("api/chat") < prompt.index("[2]") < prompt.index("plain def")
    assert f"Question: {WHOLE}" in prompt  # the model answers the original question, not a sub-question
    validation = events[-1][1]["validation"]
    assert validation["cited_valid"] == [1, 2] and validation["cited_invalid"] == [9]


def test_when_splitting_is_the_default_plain_questions_are_split_too(app, stored):
    decomposer = Scripted(PARTS)
    with make_client(app, word_overlap_reranker(), decomposer, split_by_default=True) as client:
        default = parse_sse(ask(client).text)
        by_name = parse_sse(ask(client, mode="rerank-split-llm").text)
    assert decomposer.questions == [WHOLE, WHOLE]
    assert chunk_ids(default) == chunk_ids(by_name)


def test_a_question_that_is_not_split_gives_the_same_sources_as_configuration_c(app, stored):
    with make_client(app, word_overlap_reranker(), Scripted(None)) as client:
        split = parse_sse(ask(client, mode="rerank-split-llm").text)
        c_mode = parse_sse(ask(client, mode="vector+rerank").text)
    assert chunk_ids(split) == chunk_ids(c_mode)


def test_the_split_mode_is_refused_when_no_splitter_is_available(app, stored):
    with make_client(app, word_overlap_reranker(), None) as client:
        refused = ask(client, mode="rerank-split-llm")
    assert refused.status_code == 400 and "not available" in refused.json()["detail"]


def test_without_a_reranker_the_default_is_plain_hybrid_and_the_split_mode_is_refused(app, stored):
    decomposer = Scripted(PARTS)
    with make_client(app, None, decomposer, split_by_default=True) as client:
        sources = parse_sse(ask(client).text)[0][1]
        refused = ask(client, mode="rerank-split-llm")
    assert decomposer.questions == [] and all(s["rerank_rank"] is None for s in sources)
    assert refused.status_code == 400


def test_a_model_failure_in_the_split_does_not_break_the_answer(app, stored):
    class Down:
        def generate(self, system, user):
            raise ConnectionError("ollama is down")

    with make_client(app, word_overlap_reranker(), LLMDecomposer(Down())) as client:
        events = parse_sse(ask(client, mode="rerank-split-llm").text)
    assert [n for n, _ in events][0] == "sources" and events[-1][0] == "done"
    assert events[0][1]  # searched with the whole question


# ------------------------------------------------------------------ startup


def _lifespan_app(monkeypatch, settings):
    import src.api.main as main

    class FakeEmbedder:
        model = object()

    class FakePoolObject:
        def open(self, wait=False):
            pass

        def close(self):
            pass

    monkeypatch.setattr(main, "create_pool", lambda: FakePoolObject())
    monkeypatch.setattr(main, "Embedder", FakeEmbedder)
    monkeypatch.setattr(main, "load_reranker", lambda: Reranker(score_fn=lambda p: [0.0] * len(p)))
    monkeypatch.setattr(main, "make_decomposer", lambda: Scripted())
    monkeypatch.setattr(main, "OllamaLLM", lambda: type("L", (), {"warm_up": lambda self: None})())
    return create_app(settings=settings)


def test_startup_always_makes_the_splitter_available_when_there_is_a_reranker(monkeypatch):
    for settings, expected in (
        (Settings(), True),  # splitting is not the default, but the mode must still work
        (Settings(decomposition_enabled=True), True),
        (Settings(reranker_enabled=False), False),  # it needs the reranker
    ):
        app = _lifespan_app(monkeypatch, settings)
        with TestClient(app, base_url="http://localhost"):
            assert (app.state.decomposer is not None) is expected
