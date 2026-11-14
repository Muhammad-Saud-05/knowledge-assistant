"""The reranked search (configuration C) as the application's default, and how to switch it off."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings, load_settings
from src.api.deps import get_embedder, get_llm, get_pool, get_reranker
from src.api.main import create_app
from src.retrieval.defaults import (
    PLAIN_MODE,
    RERANK_MODE,
    RerankerLoadError,
    load_reranker,
    parse_enabled,
    resolve_mode,
)
from src.retrieval.rerank import Reranker
from tests.conftest import FakePool

CHUNKS = [
    ("Fixed window counters reset at the end of every window.", None, "Fixed window"),
    ("A token bucket holds tokens and refills at a steady rate, allowing short bursts.", None, "Token bucket"),
    ("Flashcards support spaced repetition and active recall.", 3, "Cards"),
]


def parse_sse(text):
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.split("\n")
        data = [line[len("data: "):] for line in lines[1:] if line.startswith("data: ")]
        events.append((lines[0][len("event: "):], json.loads("\n".join(data))))
    return events


# ------------------------------------------------------------------ settings


def test_reranker_enabled_defaults_to_true_and_reads_clear_values():
    for value in (None, "", "  ", "true", "TRUE", "1", "yes", "on"):
        assert parse_enabled(value) is True
    for value in ("false", "False", "0", "no", "off"):
        assert parse_enabled(value) is False
    with pytest.raises(ValueError, match="RERANKER_ENABLED"):
        parse_enabled("maybe")


def test_settings_read_the_switch_from_the_environment(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    assert load_settings().reranker_enabled is False
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    assert load_settings().reranker_enabled is True
    assert Settings().reranker_enabled is True


def test_resolve_mode():
    assert resolve_mode(None, True) == RERANK_MODE == "vector+rerank"
    assert resolve_mode("default", True) == RERANK_MODE
    assert resolve_mode("default", False) == PLAIN_MODE == "hybrid"
    for mode in ("vector", "keyword", "hybrid"):
        assert resolve_mode(mode, True) == mode and resolve_mode(mode, False) == mode
    assert resolve_mode("vector+rerank", True) == "vector+rerank"
    with pytest.raises(ValueError, match="switched off"):
        resolve_mode("vector+rerank", False)
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_mode("fuzzy", True)


def test_a_reranker_that_cannot_load_gives_a_clear_error(monkeypatch):
    def broken(self):
        raise OSError("no network")

    monkeypatch.setattr(Reranker, "load", broken)
    with pytest.raises(RerankerLoadError) as info:
        load_reranker()
    message = str(info.value)
    assert "ms-marco-MiniLM-L-6-v2" in message and "RERANKER_ENABLED=false" in message
    assert "no network" not in message  # the underlying text is only chained, never shown


# ------------------------------------------------------------------ the API


def prefers(text):
    """A fake cross-encoder that scores a passage 9 when it contains `text`, otherwise 0."""
    return Reranker(score_fn=lambda pairs: [9.0 if text in p else 0.0 for _, p in pairs])


@pytest.fixture
def stored(conn, add_document):
    add_document(conn, "Rate Limiting Notes", "notes/rate.md", CHUNKS)


def make_client(app, reranker):
    app.dependency_overrides[get_reranker] = lambda: reranker
    return TestClient(app, base_url="http://localhost")


def ask(client, **body):
    body.setdefault("question", CHUNKS[0][0])
    return client.post("/api/ask", json=body)


def test_default_search_reranks_and_the_sources_list_the_reranked_order(app, stored, fake_llm):
    fake_llm.reply = "A token bucket refills steadily [1]."
    with make_client(app, prefers("token bucket")) as client:
        events = parse_sse(ask(client).text)
    sources = events[0][1]
    assert events[0][0] == "sources"
    # vector search alone would put the fixed-window chunk first (it is the question's exact text)
    assert [s["section_heading"] for s in sources][0] == "Token bucket"
    assert [s["number"] for s in sources] == [1, 2, 3]
    assert [s["rerank_rank"] for s in sources] == [1, 2, 3]
    assert sources[0]["rerank_score"] == 9.0 and sources[1]["rerank_score"] == 0.0
    assert all(s["vector_rank"] is not None for s in sources)  # first-stage rank is kept


def test_citations_validate_against_the_reranked_order(app, stored, fake_llm):
    fake_llm.reply = "A token bucket refills steadily [1]. Cards help [3]."
    with make_client(app, prefers("token bucket")) as client:
        events = parse_sse(ask(client).text)
    prompt = fake_llm.calls[0][1]
    assert prompt.index("[1]") < prompt.index("token bucket") < prompt.index("[2]")  # source 1 is the reranked first
    validation = events[-1][1]["validation"]
    assert validation["cited_valid"] == [1, 3] and validation["cited_invalid"] == []
    fake_llm.reply = "Nonexistent [4]."
    with make_client(app, prefers("token bucket")) as client:
        assert parse_sse(ask(client).text)[-1][1]["validation"]["cited_invalid"] == [4]


def test_k_limits_the_reranked_sources(app, stored):
    with make_client(app, prefers("token bucket")) as client:
        sources = parse_sse(ask(client, k=2).text)[0][1]
    assert len(sources) == 2 and sources[0]["section_heading"] == "Token bucket"


def test_without_a_reranker_the_default_is_plain_hybrid(app, stored):
    with make_client(app, None) as client:
        sources = parse_sse(ask(client).text)[0][1]
    assert sources[0]["section_heading"] == "Fixed window"  # the question's own text is the nearest chunk
    assert all(s["rerank_rank"] is None and s["rerank_score"] is None for s in sources)


def test_explicit_modes_never_rerank(app, stored):
    with make_client(app, prefers("token bucket")) as client:
        for mode in ("vector", "keyword", "hybrid"):
            sources = parse_sse(ask(client, mode=mode).text)[0][1]
            assert sources and all(s["rerank_rank"] is None for s in sources), mode
        assert parse_sse(ask(client, mode="vector+rerank").text)[0][1][0]["rerank_rank"] == 1


def test_the_reranked_mode_is_refused_when_reranking_is_off(app, stored):
    with make_client(app, None) as client:
        response = ask(client, mode="vector+rerank")
    assert response.status_code == 400 and "switched off" in response.json()["detail"]


# ------------------------------------------------------------------ startup


def _lifespan_app(monkeypatch, enabled, loader):
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
    monkeypatch.setattr(main, "load_reranker", loader)
    monkeypatch.setattr(main, "OllamaLLM", lambda: type("L", (), {"warm_up": lambda self: None})())
    return create_app(settings=Settings(reranker_enabled=enabled))


def test_startup_loads_the_reranker_once(monkeypatch):
    calls = []
    marker = Reranker(score_fn=lambda pairs: [0.0] * len(pairs))
    app = _lifespan_app(monkeypatch, True, lambda: calls.append(1) or marker)
    with TestClient(app, base_url="http://localhost"):
        assert app.state.reranker is marker
    assert calls == [1]


def test_startup_fails_clearly_when_the_reranker_cannot_load(monkeypatch):
    def broken():
        raise RerankerLoadError("Could not load the reranker model x. Set RERANKER_ENABLED=false")

    app = _lifespan_app(monkeypatch, True, broken)
    with pytest.raises(RerankerLoadError, match="RERANKER_ENABLED=false"):
        with TestClient(app, base_url="http://localhost"):
            pass


def test_startup_skips_the_reranker_when_switched_off(monkeypatch):
    def must_not_run():
        raise AssertionError("the reranker must not be loaded when RERANKER_ENABLED=false")

    app = _lifespan_app(monkeypatch, False, must_not_run)
    with TestClient(app, base_url="http://localhost"):
        assert app.state.reranker is None


# ------------------------------------------------------------------ the CLIs


def test_ask_report_shows_the_rerank_rank_and_score():
    from src.ask import _found_by
    from src.retrieval.models import RetrievedChunk

    chunk = RetrievedChunk(1, 1, "T", "p", None, None, "x", 3.0, vector_rank=4, rerank_score=3.0, rerank_rank=1)
    assert _found_by(chunk, "vector+rerank") == "vector #4, rerank #1 (score 3.00)"
    assert _found_by(RetrievedChunk(1, 1, "T", "p", None, None, "x", 0.5, vector_rank=2), "hybrid") == "vector #2, keyword -"


def test_search_cli_adds_the_reranked_view_only_when_a_reranker_is_given(conn, embedder, stored):
    from src.search import search_all_modes

    plain = search_all_modes(conn, embedder, CHUNKS[0][0], 3, None)
    assert list(plain) == ["vector", "keyword", "hybrid"]
    views = search_all_modes(conn, embedder, CHUNKS[0][0], 3, None, prefers("token bucket"))
    assert list(views) == ["vector", "keyword", "hybrid", "vector+rerank"]
    assert views["vector+rerank"][0].section_heading == "Token bucket" and views["vector+rerank"][0].rerank_rank == 1
    assert views["hybrid"] == plain["hybrid"]  # the existing views are unchanged
