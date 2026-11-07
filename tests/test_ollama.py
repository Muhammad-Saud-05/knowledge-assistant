import json

import httpx
import pytest

from src.llm.base import LLMError, LLMUnavailableError, ModelNotFoundError
from src.llm.ollama import OllamaLLM


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Ignore the real .env and any OLLAMA_* variables, so defaults are tested."""
    monkeypatch.setattr("src.llm.ollama.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)


def make_llm(handler, **kwargs):
    return OllamaLLM(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def ndjson(*objects):
    return ("\n".join(json.dumps(o) for o in objects) + "\n").encode()


def reply(text, done=True):
    return {"model": "m", "message": {"role": "assistant", "content": text}, "done": done}


# ------------------------------------------------------------------ settings


def test_defaults(monkeypatch):
    llm = OllamaLLM(client=httpx.Client())
    assert llm.host == "http://127.0.0.1:11434"
    assert llm.model == "llama3.2:3b"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example.test:9999/")
    monkeypatch.setenv("OLLAMA_MODEL", "other:1b")
    llm = OllamaLLM(client=httpx.Client())
    assert (llm.host, llm.model) == ("http://example.test:9999", "other:1b")


def test_arguments_beat_environment_variables(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    assert OllamaLLM(model="from-arg", client=httpx.Client()).model == "from-arg"


# ------------------------------------------------------------------ generate


def test_generate_parses_the_reply_and_sends_the_right_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply("Hello [1]"))

    text = make_llm(handler).generate("be brief", "hi there")

    assert text == "Hello [1]"
    body = seen["body"]
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert body["model"] == "llama3.2:3b" and body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi there"},
    ]
    assert body["options"]["num_ctx"] == 4096  # set explicitly, not left to the default
    assert body["options"]["temperature"] == 0.1


def test_generate_unreachable_server_gives_friendly_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError, match="Start Ollama"):
        make_llm(handler).generate("s", "u")


def test_generate_missing_model_gives_pull_instruction():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"})

    with pytest.raises(ModelNotFoundError, match="ollama pull llama3.2:3b"):
        make_llm(handler).generate("s", "u")


def test_generate_other_http_errors_are_reported():
    def handler(request):
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(LLMError, match="500.*out of memory"):
        make_llm(handler).generate("s", "u")


def test_generate_timeout_gives_friendly_error():
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(LLMError, match="too long"):
        make_llm(handler).generate("s", "u")


def test_generate_unexpected_reply_format_is_an_error():
    with pytest.raises(LLMError, match="unexpected format"):
        make_llm(lambda r: httpx.Response(200, json={"nope": 1})).generate("s", "u")


# -------------------------------------------------------------------- stream


def test_stream_yields_pieces_in_order_and_requests_streaming():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        content = ndjson(reply("Hel", False), reply("lo ", False), reply("[1]", False), reply("", True))
        return httpx.Response(200, content=content)

    pieces = list(make_llm(handler).stream("s", "u"))

    assert pieces == ["Hel", "lo ", "[1]"]
    assert seen["body"]["stream"] is True
    assert seen["body"]["options"]["num_ctx"] == 4096


def test_stream_stops_at_done_and_skips_blank_lines():
    content = b"\n" + ndjson(reply("a", False), reply("", True)) + b"\n" + ndjson(reply("ignored"))
    pieces = list(make_llm(lambda r: httpx.Response(200, content=content)).stream("s", "u"))
    assert pieces == ["a"]


def test_stream_unreachable_server_gives_friendly_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError, match="Start Ollama"):
        list(make_llm(handler).stream("s", "u"))


def test_stream_missing_model_gives_pull_instruction():
    handler = lambda r: httpx.Response(404, json={"error": "model 'x' not found"})  # noqa: E731
    with pytest.raises(ModelNotFoundError, match="ollama pull"):
        list(make_llm(handler, model="x").stream("s", "u"))


def test_stream_error_line_in_the_middle_is_reported():
    content = ndjson(reply("partial", False), {"error": "model crashed"})
    stream = make_llm(lambda r: httpx.Response(200, content=content)).stream("s", "u")
    assert next(stream) == "partial"
    with pytest.raises(LLMError, match="model crashed"):
        next(stream)


def test_stream_garbled_line_is_an_error():
    with pytest.raises(LLMError, match="unexpected format"):
        list(make_llm(lambda r: httpx.Response(200, content=b"not json\n")).stream("s", "u"))


# ----------------------------------------------------------------- keep_alive


def test_keep_alive_defaults_to_thirty_minutes_and_is_sent_with_every_request():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=ndjson(reply("ok", True)))

    llm = make_llm(handler)
    assert llm.keep_alive == "30m"
    llm.generate("s", "u")
    list(llm.stream("s", "u"))
    assert [b["keep_alive"] for b in bodies] == ["30m", "30m"]


def test_keep_alive_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "2h")
    assert OllamaLLM(client=httpx.Client()).keep_alive == "2h"


def test_keep_alive_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "2h")
    assert OllamaLLM(keep_alive="10m", client=httpx.Client()).keep_alive == "10m"


def test_warm_up_loads_the_model_without_generating():
    seen = {}

    def handler(request):
        seen["url"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "m", "done": True, "done_reason": "load"})

    make_llm(handler).warm_up()

    assert seen["url"] == "/api/chat"
    assert seen["body"]["messages"] == [] and seen["body"]["keep_alive"] == "30m"
    assert seen["body"]["model"] == "llama3.2:3b"


def test_warm_up_reports_the_same_friendly_errors():
    def unreachable(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError, match="Start Ollama"):
        make_llm(unreachable).warm_up()

    missing = lambda r: httpx.Response(404, json={"error": "model 'x' not found"})  # noqa: E731
    with pytest.raises(ModelNotFoundError, match="ollama pull"):
        make_llm(missing, model="x").warm_up()


# --------------------------------------------------------------------- status


def tags(*names):
    return httpx.Response(200, json={"models": [{"name": n} for n in names]})


def test_status_when_the_model_is_installed():
    status = make_llm(lambda r: tags("other:1b", "llama3.2:3b")).status()
    assert (status.reachable, status.model_installed, status.detail) == (True, True, None)


def test_status_when_the_model_is_missing_says_how_to_install_it():
    status = make_llm(lambda r: tags("other:1b")).status()
    assert status.reachable and status.model_installed is False
    assert "ollama pull llama3.2:3b" in status.detail


def test_status_treats_a_bare_model_name_as_latest():
    assert make_llm(lambda r: tags("llama3.2:latest"), model="llama3.2").status().model_installed


def test_status_when_ollama_is_not_running_does_not_raise():
    def handler(request):
        raise httpx.ConnectError("refused")

    status = make_llm(handler).status()
    assert (status.reachable, status.model_installed) == (False, None)
    assert "isn't running" in status.detail


def test_status_when_ollama_answers_with_an_error_does_not_raise():
    status = make_llm(lambda r: httpx.Response(500)).status()
    assert not status.reachable and status.detail


def test_status_with_a_garbled_reply_does_not_raise():
    assert not make_llm(lambda r: httpx.Response(200, content=b"<html>")).status().reachable
