import pytest
from fastapi.testclient import TestClient

from src.api.config import DEFAULT_ORIGINS, Settings, load_settings, parse_origins
from src.api.deps import get_llm, get_pool
from src.api.main import create_app
from src.llm.base import ModelStatus

ALLOWED = "http://localhost:5173"
ALLOWED_ALT = "http://127.0.0.1:5173"
EVIL = "http://evil.example"


class BrokenPool:
    def connection(self):
        raise RuntimeError("could not connect to 127.0.0.1:5433 as kb_user")


# --------------------------------------------------------------------- health


def test_health_when_everything_is_up(client, conn, add_document):
    add_document(conn, "Doc", "d.md", [("one chunk of text here", None, None), ("another chunk", None, None)])
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == {"ok": True, "documents": 1, "chunks": 2, "detail": None}
    assert body["ollama"] == {"ok": True, "model": "fake-model", "model_installed": True, "detail": None}


def test_health_when_ollama_is_down_is_still_200_with_the_part_marked(client, fake_llm):
    fake_llm.status_result = ModelStatus(False, "fake-model", None, "Ollama isn't running.")
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama"]["ok"] is False and body["ollama"]["detail"] == "Ollama isn't running."
    assert body["database"]["ok"] is True


def test_health_when_the_model_is_missing(client, fake_llm):
    fake_llm.status_result = ModelStatus(True, "fake-model", False, "The model fake-model isn't installed (run: ollama pull fake-model).")
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"]["ok"] is True and body["ollama"]["model_installed"] is False
    assert "ollama pull" in body["ollama"]["detail"]


def test_health_survives_the_status_check_itself_failing(client, fake_llm):
    def explode():
        raise RuntimeError("boom")

    fake_llm.status = explode
    response = client.get("/api/health")
    assert response.status_code == 200 and response.json()["ollama"]["ok"] is False


def test_health_when_the_database_is_down_is_still_200_without_internal_details(client, app):
    app.dependency_overrides[get_pool] = lambda: BrokenPool()
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"]["ok"] is False and body["database"]["documents"] is None
    assert body["ollama"]["ok"] is True
    assert "5433" not in response.text and "kb_user" not in response.text


def test_health_when_everything_is_down(client, app, fake_llm):
    app.dependency_overrides[get_pool] = lambda: BrokenPool()
    fake_llm.status_result = ModelStatus(False, "fake-model", None, "Ollama isn't running.")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert (response.json()["database"]["ok"], response.json()["ollama"]["ok"]) == (False, False)


# ---------------------------------------------------------- foreign origins


@pytest.fixture
def populated(client, conn, add_document):
    document_id, _ = add_document(conn, "Doc", "d.md", [("a chunk of text for the test", None, None)])
    return document_id


def test_post_with_a_foreign_origin_is_refused_and_nothing_happens(client, conn, fake_llm):
    response = client.post("/api/ask", json={"question": "hello"}, headers={"Origin": EVIL})
    assert response.status_code == 403
    assert fake_llm.calls == []

    upload = client.post("/api/ingest", files={"file": ("x.md", b"# T\n\n" + b"word " * 50, "text/markdown")}, headers={"Origin": EVIL})
    assert upload.status_code == 403
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_delete_with_a_foreign_origin_is_refused_and_the_document_stays(client, conn, populated):
    assert client.delete(f"/api/documents/{populated}", headers={"Origin": EVIL}).status_code == 403
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


@pytest.mark.parametrize("origin", ["null", "http://localhost:5174", "https://localhost:5173", "http://localhost", "http://127.0.0.1:8000", "file://"])
def test_lookalike_and_null_origins_are_refused(client, origin):
    assert client.post("/api/ask", json={"question": "hi"}, headers={"Origin": origin}).status_code == 403


@pytest.mark.parametrize("method", ["PUT", "PATCH"])
def test_other_changing_methods_are_checked_too(client, method):
    response = client.request(method, "/api/documents/1", headers={"Origin": EVIL})
    assert response.status_code == 403  # refused by the origin check before routing


@pytest.mark.parametrize("origin", [ALLOWED, ALLOWED_ALT, ALLOWED + "/"])
def test_post_with_an_allowed_origin_works(client, conn, add_document, origin):
    add_document(conn, "Doc", "d.md", [("a chunk of text for the test", None, None)])
    response = client.post("/api/ask", json={"question": "a chunk of text"}, headers={"Origin": origin})
    assert response.status_code == 200 and response.text.startswith("event: sources")


def test_post_without_an_origin_header_works(client):
    response = client.post("/api/ask", json={"question": "hello"})
    assert response.status_code == 200  # curl and the tests send no Origin


def test_reading_with_a_foreign_origin_is_not_blocked_by_the_server_but_gets_no_cors_headers(client):
    response = client.get("/api/documents", headers={"Origin": EVIL})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers  # so the browser will not let the page read it


# ----------------------------------------------------------------------- CORS


@pytest.mark.parametrize("origin", [ALLOWED, ALLOWED_ALT])
def test_cors_headers_are_sent_for_allowed_origins_only(client, origin):
    response = client.get("/api/documents", headers={"Origin": origin})
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-origin"] != "*"
    assert "access-control-allow-credentials" not in response.headers


def test_cors_preflight(client):
    allowed = client.options(
        "/api/ask",
        headers={"Origin": ALLOWED, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ALLOWED
    assert "POST" in allowed.headers["access-control-allow-methods"]

    refused = client.options("/api/ask", headers={"Origin": EVIL, "Access-Control-Request-Method": "POST"})
    assert refused.status_code == 400 and "access-control-allow-origin" not in refused.headers


def test_a_wildcard_origin_is_never_sent(client):
    for origin in (ALLOWED, EVIL, None):
        headers = {"Origin": origin} if origin else {}
        assert client.get("/api/health", headers=headers).headers.get("access-control-allow-origin") != "*"


# ------------------------------------------------------------------ Host header


@pytest.mark.parametrize("base_url", ["http://localhost", "http://127.0.0.1", "http://localhost:8000", "http://127.0.0.1:8000"])
def test_localhost_hosts_are_accepted(app, base_url):
    with TestClient(app, base_url=base_url) as test_client:
        assert test_client.get("/api/health").status_code == 200


@pytest.mark.parametrize("base_url", ["http://evil.example", "http://localhost.evil.example", "http://192.168.1.20:8000", "http://[::1]:8000", "http://0.0.0.0:8000"])
def test_other_hosts_are_rejected(app, base_url):
    with TestClient(app, base_url=base_url) as test_client:
        assert test_client.get("/api/health").status_code == 400


def test_a_forged_host_header_is_rejected(client):
    assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 400


# ----------------------------------------------------------------- size limits


def test_a_huge_json_body_is_refused_before_it_is_read(client, fake_llm):
    big = b'{"question": "' + b"x" * 200_000 + b'"}'
    response = client.post("/api/ask", content=big, headers={"Content-Type": "application/json"})
    assert response.status_code == 413 and fake_llm.calls == []


# -------------------------------------------------------------- request ids


@pytest.mark.parametrize("path", ["/api/health", "/api/nothing-here"])
def test_every_response_has_a_request_id(client, path):
    response = client.get(path)
    assert len(response.headers["x-request-id"]) == 12


def test_refused_requests_also_get_a_request_id(client):
    assert "x-request-id" in client.post("/api/ask", json={"question": "x"}, headers={"Origin": EVIL}).headers


# -------------------------------------------------------------------- settings


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr("src.api.config.load_dotenv", lambda *a, **k: None)
    for name in ("ALLOWED_ORIGINS", "MAX_UPLOAD_MB"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_settings(clean_env):
    settings = load_settings()
    assert settings.allowed_origins == DEFAULT_ORIGINS == (ALLOWED, ALLOWED_ALT)
    assert settings.max_upload_bytes == 25 * 1024 * 1024
    assert settings.allowed_hosts == ("localhost", "127.0.0.1")


def test_allowed_origins_can_be_overridden(clean_env):
    clean_env.setenv("ALLOWED_ORIGINS", "http://localhost:3000, https://app.example/ ")
    assert load_settings().allowed_origins == ("http://localhost:3000", "https://app.example")


@pytest.mark.parametrize("value", ["*", "http://*.example.com", "http://ok:1,*", " , ", ","])
def test_wildcard_or_empty_origin_lists_are_refused(clean_env, value):
    clean_env.setenv("ALLOWED_ORIGINS", value)
    with pytest.raises(ValueError):
        load_settings()  # a broken security setting must fail loudly, not fall back silently


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_origins_variable_means_not_set(clean_env, value):
    clean_env.setenv("ALLOWED_ORIGINS", value)
    assert load_settings().allowed_origins == DEFAULT_ORIGINS


def test_parse_origins_rejects_an_empty_list():
    with pytest.raises(ValueError):
        parse_origins(" , ")


def test_max_upload_size_can_be_overridden(clean_env):
    clean_env.setenv("MAX_UPLOAD_MB", "2.5")
    assert load_settings().max_upload_bytes == int(2.5 * 1024 * 1024)


@pytest.mark.parametrize("value", ["abc", "0", "-5"])
def test_invalid_upload_limits_are_refused(clean_env, value):
    clean_env.setenv("MAX_UPLOAD_MB", value)
    with pytest.raises(ValueError):
        load_settings()


def test_a_custom_origin_list_is_what_the_app_enforces(pool, embedder, fake_llm, tmp_path):
    settings = Settings(allowed_origins=("http://myapp.test:3000",), upload_dir=tmp_path / "up")
    custom = create_app(settings=settings, with_lifespan=False)
    custom.dependency_overrides[get_pool] = lambda: pool
    custom.dependency_overrides[get_llm] = lambda: fake_llm
    from src.api.deps import get_embedder

    custom.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(custom, base_url="http://localhost") as test_client:
        assert test_client.post("/api/ask", json={"question": "q"}, headers={"Origin": "http://myapp.test:3000"}).status_code == 200
        assert test_client.post("/api/ask", json={"question": "q"}, headers={"Origin": ALLOWED}).status_code == 403
