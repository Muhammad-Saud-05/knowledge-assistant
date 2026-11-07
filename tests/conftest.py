import hashlib
import math
import random
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from src.db.connection import connection_params, get_connection

TEST_DB = "kb_test"
INIT_SQL = Path(__file__).resolve().parents[1] / "db" / "init.sql"


# ------------------------------------------------------------ test database


@pytest.fixture(scope="session")
def test_database():
    """Create a throwaway database on the same server, and drop it afterwards."""
    try:
        params = connection_params("postgres")
    except RuntimeError as error:
        pytest.skip(f"Database tests skipped: {error}")
    # Never create or drop a database that has the same name as the real one.
    assert TEST_DB != connection_params()["dbname"]

    try:
        admin = psycopg.connect(**params, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError:
        pytest.skip(
            "Database tests skipped: Postgres is not reachable "
            "(start it with `docker compose up -d`)."
        )

    name = sql.Identifier(TEST_DB)
    admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))
    admin.execute(sql.SQL("CREATE DATABASE {}").format(name))
    try:
        # The vector type is not registered yet: init.sql is what creates the extension.
        with get_connection(TEST_DB, vector=False, autocommit=True) as setup:
            setup.execute(INIT_SQL.read_text(encoding="utf-8"))
        yield TEST_DB
    finally:
        admin.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
            (TEST_DB,),
        )
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))
        admin.close()


@pytest.fixture
def conn(test_database):
    """A connection to the test database, with empty tables for every test."""
    connection = get_connection(test_database)
    # Safety net: never empty tables in any database other than the test one.
    current = connection.execute("SELECT current_database()").fetchone()[0]
    assert current == TEST_DB, f"refusing to touch database {current!r}"
    connection.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    connection.commit()
    yield connection
    connection.rollback()
    connection.close()


# ------------------------------------------------------------ fake embedders


def fake_vector(text: str) -> list[float]:
    """A repeatable, normalised 384-number vector derived from the text."""
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    values = [rng.gauss(0, 1) for _ in range(384)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class FakeEmbedder:
    """Fast stand-in for the real model. Records how many texts it embedded."""

    def __init__(self):
        self.texts_embedded = 0

    def embed_documents(self, texts):
        self.texts_embedded += len(texts)
        return [fake_vector(t) for t in texts]

    def embed_query(self, text):
        return fake_vector(text)


class WrongSizeEmbedder(FakeEmbedder):
    """Returns 10-number vectors, which the vector(384) column rejects."""

    def embed_documents(self, texts):
        return [[0.1] * 10 for _ in texts]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def wrong_size_embedder():
    return WrongSizeEmbedder()


# ------------------------------------------------------------ data helpers


@pytest.fixture
def add_document(embedder):
    """Store a document with chunks straight into the test database.

    Each chunk is (text, page_number, section_heading); doc_date is optional. The stored vector is the
    fake embedder's vector for the chunk text, so a query worded exactly like a
    chunk's text finds that chunk with similarity 1.0.
    """
    from pgvector import Vector

    def add(conn, title, path, chunks, source_type="markdown", doc_date=None):
        doc_id = conn.execute(
            "INSERT INTO documents (source_type, path, title, content_hash, doc_date) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (source_type, path, title, "hash-" + path, doc_date),
        ).fetchone()[0]
        chunk_ids = []
        for index, (text, page, heading) in enumerate(chunks):
            chunk_ids.append(
                conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, text, section_heading, "
                    "page_number, embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (doc_id, index, text, heading, page, Vector(embedder.embed_documents([text])[0])),
                ).fetchone()[0]
            )
        conn.commit()
        return doc_id, chunk_ids

    return add


# ------------------------------------------------------------- web API helpers


class FakePool:
    """Stands in for the connection pool: each .connection() is a fresh connection to the test database."""

    def __init__(self, dbname):
        self.dbname = dbname

    def connection(self):
        return get_connection(self.dbname)  # as a context manager: commits on success, then closes


class FakeApiLLM:
    """A scripted language model for API tests. Records every prompt it receives."""

    def __init__(self, reply="A cited answer [1].", error=None):
        from src.llm.base import ModelStatus

        self.reply = reply
        self.error = error  # an exception to raise when asked to generate
        self.calls = []
        self.model = "fake-model"
        self.status_result = ModelStatus(True, "fake-model", True, None)

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply

    def stream(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        for i in range(0, len(self.reply), 6):
            yield self.reply[i : i + 6]

    def status(self):
        return self.status_result

    def warm_up(self):
        pass


@pytest.fixture
def pool(test_database):
    return FakePool(test_database)


@pytest.fixture
def api_settings(tmp_path):
    from src.api.config import Settings

    return Settings(upload_dir=tmp_path / "data" / "uploads", max_upload_bytes=64 * 1024)


@pytest.fixture
def fake_llm():
    return FakeApiLLM()


@pytest.fixture
def app(conn, test_database, embedder, fake_llm, api_settings):
    """The real app (routes, middleware) with the pool, embedder and model replaced by fakes."""
    from src.api.deps import get_embedder, get_llm, get_pool
    from src.api.main import create_app

    application = create_app(settings=api_settings, with_lifespan=False)
    application.dependency_overrides[get_pool] = lambda: FakePool(test_database)
    application.dependency_overrides[get_embedder] = lambda: embedder
    application.dependency_overrides[get_llm] = lambda: fake_llm
    return application


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    # Used as a context manager so all requests share one event loop.
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client
