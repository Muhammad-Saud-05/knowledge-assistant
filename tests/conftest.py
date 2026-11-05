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
