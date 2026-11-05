"""Open connections to the project's Postgres database.

Settings come from the environment (loaded from the project's .env file):
POSTGRES_USER, POSTGRES_PASSWORD and POSTGRES_DB are required;
POSTGRES_HOST (default 127.0.0.1) and POSTGRES_PORT (default 5433) are optional.
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
REQUIRED = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


def connection_params(dbname: str | None = None) -> dict:
    """Keyword arguments for psycopg.connect, read from the environment."""
    # Variables already set in the environment win over the .env file.
    load_dotenv(ENV_FILE)
    missing = [key for key in REQUIRED if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            f"Missing database settings: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in ({ENV_FILE})."
        )
    return {
        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.environ.get("POSTGRES_PORT", "5433")),
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": dbname or os.environ["POSTGRES_DB"],
    }


def get_connection(
    dbname: str | None = None, *, vector: bool = True, autocommit: bool = False
) -> psycopg.Connection:
    """Connect to the database (POSTGRES_DB unless `dbname` is given).

    With vector=True the pgvector type is registered, so Python lists and
    numpy arrays can be stored in and read from `vector` columns directly.
    That needs the `vector` extension to exist in the database.
    """
    conn = psycopg.connect(**connection_params(dbname), autocommit=autocommit)
    if vector:
        try:
            register_vector(conn)
        except Exception:
            conn.close()
            raise
    return conn
