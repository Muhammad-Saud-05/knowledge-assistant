"""A small pool of reusable database connections for the web server."""

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .connection import connection_params


def _configure(conn) -> None:
    register_vector(conn)
    conn.commit()  # a pooled connection must be idle (no open transaction) after setup


def create_pool(min_size: int = 1, max_size: int = 5, timeout: float = 5.0) -> ConnectionPool:
    """A closed pool; call .open() to start it.

    Credentials are passed as connection arguments, never put into a text string.
    `timeout` is how long a request waits for a connection before giving up, so a
    stopped database shows up quickly as an error instead of hanging.
    """
    return ConnectionPool(
        conninfo="",
        kwargs=connection_params(),
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        configure=_configure,
        check=ConnectionPool.check_connection,  # drop connections the database has closed
        open=False,
    )
