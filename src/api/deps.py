"""FastAPI dependencies: how routes get the pool, embedder, language model and settings.

Everything is looked up on app.state, filled in by the lifespan handler in main.py.
Routes only ever ask for these through Depends(...), so tests can replace any of
them with a fake using app.dependency_overrides.
"""

import asyncio

from fastapi import HTTPException, Request

from .config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_pool(request: Request):
    """Something with a .connection() context manager that yields a database connection."""
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="The server is still starting up.")
    return pool


def get_embedder(request: Request):
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(status_code=503, detail="The server is still starting up.")
    return embedder


def get_llm(request: Request):
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(status_code=503, detail="The server is still starting up.")
    return llm


def get_gate(request: Request) -> asyncio.Semaphore:
    """The one-at-a-time gate for answer generation.

    A CPU can only run one language model generation well at a time, so /api/ask
    requests take turns. Created on first use so that it belongs to the running
    event loop.
    """
    gate = getattr(request.app.state, "gate", None)
    if gate is None:
        gate = request.app.state.gate = asyncio.Semaphore(1)
    return gate
