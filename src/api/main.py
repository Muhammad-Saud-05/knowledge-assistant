"""The web API.

Start it (localhost only) with:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Interactive documentation is served at /docs; every API route is under /api.
"""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ..db.pool import create_pool
from ..ingestion.embedder import Embedder
from ..llm.ollama import OllamaLLM
from .config import Settings, load_settings
from .observability import RequestLogMiddleware, configure_logging, request_id_of

logger = logging.getLogger("kb.api")


def _warm_up(llm) -> None:
    """Ask Ollama to load the model now, so the first question does not wait for it.

    Runs in the background and never stops the server from working if it fails
    (for example because Ollama is not running yet).
    """
    try:
        llm.warm_up()
        logger.info("language model loaded and ready")
    except Exception as error:  # noqa: BLE001 - any failure here is non-fatal
        logger.warning("language model warm-up failed: %s", str(error)[:200])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open the database pool, load the embedding model once (a few seconds),
    # create the language model client and start loading the model in the background.
    pool = create_pool()
    pool.open(wait=False)  # keep going even if the database is down; /api/health reports it
    app.state.pool = pool

    embedder = Embedder()
    await run_in_threadpool(lambda: embedder.model)
    app.state.embedder = embedder

    llm = OllamaLLM()
    app.state.llm = llm
    threading.Thread(target=_warm_up, args=(llm,), daemon=True, name="llm-warm-up").start()
    logger.info("server ready")
    try:
        yield
    finally:
        await run_in_threadpool(pool.close)  # shutdown: close the database connections


def create_app(settings: Settings | None = None, with_lifespan: bool = True) -> FastAPI:
    """Build the app. Tests pass with_lifespan=False and override the dependencies."""
    configure_logging()
    settings = settings or load_settings()
    app = FastAPI(
        title="Personal Knowledge Assistant",
        lifespan=lifespan if with_lifespan else None,
    )
    app.state.settings = settings

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, error: Exception):
        # Log only the kind of error and the request id, never the request's content.
        logger.error("unhandled error id=%s type=%s", request_id_of(request), type(error).__name__)
        return JSONResponse({"detail": "Something went wrong on the server."}, status_code=500)

    # Added last = outermost: every request, even a rejected one, gets an id and a log line.
    app.add_middleware(RequestLogMiddleware)
    return app


app = create_app()
