"""Request ids and request logging.

What is logged: a request id, the HTTP method, the path, the status code and how long it
took (and, for /api/ask, ids and timings: see ask.py). What is NEVER logged: question
or answer text, document text, upload file contents, query strings, or credentials.
This middleware only ever sees the method, the path and the status, so it cannot leak
more than that.
"""

import logging
import time
import uuid

LOGGER_NAME = "kb.api"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging() -> None:
    """Make our log lines visible when the server runs (once; safe to call again)."""
    kb_logger = logging.getLogger("kb")
    kb_logger.setLevel(logging.INFO)
    if not kb_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        kb_logger.addHandler(handler)


def request_id_of(scope_or_request) -> str:
    state = getattr(scope_or_request, "state", None)
    return getattr(state, "request_id", "-") if state is not None else "-"


class RequestLogMiddleware:
    """Gives every request an id (also sent back as X-Request-ID) and logs one line per request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = {"code": 500}

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            logger.info(
                "request id=%s method=%s path=%s status=%s ms=%d",
                request_id,
                scope["method"],
                scope["path"],
                status["code"],
                (time.perf_counter() - started) * 1000,
            )
