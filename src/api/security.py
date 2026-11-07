"""Protection for an app that runs on your own computer.

Why this is needed, in plain language:

The server listens on 127.0.0.1, so other computers cannot reach it. But YOUR browser
can: any website you have open in another tab can make your browser send a request to
http://127.0.0.1:8000, for example a hidden form that posts a file to /api/ingest or
deletes a document. The browser's CORS rules only stop that page from READING the reply;
they do not stop the request from being SENT and acted on. So this module does four things:

1. CORS allows exactly the web addresses of our own front end (no wildcard), so only the
   front end may read replies.
2. Any request that changes something (POST, PUT, PATCH, DELETE) is refused with 403 when
   it carries an Origin header that is not on the allowed list. Browsers always attach
   Origin to such requests, so a foreign website is rejected before anything happens.
   Tools such as curl and the tests send no Origin header, and are allowed.
3. Only the Host names "localhost" and "127.0.0.1" are accepted, which blocks DNS
   rebinding (a trick where a hostile site's name is pointed at 127.0.0.1).
4. Oversized request bodies are refused up front.

There is deliberately no login: this is a single-user app on your own machine, and the
checks above are what keep other websites out. Do not expose the server to a network
without adding real authentication first.
"""

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings

STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
INGEST_PATH = "/api/ingest"
# Room for the multipart envelope around an uploaded file.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


async def _send_json(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _header(scope, name: bytes) -> str | None:
    for key, value in scope["headers"]:
        if key == name:
            return value.decode("latin-1")
    return None


class OriginCheckMiddleware:
    """Refuse changes asked for by a web page from an origin we do not know."""

    def __init__(self, app, allowed_origins: tuple[str, ...]):
        self.app = app
        self.allowed = {origin.lower() for origin in allowed_origins}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in STATE_CHANGING_METHODS:
            origin = _header(scope, b"origin")
            if origin is not None and origin.strip().rstrip("/").lower() not in self.allowed:
                await _send_json(send, 403, "This origin is not allowed to make changes.")
                return
        await self.app(scope, receive, send)


class BodyLimitMiddleware:
    """Refuse a request whose declared size is too big, before reading any of it."""

    def __init__(self, app, max_upload_bytes: int, max_json_body_bytes: int):
        self.app = app
        self.upload_limit = max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        self.json_limit = max_json_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in STATE_CHANGING_METHODS:
            declared = _header(scope, b"content-length")
            if declared is not None:
                try:
                    size = int(declared)
                except ValueError:
                    await _send_json(send, 400, "Invalid Content-Length header.")
                    return
                limit = self.upload_limit if scope["path"] == INGEST_PATH else self.json_limit
                if size > limit:
                    await _send_json(send, 413, "The request is too large.")
                    return
        await self.app(scope, receive, send)


def install_security(app: FastAPI, settings: Settings) -> None:
    """Add the protections. Middleware added LAST is the OUTERMOST, so add innermost first."""
    app.add_middleware(
        BodyLimitMiddleware,
        max_upload_bytes=settings.max_upload_bytes,
        max_json_body_bytes=settings.max_json_body_bytes,
    )
    app.add_middleware(OriginCheckMiddleware, allowed_origins=settings.allowed_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),  # exact origins, never "*"
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        expose_headers=["X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
