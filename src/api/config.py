"""Settings for the web API, read from the environment (or the project's .env file).

ALLOWED_ORIGINS  comma-separated web addresses allowed to call the API from a browser
                 (default: the Vite dev server on http://localhost:5173 and http://127.0.0.1:5173)
MAX_UPLOAD_MB    largest file /api/ingest accepts (default 25)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ..db.connection import ENV_FILE

PROJECT_ROOT = ENV_FILE.parent
DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_MAX_UPLOAD_MB = 25.0
# Only these host names may appear in the Host header (see security.py).
ALLOWED_HOSTS = ("localhost", "127.0.0.1")


@dataclass(frozen=True)
class Settings:
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS
    allowed_hosts: tuple[str, ...] = ALLOWED_HOSTS
    max_upload_bytes: int = int(DEFAULT_MAX_UPLOAD_MB * 1024 * 1024)
    upload_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    # A question and its filters are tiny; anything bigger than this is refused up front.
    max_json_body_bytes: int = 64 * 1024


def parse_origins(text: str) -> tuple[str, ...]:
    origins = tuple(o.strip().rstrip("/") for o in text.split(",") if o.strip())
    if not origins:
        raise ValueError("ALLOWED_ORIGINS is empty. Example: http://localhost:5173")
    if any(o == "*" or "*" in o for o in origins):
        raise ValueError("ALLOWED_ORIGINS must list exact origins; a wildcard is not allowed.")
    return origins


def load_settings() -> Settings:
    load_dotenv(ENV_FILE)  # variables already set in the environment win
    origins = DEFAULT_ORIGINS
    if os.environ.get("ALLOWED_ORIGINS", "").strip():
        origins = parse_origins(os.environ["ALLOWED_ORIGINS"])

    max_mb = DEFAULT_MAX_UPLOAD_MB
    raw = os.environ.get("MAX_UPLOAD_MB", "").strip()
    if raw:
        try:
            max_mb = float(raw)
        except ValueError:
            raise ValueError(f"MAX_UPLOAD_MB must be a number, got {raw!r}") from None
        if max_mb <= 0:
            raise ValueError("MAX_UPLOAD_MB must be greater than 0")
    return Settings(allowed_origins=origins, max_upload_bytes=int(max_mb * 1024 * 1024))
