"""Which retrieval the application uses unless it is told otherwise.

The default is configuration C from the evaluation report: the 30 nearest chunks by vector search,
reordered by the cross-encoder reranker, with the best k (5) going into the prompt. Setting
RERANKER_ENABLED=false in the environment (or .env) switches back to plain hybrid search (vector and
keyword results fused with reciprocal rank fusion), the previous default.
"""

import os

from dotenv import load_dotenv

from ..db.connection import ENV_FILE
from .rerank import Reranker

RERANK_MODE = "vector+rerank"  # configuration C
PLAIN_MODE = "hybrid"  # configuration A, used when the reranker is switched off
RERANK_CANDIDATES = 30
# Modes a user may ask for by name; "default" (or None) means "whatever this server is set up for".
SELECTABLE_MODES = ("vector", "keyword", "hybrid", RERANK_MODE)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_enabled(text: str | None) -> bool:
    """RERANKER_ENABLED: true when unset or empty. Anything but a clear yes or no is an error."""
    value = (text or "").strip().lower()
    if not value or value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"RERANKER_ENABLED must be true or false, got {text!r}")


def reranker_enabled() -> bool:
    load_dotenv(ENV_FILE)  # variables already set in the environment win
    return parse_enabled(os.environ.get("RERANKER_ENABLED"))


def resolve_mode(requested: str | None, reranker_available: bool) -> str:
    """The mode to search with. None or "default" gives configuration C when a reranker is loaded
    and plain hybrid otherwise. Asking for the reranked mode without a reranker is an error."""
    if requested in (None, "default"):
        return RERANK_MODE if reranker_available else PLAIN_MODE
    if requested not in SELECTABLE_MODES:
        raise ValueError(f"unknown mode {requested!r}; use one of: default, {', '.join(SELECTABLE_MODES)}")
    if requested == RERANK_MODE and not reranker_available:
        raise ValueError("reranking is switched off (RERANKER_ENABLED=false), so this mode is not available")
    return requested


class RerankerLoadError(RuntimeError):
    """The reranker model could not be loaded; the message says what to do about it."""


def load_reranker() -> Reranker:
    """Create the reranker and load the model now. Raises RerankerLoadError with a clear message."""
    reranker = Reranker()
    try:
        reranker.load()
    except Exception as error:  # noqa: BLE001 - any failure (download, corrupt files, memory) is fatal here
        raise RerankerLoadError(
            f"Could not load the reranker model {reranker.model_name} ({type(error).__name__}). "
            "Check RERANKER_MODEL and your internet connection for the first download, or set "
            "RERANKER_ENABLED=false to run without reranking."
        ) from error
    return reranker
