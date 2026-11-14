"""Which retrieval the application uses unless it is told otherwise.

The default is configuration L from the evaluation report (Phase 2b, chosen by its pre-registered
rule): the question is split into sub-questions by the local language model (a question that asks one
thing is left alone); for each sub-question the 30 nearest chunks by vector search are reordered by the
cross-encoder reranker; the lists are merged round-robin and the best k (5) go into the prompt.

Two switches in the environment (or .env), each true by default:
  DECOMPOSITION_ENABLED=false  do not split questions: configuration C, vector search plus reranker
  RERANKER_ENABLED=false       no reranker and no splitting: plain hybrid search (vector and keyword
                               results fused with reciprocal rank fusion), the original default
"""

import os

from dotenv import load_dotenv

from ..db.connection import ENV_FILE
from .filters import SearchFilters
from .hybrid import search
from .models import RetrievedChunk
from .rerank import Reranker

SPLIT_MODE = "rerank-split-llm"  # configuration L
RERANK_MODE = "vector+rerank"  # configuration C
PLAIN_MODE = "hybrid"  # configuration A, used when the reranker is switched off
RERANK_CANDIDATES = 30
# Modes a user may ask for by name; "default" (or None) means "whatever this server is set up for".
SELECTABLE_MODES = ("vector", "keyword", "hybrid", RERANK_MODE, SPLIT_MODE)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_enabled(text: str | None, name: str = "RERANKER_ENABLED") -> bool:
    """A true/false setting: true when unset or empty. Anything but a clear yes or no is an error."""
    value = (text or "").strip().lower()
    if not value or value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {text!r}")


def reranker_enabled() -> bool:
    load_dotenv(ENV_FILE)  # variables already set in the environment win
    return parse_enabled(os.environ.get("RERANKER_ENABLED"))


def decomposition_enabled() -> bool:
    load_dotenv(ENV_FILE)
    return parse_enabled(os.environ.get("DECOMPOSITION_ENABLED"), "DECOMPOSITION_ENABLED")


def resolve_mode(requested: str | None, reranker_available: bool, decomposer_available: bool = False) -> str:
    """The mode to search with. None or "default" gives configuration L when a decomposer is available,
    otherwise C when a reranker is loaded, otherwise plain hybrid. Asking by name for a mode that needs
    something that is switched off is an error."""
    if requested in (None, "default"):
        if reranker_available and decomposer_available:
            return SPLIT_MODE
        return RERANK_MODE if reranker_available else PLAIN_MODE
    if requested not in SELECTABLE_MODES:
        raise ValueError(f"unknown mode {requested!r}; use one of: default, {', '.join(SELECTABLE_MODES)}")
    if requested in (RERANK_MODE, SPLIT_MODE) and not reranker_available:
        raise ValueError("reranking is switched off (RERANKER_ENABLED=false), so this mode is not available")
    if requested == SPLIT_MODE and not decomposer_available:
        raise ValueError("question splitting is switched off (DECOMPOSITION_ENABLED=false), so this mode is not available")
    return requested


def make_decomposer(model: str | None = None):
    """The language-model question splitter: temperature 0, JSON output, same model and context size as
    the answer model (so Ollama does not reload anything). It never raises: any failure means "do not split"."""
    from ..llm.ollama import OllamaLLM
    from .decompose import LLMDecomposer

    return LLMDecomposer(OllamaLLM(model=model, temperature=0, json_output=True))


def retrieve(
    conn,
    embedder,
    question: str,
    k: int,
    mode: str,
    filters: SearchFilters | None = None,
    reranker=None,
    decomposer=None,
) -> list[RetrievedChunk]:
    """The chunks for a question in any selectable mode (as returned by resolve_mode)."""
    if mode == SPLIT_MODE:
        from .decompose import search_decomposed

        return search_decomposed(
            conn, embedder, question, k, reranker, decomposer, filters, RERANK_CANDIDATES
        ).chunks
    return search(
        conn, embedder, question, k=k, mode=mode, filters=filters,
        reranker=reranker if mode.endswith("+rerank") else None, rerank_candidates=RERANK_CANDIDATES,
    )


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
