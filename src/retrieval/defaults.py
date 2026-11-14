"""Which retrieval the application uses unless it is told otherwise.

The default is configuration C from the evaluation report: the 30 nearest chunks by vector search,
reordered by the cross-encoder reranker, with the best k (5) going into the prompt.

Configuration L (the question is split into sub-questions by the local language model, each is searched
and reranked, and the lists are merged round-robin) is available by name (mode "rerank-split-llm") and
as an opt-in default. The Phase 2b decision rule selected L, but it is not the default because it adds
about 10 seconds per question (see the addendum in docs/eval_report.md).

Two switches in the environment (or .env):
  DECOMPOSITION_ENABLED  false by default. true makes L the default search.
  RERANKER_ENABLED       true by default. false means no reranker and no splitting: plain hybrid
                         search (vector and keyword results fused with reciprocal rank fusion)
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


def parse_enabled(text: str | None, name: str = "RERANKER_ENABLED", default: bool = True) -> bool:
    """A true/false setting: `default` when unset or empty. Anything but a clear yes or no is an error."""
    value = (text or "").strip().lower()
    if not value:
        return default
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {text!r}")


def reranker_enabled() -> bool:
    load_dotenv(ENV_FILE)  # variables already set in the environment win
    return parse_enabled(os.environ.get("RERANKER_ENABLED"))


def decomposition_enabled() -> bool:
    load_dotenv(ENV_FILE)
    return parse_enabled(os.environ.get("DECOMPOSITION_ENABLED"), "DECOMPOSITION_ENABLED", default=False)


def resolve_mode(
    requested: str | None,
    reranker_available: bool,
    decomposer_available: bool = False,
    split_by_default: bool = False,
) -> str:
    """The mode to search with. None or "default" gives configuration C when a reranker is loaded (plain
    hybrid otherwise), or configuration L instead of C when split_by_default is on and a decomposer is
    available. Asking by name for a mode that needs something that is not available is an error."""
    if requested in (None, "default"):
        if reranker_available and decomposer_available and split_by_default:
            return SPLIT_MODE
        return RERANK_MODE if reranker_available else PLAIN_MODE
    if requested not in SELECTABLE_MODES:
        raise ValueError(f"unknown mode {requested!r}; use one of: default, {', '.join(SELECTABLE_MODES)}")
    if requested in (RERANK_MODE, SPLIT_MODE) and not reranker_available:
        raise ValueError("reranking is switched off (RERANKER_ENABLED=false), so this mode is not available")
    if requested == SPLIT_MODE and not decomposer_available:
        raise ValueError("question splitting is not available on this server, so this mode cannot be used")
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
