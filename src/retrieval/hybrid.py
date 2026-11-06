"""One entry point for searching: vector, keyword, or both combined."""

from .filters import SearchFilters
from .fusion import reciprocal_rank_fusion
from .keyword import keyword_search
from .models import RetrievedChunk
from .vector import vector_search

MODES = ("vector", "keyword", "hybrid")
DEFAULT_CANDIDATES = 30


def search(
    conn,
    embedder,
    query: str,
    k: int = 5,
    mode: str = "hybrid",
    filters: SearchFilters | None = None,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[RetrievedChunk]:
    """The top k chunks for a query.

    mode "vector":  nearest by meaning.
    mode "keyword": best word match.
    mode "hybrid":  take `candidates` results from each of the two methods, fuse
                    them with reciprocal rank fusion and return the best k.

    About `score` in the results: it is the score of the method that produced them
    (cosine similarity for vector, ts_rank_cd for keyword). For hybrid results it
    is the fused score, which is NOT a similarity. It is just the sum of 1 / (60 +
    rank) over the lists, so it is always small (at most about 0.033 for two lists)
    and says only how high the chunk ranked, never how relevant it really is. It
    must not be used as a cutoff for deciding that nothing relevant was found.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of: {', '.join(MODES)}")
    if k < 1:
        raise ValueError("k must be at least 1")

    if mode == "vector":
        return vector_search(conn, embedder, query, k=k, filters=filters)
    if mode == "keyword":
        return keyword_search(conn, query, k=k, filters=filters)

    if candidates < 1:
        raise ValueError("candidates must be at least 1")
    vector_results = vector_search(conn, embedder, query, k=candidates, filters=filters)
    keyword_results = keyword_search(conn, query, k=candidates, filters=filters)
    return reciprocal_rank_fusion([vector_results, keyword_results], limit=k)
