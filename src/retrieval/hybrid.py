"""One entry point for searching: vector, keyword, or both combined."""

from .filters import SearchFilters
from .fusion import reciprocal_rank_fusion
from .keyword import keyword_search
from .models import RetrievedChunk
from .vector import vector_search

MODES = ("vector", "keyword", "hybrid")  # the modes every caller can use
BASE_MODES = MODES + ("hybrid-bm25",)  # hybrid-bm25 is an experiment and needs bm25=
ALL_MODES = BASE_MODES + tuple(f"{m}+rerank" for m in BASE_MODES)  # +rerank needs reranker=
DEFAULT_CANDIDATES = 30


def parse_mode(mode: str) -> tuple[str, bool]:
    """Split a mode name: hybrid+rerank gives ("hybrid", True). Unknown names raise ValueError."""
    if mode not in ALL_MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of: {', '.join(ALL_MODES)}")
    base, plus, _ = mode.partition("+")
    return base, bool(plus)


def search(
    conn,
    embedder,
    query: str,
    k: int = 5,
    mode: str = "hybrid",
    filters: SearchFilters | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    reranker=None,
    rerank_candidates: int = DEFAULT_CANDIDATES,
    bm25=None,
) -> list[RetrievedChunk]:
    """The top k chunks for a query.

    mode "vector":  nearest by meaning.
    mode "keyword": best word match.
    mode "hybrid":  take `candidates` results from each of the two methods, fuse
                    them with reciprocal rank fusion and return the best k.
    mode "hybrid-bm25": the same, with a BM25 leg (bm25=Bm25Searcher) in place of Postgres full-text.
    Add "+rerank" to any mode (and pass reranker=): the first stage then returns
    `rerank_candidates` results, which the cross-encoder reorders; the best k are returned and their
    `score` is the raw rerank score. Without a reranker nothing changes.

    About `score` in the results: it is the score of the method that produced them
    (cosine similarity for vector, ts_rank_cd for keyword). For hybrid results it
    is the fused score, which is NOT a similarity. It is just the sum of 1 / (60 +
    rank) over the lists, so it is always small (at most about 0.033 for two lists)
    and says only how high the chunk ranked, never how relevant it really is. It
    must not be used as a cutoff for deciding that nothing relevant was found.
    """
    base, rerank = parse_mode(mode)
    if k < 1:
        raise ValueError("k must be at least 1")
    if rerank:
        if reranker is None:
            raise ValueError(f"mode {mode!r} needs a reranker")
        if rerank_candidates < 1:
            raise ValueError("rerank_candidates must be at least 1")
    elif reranker is not None:
        raise ValueError("a reranker was given but the mode does not use one; add +rerank to the mode")

    # With a reranker, the first stage returns rerank_candidates results and the reranker keeps k.
    first_stage_k = rerank_candidates if rerank else k
    results = _first_stage(conn, embedder, query, first_stage_k, base, filters, candidates, bm25)
    if rerank:
        return reranker.rerank(query, results, k)
    return results


def _first_stage(conn, embedder, query, k, base, filters, candidates, bm25) -> list[RetrievedChunk]:
    if base == "vector":
        return vector_search(conn, embedder, query, k=k, filters=filters)
    if base == "keyword":
        return keyword_search(conn, query, k=k, filters=filters)

    if candidates < 1:
        raise ValueError("candidates must be at least 1")
    vector_results = vector_search(conn, embedder, query, k=candidates, filters=filters)
    if base == "hybrid-bm25":
        if bm25 is None:
            raise ValueError("mode 'hybrid-bm25' needs a Bm25Searcher (bm25=...)")
        keyword_results = bm25.search(query, k=candidates, filters=filters)
    else:
        keyword_results = keyword_search(conn, query, k=candidates, filters=filters)
    return reciprocal_rank_fusion([vector_results, keyword_results], limit=k)
