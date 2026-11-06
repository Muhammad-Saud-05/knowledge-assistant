"""Combine several ranked result lists into one with reciprocal rank fusion.

Why reciprocal rank fusion (RRF) and not adding the scores together:

Vector search scores are cosine similarities (about 0.4 to 0.8 on this corpus),
while keyword scores are ts_rank_cd values on a completely different scale
(about 0.1 to 0.9, and they change with the length of the query). Adding or
averaging numbers that mean different things lets whichever method uses bigger
numbers drown out the other. RRF ignores the scores and uses only each result's
position (rank) in each list: a chunk earns 1 / (k + rank) from every list it
appears in, so a chunk that both methods rank highly rises to the top, and no
tuning of scales is needed. k (60 by default) softens the gap between rank 1
and rank 2, so a single list cannot dominate by putting one chunk first.
"""

from dataclasses import replace
from fractions import Fraction

from .models import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K, limit: int | None = None
) -> list[RetrievedChunk]:
    """Merge ranked lists (best first) into one list, best first.

    A chunk's fused score is the sum over the lists containing it of 1 / (k + rank),
    with rank starting at 1. Chunks are de-duplicated by chunk_id. Each returned
    chunk keeps the ranks and scores that the input lists gave it (vector_rank,
    keyword_rank, ...) and a `list_ranks` tuple with its rank in every input list
    (None where absent). Ties are broken by the best single rank, then by chunk_id,
    so the order is always the same for the same input.
    """
    if k < 0:
        raise ValueError("k must not be negative")

    scores: dict[int, Fraction] = {}  # exact arithmetic: equal sums are truly equal
    list_ranks: dict[int, list[int | None]] = {}
    merged: dict[int, RetrievedChunk] = {}
    n_lists = len(result_lists)

    for list_index, results in enumerate(result_lists):
        rank = 0
        for chunk in results:
            if list_ranks.get(chunk.chunk_id, [None] * n_lists)[list_index] is not None:
                continue  # the same chunk twice in one list: only its first place counts
            rank += 1
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, Fraction(0)) + Fraction(1, k + rank)
            list_ranks.setdefault(chunk.chunk_id, [None] * n_lists)[list_index] = rank

            if chunk.chunk_id not in merged:
                merged[chunk.chunk_id] = replace(chunk)
            else:  # fill in details that only this list knew
                existing = merged[chunk.chunk_id]
                for name in ("vector_rank", "vector_score", "keyword_rank", "keyword_score"):
                    if getattr(existing, name) is None:
                        setattr(existing, name, getattr(chunk, name))

    def sort_key(chunk_id: int):
        best_rank = min(r for r in list_ranks[chunk_id] if r is not None)
        return (-scores[chunk_id], best_rank, chunk_id)

    fused = []
    for chunk_id in sorted(scores, key=sort_key):
        chunk = merged[chunk_id]
        chunk.fused_score = float(scores[chunk_id])
        chunk.score = chunk.fused_score  # for fused results the main score is the fused score
        chunk.list_ranks = tuple(list_ranks[chunk_id])
        fused.append(chunk)
    return fused[:limit] if limit is not None else fused
