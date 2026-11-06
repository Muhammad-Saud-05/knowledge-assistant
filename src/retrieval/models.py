from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    title: str | None
    path: str
    page_number: int | None
    section_heading: str | None
    text: str
    # The main score of the method that produced this result:
    #   vector search -> cosine similarity (1 - distance), higher is closer
    #   keyword search -> ts_rank_cd rank, higher is a better match
    #   hybrid search -> the fused score, which is NOT a similarity
    score: float

    # Where the chunk was found, for debugging and for showing in the results.
    # A rank starts at 1; None means that method did not return the chunk.
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    fused_score: float | None = None
    # Rank in each list that was fused, in the order the lists were given.
    list_ranks: tuple[int | None, ...] | None = None
