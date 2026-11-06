"""Keyword (exact-word) retrieval over the chunks' full-text index.

Vector search is good at meaning but can miss exact things such as an error
code, a command or a rare name. Keyword search finds those, so the two are
combined (see hybrid.py).
"""

from .filters import SearchFilters, build_filter_sql
from .models import RetrievedChunk

# How the question becomes a search expression:
#
# plainto_tsquery() is the safe starting point. It treats the text as plain words:
# it ignores every symbol (& | ! : * ( ) quotes ...) so no input can cause a syntax
# error, it drops stop words ("how", "the", "from") and it stems what is left
# ("requests" -> "request"). But it joins the words with AND, so a long question
# would only match a chunk containing every single word, which almost never
# happens. We therefore turn its AND (" & ") into OR (" | ") in the text form of the
# query. Matching the operator with its surrounding spaces leaves a "&" that is
# part of a quoted word (such as inside a URL) untouched. With OR, a chunk matches if
# it has any of the words, and ranking puts the chunks with more of them first.
# If nothing usable is left (only stop words or symbols) the expression has no
# terms (numnode = 0) and no rows are returned.
#
# Ranking uses ts_rank_cd, which rewards query words that occur close together
# (a phrase-like match scores higher than the same words far apart). Length
# normalisation flag 1 divides the rank by 1 + log(length): chunks here range from
# about 20 to 300 words, and without any normalisation a long chunk would win just
# by containing more words. Flag 2 (divide by length) would be too strong: a tiny
# chunk that mentions one word would beat a long chunk that answers the question.
DEFAULT_NORMALIZATION = 1

_SEARCH_SQL = """
    WITH q AS (
        SELECT replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')::tsquery AS tsq
    )
    SELECT c.id, c.document_id, d.title, d.path, c.page_number,
           c.section_heading, c.text,
           ts_rank_cd(c.tsv, q.tsq, %(normalization)s) AS score
    FROM q
    JOIN chunks c ON c.tsv @@ q.tsq
    JOIN documents d ON d.id = c.document_id
    WHERE numnode(q.tsq) > 0 AND {condition}
    ORDER BY score DESC, c.id
    LIMIT %(k)s
"""


def keyword_search(
    conn,
    query: str,
    k: int = 30,
    filters: SearchFilters | None = None,
    normalization: int = DEFAULT_NORMALIZATION,
) -> list[RetrievedChunk]:
    """The k chunks that best match the words of the query, best first.

    Returns an empty list when the query has no usable words. Any text is accepted.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    query = query.replace("\x00", "")  # Postgres text cannot contain NUL
    if not query.strip():
        return []

    condition, filter_params = build_filter_sql(filters)
    params = {"query": query, "k": k, "normalization": normalization, **filter_params}
    rows = conn.execute(_SEARCH_SQL.format(condition=condition), params).fetchall()

    results = []
    for rank, row in enumerate(rows, start=1):
        chunk = RetrievedChunk(*row)
        chunk.keyword_rank = rank
        chunk.keyword_score = chunk.score
        results.append(chunk)
    return results
