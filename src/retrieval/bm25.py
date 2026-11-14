"""BM25 keyword leg, an alternative to the Postgres full-text leg (an experiment).

The chunk texts are loaded from the database once per filter combination and indexed in memory with
rank_bm25's BM25Okapi (k1 = 1.5, b = 0.75). A chunk is returned only if it shares at least one
word with the question, which matches the OR behaviour of the Postgres leg. The existing Postgres
keyword search is untouched.
"""

import re

from .filters import SearchFilters, build_filter_sql
from .models import RetrievedChunk

_WORD = re.compile(r"\w+")
STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or so "
    "that the their them then there these this to was we what when where which who why will with "
    "would you your should could".split()
)

_LOAD_SQL = """
    SELECT c.id, c.document_id, d.title, d.path, c.page_number, c.section_heading, c.text
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE {condition}
    ORDER BY c.id
"""


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP_WORDS]


class _Index:
    def __init__(self, rows):
        from rank_bm25 import BM25Okapi

        self.rows = rows
        tokens = [tokenize(r[6]) for r in rows]
        self.words = [set(words) for words in tokens]
        self.bm25 = BM25Okapi([words or ["_empty_"] for words in tokens])


class Bm25Searcher:
    def __init__(self, conn):
        self.conn = conn
        self._indexes: dict[tuple | None, _Index] = {}

    def _index(self, filters: SearchFilters | None) -> _Index:
        key = (
            None
            if filters is None
            else (tuple(filters.source_types or ()), filters.date_from, filters.date_to, filters.path_prefix)
        )
        if key not in self._indexes:
            condition, params = build_filter_sql(filters)
            rows = self.conn.execute(_LOAD_SQL.format(condition=condition), params).fetchall()
            self._indexes[key] = _Index(rows)
        return self._indexes[key]

    def warm_up(self, filters: SearchFilters | None = None) -> None:
        self._index(filters)

    def search(self, query: str, k: int = 30, filters: SearchFilters | None = None) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be at least 1")
        terms = tokenize(query.replace("\x00", ""))
        index = self._index(filters)
        if not terms or not index.rows:
            return []
        scores = index.bm25.get_scores(terms)
        # A chunk is a match if it contains a query word. (The BM25 score itself can be zero or
        # negative for a word found in more than half of the chunks, so it is not used as the test.)
        wanted = set(terms)
        matching = [i for i in range(len(scores)) if index.words[i] & wanted]
        order = sorted(matching, key=lambda i: (-scores[i], index.rows[i][0]))
        results = []
        for i in order:
            if len(results) == k:
                break
            chunk = RetrievedChunk(*index.rows[i], score=float(scores[i]))
            chunk.keyword_rank = len(results) + 1
            chunk.keyword_score = chunk.score
            results.append(chunk)
        return results
