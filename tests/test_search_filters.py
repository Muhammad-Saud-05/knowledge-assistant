from datetime import date

import pytest

from src.retrieval.filters import SearchFilters, build_filter_sql
from src.retrieval.keyword import keyword_search
from src.retrieval.vector import vector_search

# Four documents whose chunks all contain "alpha", so keyword search sees every one.
DOCS = [
    ("PDF Guide", "pdfs/a.pdf", "pdf", date(2026, 1, 10)),
    ("Notes B", "notes/b.md", "markdown", date(2026, 2, 1)),
    ("Notes C", "notes/c.md", "markdown", date(2026, 3, 1)),
    ("Undated", "misc/d.md", "markdown", None),
]


@pytest.fixture
def stored(conn, add_document):
    for title, path, source_type, doc_date in DOCS:
        add_document(
            conn, title, path, [(f"alpha text of {title}", None, "H")], source_type=source_type, doc_date=doc_date
        )


def vector(conn, embedder, filters):
    return vector_search(conn, embedder, "alpha", k=10, filters=filters)


def keyword(conn, embedder, filters):
    return keyword_search(conn, "alpha", k=10, filters=filters)


@pytest.fixture(params=[vector, keyword], ids=["vector", "keyword"])
def paths(request, conn, embedder, stored):
    """Paths found by the search under test (vector, then keyword) for some filters."""

    def run(**filter_fields):
        filters = SearchFilters(**filter_fields) if filter_fields else None
        return sorted(r.path for r in request.param(conn, embedder, filters))

    return run


# --------------------------------------------------------------- source type


def test_no_filter_returns_everything(paths):
    assert paths() == ["misc/d.md", "notes/b.md", "notes/c.md", "pdfs/a.pdf"]


def test_filter_by_pdf(paths):
    assert paths(source_types=["pdf"]) == ["pdfs/a.pdf"]


def test_filter_by_markdown(paths):
    assert paths(source_types=["markdown"]) == ["misc/d.md", "notes/b.md", "notes/c.md"]


def test_filter_by_both_types(paths):
    assert len(paths(source_types=["pdf", "markdown"])) == 4


# ---------------------------------------------------------------- date range


def test_date_range_edges_are_inclusive(paths):
    assert paths(date_from=date(2026, 2, 1), date_to=date(2026, 3, 1)) == ["notes/b.md", "notes/c.md"]


def test_date_from_is_inclusive_and_open_ended(paths):
    assert paths(date_from=date(2026, 3, 1)) == ["notes/c.md"]
    assert paths(date_from=date(2026, 2, 2)) == ["notes/c.md"]
    assert paths(date_from=date(2026, 1, 10)) == ["notes/b.md", "notes/c.md", "pdfs/a.pdf"]


def test_date_to_is_inclusive_and_open_ended(paths):
    assert paths(date_to=date(2026, 1, 10)) == ["pdfs/a.pdf"]
    assert paths(date_to=date(2026, 1, 9)) == []
    assert paths(date_to=date(2026, 2, 1)) == ["notes/b.md", "pdfs/a.pdf"]


def test_single_day_range(paths):
    assert paths(date_from=date(2026, 2, 1), date_to=date(2026, 2, 1)) == ["notes/b.md"]


def test_undated_documents_never_match_a_date_filter(paths):
    assert "misc/d.md" not in paths(date_from=date(2000, 1, 1), date_to=date(2100, 1, 1))


# --------------------------------------------------------------- path prefix


def test_path_prefix(paths):
    assert paths(path_prefix="notes/") == ["notes/b.md", "notes/c.md"]
    assert paths(path_prefix="notes/b") == ["notes/b.md"]
    assert paths(path_prefix="pdfs/a.pdf") == ["pdfs/a.pdf"]


def test_path_prefix_wildcard_characters_are_literal(paths):
    assert paths(path_prefix="notes/%") == []
    assert paths(path_prefix="%") == []
    assert paths(path_prefix="_otes/") == []


# -------------------------------------------------------------- combinations


def test_combined_filters(paths):
    assert paths(source_types=["markdown"], date_from=date(2026, 2, 15)) == ["notes/c.md"]
    assert paths(source_types=["markdown"], path_prefix="notes/", date_to=date(2026, 2, 1)) == ["notes/b.md"]
    assert paths(source_types=["pdf"], path_prefix="pdfs/", date_from=date(2026, 1, 1), date_to=date(2026, 1, 31)) == [
        "pdfs/a.pdf"
    ]


def test_filters_that_match_nothing_return_an_empty_list(paths):
    assert paths(source_types=["pdf"], date_from=date(2026, 2, 1)) == []
    assert paths(path_prefix="nowhere/") == []
    assert paths(source_types=["pdf"], path_prefix="notes/") == []


def test_hostile_path_prefix_is_just_text(conn, embedder, stored):
    filters = SearchFilters(path_prefix="'; DROP TABLE documents; --")
    assert vector(conn, embedder, filters) == [] and keyword(conn, embedder, filters) == []
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 4


# ------------------------------------------------- the filter object itself


def test_no_filters_means_no_condition():
    assert build_filter_sql(None) == ("TRUE", {})
    assert build_filter_sql(SearchFilters()) == ("TRUE", {})


def test_values_are_parameters_never_part_of_the_sql_text():
    filters = SearchFilters(["pdf"], date(2026, 1, 1), date(2026, 2, 2), "it's/'; --")
    condition, params = build_filter_sql(filters)
    for value in ("pdf", "2026", "it's", "--"):
        assert value not in condition
    assert params == {
        "filter_source_types": ["pdf"],
        "filter_date_from": date(2026, 1, 1),
        "filter_date_to": date(2026, 2, 2),
        "filter_path_prefix": "it's/'; --",
    }
    assert condition.count("%(") == 4


@pytest.mark.parametrize(
    "bad",
    [
        SearchFilters(source_types=[]),
        SearchFilters(source_types=["docx"]),
        SearchFilters(date_from=date(2026, 5, 2), date_to=date(2026, 5, 1)),
    ],
)
def test_invalid_filters_are_rejected(bad):
    with pytest.raises(ValueError):
        build_filter_sql(bad)
