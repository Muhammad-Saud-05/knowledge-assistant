"""Optional restrictions on which documents a search may look at."""

from dataclasses import dataclass
from datetime import date

VALID_SOURCE_TYPES = ("pdf", "markdown")


@dataclass
class SearchFilters:
    source_types: list[str] | None = None  # "pdf" and/or "markdown"
    date_from: date | None = None  # inclusive, compared with documents.doc_date
    date_to: date | None = None  # inclusive
    path_prefix: str | None = None  # documents.path must start with this

    def validate(self) -> None:
        if self.source_types is not None:
            if not self.source_types:
                raise ValueError("source_types must not be empty (use None for no filter)")
            unknown = [t for t in self.source_types if t not in VALID_SOURCE_TYPES]
            if unknown:
                raise ValueError(
                    f"unknown source type(s) {unknown}; use {' or '.join(VALID_SOURCE_TYPES)}"
                )
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError(f"date_from ({self.date_from}) is after date_to ({self.date_to})")


def build_filter_sql(filters: SearchFilters | None, documents_alias: str = "d") -> tuple[str, dict]:
    """Turn filters into (SQL condition, query parameters).

    The condition is made only of fixed text and %(name)s placeholders; the
    values travel separately as parameters, so they are never part of the SQL
    text. With no filters the condition is "TRUE", so it can always follow WHERE.
    A document with no doc_date never matches a date filter.
    """
    if filters is None:
        return "TRUE", {}
    filters.validate()

    conditions: list[str] = []
    params: dict = {}
    if filters.source_types is not None:
        conditions.append(f"{documents_alias}.source_type = ANY(%(filter_source_types)s)")
        params["filter_source_types"] = list(filters.source_types)
    if filters.date_from is not None:
        conditions.append(f"{documents_alias}.doc_date >= %(filter_date_from)s")
        params["filter_date_from"] = filters.date_from
    if filters.date_to is not None:
        conditions.append(f"{documents_alias}.doc_date <= %(filter_date_to)s")
        params["filter_date_to"] = filters.date_to
    if filters.path_prefix:
        # starts_with() instead of LIKE: "%" and "_" in a path are not wildcards.
        conditions.append(f"starts_with({documents_alias}.path, %(filter_path_prefix)s)")
        params["filter_path_prefix"] = filters.path_prefix
    return (" AND ".join(conditions) if conditions else "TRUE"), params
