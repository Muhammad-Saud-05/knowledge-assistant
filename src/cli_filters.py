"""Command-line options for search filters, shared by the search and ask commands."""

import argparse
import re
from datetime import date

from .retrieval.filters import VALID_SOURCE_TYPES, SearchFilters

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_date(text: str) -> date:
    """Parse YYYY-MM-DD, with a friendly message when it is wrong."""
    try:
        if not _DATE_RE.match(text):
            raise ValueError
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a valid date. Use YYYY-MM-DD, for example 2026-10-01."
        ) from None


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("filters (only search matching documents)")
    group.add_argument(
        "--type",
        dest="source_types",
        action="append",
        choices=VALID_SOURCE_TYPES,
        help="only this kind of document (repeat to allow both)",
    )
    group.add_argument("--since", type=parse_date, metavar="YYYY-MM-DD", help="document date on or after")
    group.add_argument("--until", type=parse_date, metavar="YYYY-MM-DD", help="document date on or before")
    group.add_argument("--path-prefix", metavar="PREFIX", help="only documents whose path starts with this")


def filters_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> SearchFilters | None:
    """Build SearchFilters from parsed arguments (None if no filter was given)."""
    filters = SearchFilters(
        source_types=args.source_types,
        date_from=args.since,
        date_to=args.until,
        path_prefix=args.path_prefix,
    )
    if args.since and args.until and args.since > args.until:
        parser.error(f"--since ({args.since}) is after --until ({args.until}), so nothing can match.")
    try:
        filters.validate()
    except ValueError as error:
        parser.error(str(error))
    if all(v is None for v in (filters.source_types, filters.date_from, filters.date_to, filters.path_prefix)):
        return None
    return filters
