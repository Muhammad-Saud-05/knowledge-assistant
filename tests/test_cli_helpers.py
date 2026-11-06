import argparse
from datetime import date

import pytest

from src.ask import _found_by
from src.cli_filters import add_filter_arguments, filters_from_args, parse_date
from src.retrieval.models import RetrievedChunk
from src.search import format_result, rank_text, where


def make_parser():
    parser = argparse.ArgumentParser(prog="tool")
    parser.add_argument("query")
    add_filter_arguments(parser)
    return parser


def parse(*argv):
    parser = make_parser()
    return filters_from_args(parser, parser.parse_args(["query", *argv]))


# ------------------------------------------------------------------- dates


def test_valid_date_is_parsed():
    assert parse_date("2026-10-01") == date(2026, 10, 1)


@pytest.mark.parametrize(
    "text", ["yesterday", "2026-13-01", "2026-02-30", "26-10-01", "2026/10/01", "20261001", "2026-1-1", "", " 2026-10-01"]
)
def test_invalid_dates_give_a_friendly_message(text):
    with pytest.raises(argparse.ArgumentTypeError) as error:
        parse_date(text)
    message = str(error.value)
    assert "YYYY-MM-DD" in message and "2026-10-01" in message
    assert "Traceback" not in message and "ValueError" not in message


def test_bad_date_on_the_command_line_exits_with_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        make_parser().parse_args(["query", "--since", "next week"])
    assert exit_info.value.code == 2
    error_text = capsys.readouterr().err
    assert "--since" in error_text and "YYYY-MM-DD" in error_text and "Traceback" not in error_text


def test_since_after_until_is_a_friendly_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse("--since", "2026-11-01", "--until", "2026-10-01")
    assert exit_info.value.code == 2
    error_text = capsys.readouterr().err
    assert "--since" in error_text and "--until" in error_text and "date_from" not in error_text


# ------------------------------------------------------------ filter options


def test_no_options_means_no_filters():
    assert parse() is None


def test_options_become_filters():
    filters = parse("--type", "pdf", "--since", "2026-10-01", "--until", "2026-12-31", "--path-prefix", "sample_corpus/")
    assert filters.source_types == ["pdf"]
    assert (filters.date_from, filters.date_to) == (date(2026, 10, 1), date(2026, 12, 31))
    assert filters.path_prefix == "sample_corpus/"


def test_type_can_be_repeated():
    assert parse("--type", "pdf", "--type", "markdown").source_types == ["pdf", "markdown"]


def test_unknown_type_is_rejected(capsys):
    with pytest.raises(SystemExit):
        make_parser().parse_args(["query", "--type", "docx"])
    assert "invalid choice" in capsys.readouterr().err


# ------------------------------------------------------------ display helpers


def make_chunk(**fields):
    base = dict(chunk_id=1, document_id=1, title="Guide", path="g.pdf", page_number=5,
                section_heading="4. Retrieval", text="word " * 60, score=0.1)
    return RetrievedChunk(**{**base, **fields})


def test_where_shows_page_and_heading():
    assert where(make_chunk()) == "p. 5, 4. Retrieval"
    assert where(make_chunk(page_number=None)) == "4. Retrieval"
    assert where(make_chunk(section_heading=None)) == "p. 5"
    assert where(make_chunk(page_number=None, section_heading=None)) == "-"


def test_rank_text():
    assert rank_text(3) == "#3" and rank_text(None) == "-"


def test_result_line_shows_ranks_fused_score_and_a_short_preview():
    text = format_result(1, make_chunk(vector_rank=2, keyword_rank=None, fused_score=0.0164))
    assert "Guide | p. 5, 4. Retrieval" in text
    assert "vector #2 | keyword - | fused 0.0164" in text
    preview = text.splitlines()[-1].strip()
    assert len(preview) <= 100


def test_found_by_shows_only_the_methods_that_were_used():
    chunk = make_chunk(vector_rank=3, keyword_rank=None)
    assert _found_by(chunk, "vector") == "vector #3"
    assert _found_by(chunk, "keyword") == "keyword -"
    assert _found_by(chunk, "hybrid") == "vector #3, keyword -"
