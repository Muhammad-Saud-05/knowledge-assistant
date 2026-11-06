import pytest

from src.retrieval.keyword import keyword_search

RATE = (
    "Rate limiting caps requests. When a request is rejected the service returns HTTP status "
    "429 and a Retry-After header so well-behaved clients know when to try again.",
    None,
    "Why rate limiting exists",
)
BUCKET = ("A token bucket refills tokens at a steady rate and allows short bursts.", None, "Token bucket")
GIT = ("Undo the last commit with git reset --soft HEAD~1 but keep your changes.", None, "Fixing mistakes")
CARDS = ("Flashcards help with spaced repetition and active recall of facts.", None, "Study habits")


@pytest.fixture
def stored(conn, add_document):
    add_document(conn, "Notes", "notes/all.md", [RATE, BUCKET, GIT, CARDS])


def headings(results):
    return [r.section_heading for r in results]


# ------------------------------------------------------------- finding things


@pytest.mark.parametrize("query", ["Retry-After", "429", "HTTP 429 Retry-After", "retry-after header"])
def test_exact_terms_find_their_chunk(conn, stored, query):
    results = keyword_search(conn, query)
    assert results and results[0].section_heading == "Why rate limiting exists"


def test_exact_command_finds_its_chunk(conn, stored):
    assert keyword_search(conn, "git reset --soft HEAD~1")[0].section_heading == "Fixing mistakes"


def test_long_question_with_a_word_that_matches_nothing_still_returns_results(conn, stored):
    # "banana" and "quickly" appear nowhere; an AND query would find nothing.
    question = "how do I make the service tell clients when to try again after a rejected request banana quickly"
    results = keyword_search(conn, question)
    assert results and results[0].section_heading == "Why rate limiting exists"


def test_words_from_different_chunks_all_match(conn, stored):
    found = headings(keyword_search(conn, "token flashcards"))
    assert set(found) == {"Token bucket", "Study habits"}  # OR semantics


def test_stemming_matches_other_word_forms(conn, stored):
    assert keyword_search(conn, "refilling")[0].section_heading == "Token bucket"


def test_chunk_with_more_of_the_words_ranks_higher(conn, stored):
    results = keyword_search(conn, "token bucket refills rate")
    assert results[0].section_heading == "Token bucket"


# ------------------------------------------------------- results and ranking


def test_results_have_ranks_scores_and_details(conn, stored):
    results = keyword_search(conn, "rate request token spaced")
    assert [r.keyword_rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True) and scores[0] > 0
    for r in results:
        assert r.score == r.keyword_score
        assert r.vector_rank is None and r.fused_score is None
        assert (r.title, r.path) == ("Notes", "notes/all.md")


def test_k_limits_the_number_of_results(conn, stored):
    assert len(keyword_search(conn, "rate request token spaced git", k=2)) == 2


def test_invalid_k_is_rejected(conn):
    with pytest.raises(ValueError):
        keyword_search(conn, "anything", k=0)


# ------------------------------------------------------ nothing usable in query


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "the and of a to", "!!! ??? ...", "😀🚀"])
def test_queries_without_usable_terms_return_an_empty_list(conn, stored, query):
    assert keyword_search(conn, query) == []


def test_query_whose_words_match_nothing_returns_an_empty_list(conn, stored):
    assert keyword_search(conn, "zyxwvut qwertyuiop") == []


# ---------------------------------------------------------- hostile / odd input


@pytest.mark.parametrize(
    "query",
    [
        '"unclosed quote',
        "'single quote",
        "rate & limiting",
        "rate | limiting",
        "!rate",
        "rate:*",
        ":*",
        "(rate",
        "rate))",
        "((()))",
        "rate <-> limiting",
        "a & b | !c :* ( ) <->",
        "rate \\ limiting",
        "'; DROP TABLE chunks; --",
        "token 😀 bucket",
        "café naïve",
        "rate\x00limiting",
        "http://example.com/a&b?x=1",
        "AT&T",
        "x" * 20000,
    ],
)
def test_any_input_is_accepted_without_errors(conn, stored, query):
    assert isinstance(keyword_search(conn, query), list)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 4  # nothing was dropped


def test_operator_characters_are_just_words(conn, stored):
    # "!token" must not mean "NOT token", and "&" must not mean AND.
    assert keyword_search(conn, "!token")[0].section_heading == "Token bucket"
    assert "Study habits" in headings(keyword_search(conn, "flashcards & zzzzzz"))
