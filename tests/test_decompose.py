import json

import pytest

from eval import run_eval
from eval.questions import Expected, Question
from eval.report import (
    coverage_paired_table,
    decomposition_section,
    render_report,
    results_from_json,
    results_to_json,
)
from eval.run_eval import DECOMPOSE_MODES, describe_mode, parse_modes, run_questions
from eval.validate_questions import DocInfo
from src.llm.ollama import OllamaLLM
from src.retrieval.decompose import (
    LLM_SYSTEM_PROMPT,
    LLMDecomposer,
    RuleDecomposer,
    merge_round_robin,
    parse_sub_questions,
    search_decomposed,
    split_by_rule,
)
from src.retrieval.models import RetrievedChunk
from src.retrieval.rerank import Reranker

# ------------------------------------------------------------------ the rule


def test_two_question_marks_split():
    assert split_by_rule("What is a token bucket? How does it refill?") == ["What is a token bucket?", "How does it refill?"]


def test_a_semicolon_splits():
    assert split_by_rule("Explain the HNSW index; list its two main settings") == [
        "Explain the HNSW index",
        "list its two main settings",
    ]


def test_and_before_a_question_word_splits_and_the_question_mark_is_kept():
    parts = split_by_rule("Which status code should I return, and how can I run the check on every request?")
    assert parts == ["Which status code should I return?", "how can I run the check on every request?"]
    assert split_by_rule("Which URL do I call and what must I install first?")[1] == "what must I install first?"


def test_and_between_nouns_does_not_split():
    for q in (
        "Which status code and header should my API return?",
        "How do I use FastAPI and Ollama together?",
        "Why do people give up on flashcard apps after a few weeks?",
    ):
        assert split_by_rule(q) == [q]


def test_short_parts_are_dropped_and_a_single_part_means_no_split():
    assert split_by_rule("What is HNSW? Why?") == ["What is HNSW? Why?"]
    assert split_by_rule("What is a token bucket and why?") == ["What is a token bucket and why?"]


def test_at_most_three_parts_are_used():
    q = "What is one thing? What is two things? What is three things? What is four things?"
    assert split_by_rule(q) == ["What is one thing?", "What is two things?", "What is three things?"]


def test_rule_decomposer_always_returns_a_list():
    assert RuleDecomposer().decompose("What is a bucket?") == ["What is a bucket?"]


# -------------------------------------------------------------- the LLM reply


def reply(*items):
    return json.dumps({"sub_questions": list(items)})


def test_a_valid_reply_gives_the_sub_questions():
    assert parse_sub_questions(reply("What is A?", "How is B  used?"), "orig") == ["What is A?", "How is B used?"]
    assert len(parse_sub_questions(reply("a b c", "d e f", "g h i"), "orig")) == 3


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        "",
        "[]",
        "null",
        '{"other": ["a", "b"]}',
        '{"sub_questions": "a question"}',
        '{"sub_questions": []}',
        reply("one", "two", "three", "four"),  # more than 3
        reply("fine question", ""),
        reply("fine question", "   "),
        json.dumps({"sub_questions": ["fine question", 3]}),
        reply("fine question", "x" * 301),
    ],
)
def test_any_invalid_reply_falls_back_to_the_original_question(bad):
    assert parse_sub_questions(bad, "the original") == ["the original"]


def test_a_single_sub_question_means_use_the_original_unchanged():
    assert parse_sub_questions(reply("a reworded version"), "the original") == ["the original"]


class FakeLLM:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, []

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.text


def test_llm_decomposer_sends_the_pre_registered_prompt_and_the_question():
    llm = FakeLLM(reply("What is A?", "What is B?"))
    assert LLMDecomposer(llm).decompose("A and B?") == ["What is A?", "What is B?"]
    assert llm.calls == [(LLM_SYSTEM_PROMPT, "A and B?")]


def test_llm_decomposer_falls_back_on_garbage_and_on_any_error():
    assert LLMDecomposer(FakeLLM("sorry, I cannot")).decompose("the question") == ["the question"]
    for error in (RuntimeError("boom"), ConnectionError("down"), TimeoutError()):
        assert LLMDecomposer(FakeLLM(error=error)).decompose("the question") == ["the question"]


def test_the_ollama_adapter_asks_for_json_only_when_told_to():
    assert "format" not in OllamaLLM(host="http://x", json_output=False)._payload("s", "u", False)
    payload = OllamaLLM(host="http://x", json_output=True, temperature=0)._payload("s", "u", False)
    assert payload["format"] == "json" and payload["options"]["temperature"] == 0


# ------------------------------------------------------------------ the merge


def c(cid):
    return RetrievedChunk(cid, 1, "T", f"d{cid}.md", None, None, f"text {cid}", 0.0)


def ids(chunks):
    return [x.chunk_id for x in chunks]


def test_round_robin_takes_the_best_of_each_list_in_turn():
    assert ids(merge_round_robin([[c(1), c(2), c(3)], [c(4), c(5)]], 10)) == [1, 4, 2, 5, 3]


def test_round_robin_skips_duplicates_and_respects_the_limit():
    assert ids(merge_round_robin([[c(1), c(2)], [c(1), c(3)], [c(2), c(1), c(4)]], 10)) == [1, 3, 2, 4]  # list 2 skips its duplicate 1 and gives 3
    assert ids(merge_round_robin([[c(1), c(2), c(3)], [c(4), c(5), c(6)]], 4)) == [1, 4, 2, 5]
    assert ids(merge_round_robin([[c(1), c(2), c(3)], [c(4), c(5), c(6)]], 3)) == [1, 4, 2]


def test_round_robin_handles_empty_and_single_lists():
    assert merge_round_robin([], 5) == []
    assert merge_round_robin([[], []], 5) == []
    assert ids(merge_round_robin([[c(3), c(1), c(2)]], 2)) == [3, 1]
    assert ids(merge_round_robin([[], [c(7)]], 5)) == [7]


# ----------------------------------------------- searching with a decomposer

DOCS = [
    ("The Ollama chat endpoint is called at http://localhost:11434/api/chat", None, "Ollama chat"),
    ("Declare the endpoint function with plain def so FastAPI runs it in a thread pool", None, "Plain def"),
    ("Fixed window counters reset at the end of every window", None, "Fixed window"),
    ("Flashcards support spaced repetition and active recall", None, "Cards"),
]


class ScriptedDecomposer:
    def __init__(self, parts):
        self.parts = parts

    def decompose(self, question):
        return self.parts if self.parts else [question]


def keyword_reranker(queries_seen):
    """Scores a passage by how many of the query's words it contains; records each query."""

    def score(pairs):
        queries_seen.append(pairs[0][0])
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in pairs]

    return Reranker(score_fn=score)


@pytest.fixture
def stored(conn, add_document):
    add_document(conn, "Notes", "notes/all.md", DOCS)


def test_an_unsplit_question_is_exactly_configuration_c(conn, embedder, stored):
    from src.retrieval.hybrid import search

    rr = keyword_reranker([])
    expected = search(conn, embedder, "ollama chat endpoint url", k=3, mode="vector+rerank", reranker=rr, rerank_candidates=30)
    result = search_decomposed(conn, embedder, "ollama chat endpoint url", 3, rr, ScriptedDecomposer(None))
    assert ids(result.chunks) == ids(expected)
    assert result.sub_questions == ["ollama chat endpoint url"] and result.decomposition_seconds >= 0


def test_each_sub_question_is_reranked_against_itself_and_merged_round_robin(conn, embedder, stored):
    seen = []
    parts = ["ollama chat endpoint url", "declare endpoint function plain def"]
    result = search_decomposed(conn, embedder, "both asks together", 4, keyword_reranker(seen), ScriptedDecomposer(parts))
    assert result.sub_questions == parts
    assert set(seen) == set(parts)  # never the whole question
    headings = [x.section_heading for x in result.chunks]
    assert headings[0] == "Ollama chat" and headings[1] == "Plain def"  # the best chunk of each list first
    assert len(set(ids(result.chunks))) == len(result.chunks) == 4


# ------------------------------------------------------- the runner and report

A, B = "docs/a.md", "docs/b.md"
CORPUS = {A: DocInfo("markdown", None, []), B: DocInfo("markdown", None, [])}
QUESTIONS = [
    Question("q01", "exact_term", "plain question", (Expected(A, ("alpha phrase",)),), False, difficulty="high_overlap", lexical_overlap=0.5),
    Question("q02", "multi_doc", "ask one and ask two", (Expected(A, ("alpha phrase",)), Expected(B, ("beta phrase",))), False, difficulty="high_overlap", lexical_overlap=0.5),
]


def chunk(cid, path, text):
    return RetrievedChunk(cid, 1, "T", path, None, None, text, 1.0)


class FakeSearch:
    """Mode 'base' returns only A for q02; mode 'split' finds both and reports sub-questions."""

    def __init__(self):
        self.last_decomposition = None

    def __call__(self, q, mode):
        self.last_decomposition = None
        both = [chunk(1, A, "alpha phrase"), chunk(2, B, "beta phrase")]
        if mode == "split":
            self.last_decomposition = (["ask one?", "ask two?"] if q.id == "q02" else [q.question], 0.5)
            return both
        return both[:1]


def test_the_runner_records_sub_questions_and_decomposition_time():
    results = run_questions(QUESTIONS, ["base", "split"], FakeSearch(), CORPUS)
    assert results["base"]["q02"].sub_questions is None
    assert results["split"]["q02"].sub_questions == ["ask one?", "ask two?"]
    assert results["split"]["q02"].decomposition_seconds == 0.5
    assert results["split"]["q01"].sub_questions == ["plain question"]


def test_sub_questions_survive_the_json_round_trip():
    results = run_questions(QUESTIONS, ["base", "split"], FakeSearch(), CORPUS)
    data = json.loads(json.dumps(results_to_json(QUESTIONS, results, {"x": 1})))
    again = results_from_json(data)
    assert again["split"]["q02"].sub_questions == ["ask one?", "ask two?"]
    assert again["split"]["q02"].decomposition_seconds == 0.5
    assert again["base"]["q02"].sub_questions is None


def test_the_report_lists_sub_questions_and_compares_coverage_with_the_baseline():
    results = run_questions(QUESTIONS, ["base", "split"], FakeSearch(), CORPUS)
    text = decomposition_section(QUESTIONS, results, ["base", "split"])
    assert "split | 1/2 |" in text and "`ask one?` | `ask two?`" in text
    assert "base" not in text.split("| mode |")[1].split("\n")[2]  # only splitting modes are in the table
    table = coverage_paired_table(QUESTIONS, results, [("split", "base")])
    assert "split - base" in table and "+0.500" in table  # coverage 1.0 against 0.5 on the one multi_doc question
    full = render_report(QUESTIONS, results, ["base", "split"], 5, baseline="base")
    assert "## Question decomposition" in full and "multi_doc coverage@5" in full


def test_decomposing_modes_are_known_to_the_runner():
    assert describe_mode("rerank-split-rule") == ("vector", True, "rule")
    assert describe_mode("rerank-split-llm") == ("vector", True, "llm")
    assert describe_mode("vector+rerank") == ("vector", True, None)
    assert describe_mode("hybrid") == ("hybrid", False, None)
    assert parse_modes("vector+rerank,rerank-split-rule,rerank-split-llm") == ["vector+rerank", "rerank-split-rule", "rerank-split-llm"]
    assert set(DECOMPOSE_MODES.values()) == {"rule", "llm"}


def test_build_search_fn_dispatches_decomposing_modes(monkeypatch):
    seen = []

    def fake_decomposed(conn, embedder, question, k, reranker, decomposer, filters, rerank_candidates):
        from src.retrieval.decompose import Decomposed

        seen.append((question, k, decomposer.name, rerank_candidates))
        return Decomposed([], ["a", "b"], 0.25)

    monkeypatch.setattr("src.retrieval.decompose.search_decomposed", fake_decomposed)
    fn = run_eval.build_search_fn(None, None, 30, reranker="RR", rerank_candidates=20, decomposers={"rule": RuleDecomposer()})
    assert fn(QUESTIONS[1], "rerank-split-rule") == []
    assert seen == [("ask one and ask two", run_eval.DEPTH, "rule", 20)]
    assert fn.last_decomposition == (["a", "b"], 0.25)
