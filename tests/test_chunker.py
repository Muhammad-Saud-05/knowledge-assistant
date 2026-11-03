import re

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.models import ParsedDocument, Section


def make_doc(sections, source_type="markdown"):
    return ParsedDocument(
        source_type=source_type,
        path="x",
        title="t",
        content_hash="h",
        doc_date=None,
        sections=sections,
    )


def sentences(prefix, count):
    """`count` unique 7-word sentences, e.g. "ALPHA 12 talks about the same topic."."""
    return [f"{prefix} {i} talks about the same topic." for i in range(count)]


def paragraphs(prefix, n_paragraphs, per_paragraph=5):
    out = []
    for p in range(n_paragraphs):
        out.append(" ".join(sentences(f"{prefix}p{p}s", per_paragraph)))
    return "\n\n".join(out)


def words(text):
    return len(text.split())


# ------------------------------------------------------------------- size


def test_chunks_do_not_exceed_max_words():
    doc = make_doc([Section("H", None, paragraphs("A", 30))])
    chunks = chunk_document(doc, max_words=100, overlap_words=20)
    assert len(chunks) > 3
    assert max(words(c.text) for c in chunks) <= 100


def test_long_paragraph_without_breaks_is_split_on_sentences():
    text = " ".join(sentences("Long", 120))  # one 840-word paragraph
    chunks = chunk_document(make_doc([Section("H", None, text)]), 100, 20)
    assert len(chunks) > 5
    assert max(words(c.text) for c in chunks) <= 100
    # Chunks start at sentence starts, so no sentence is cut in half.
    assert all(c.text.startswith("Long ") for c in chunks)


def test_single_endless_sentence_falls_back_to_word_windows():
    text = " ".join(f"w{i}" for i in range(1000))  # no punctuation at all
    chunks = chunk_document(make_doc([Section("H", None, text)]), 100, 20)
    assert max(words(c.text) for c in chunks) <= 100
    seen = {w for c in chunks for w in c.text.split()}
    assert seen == {f"w{i}" for i in range(1000)}


def test_short_section_stays_one_chunk():
    chunks = chunk_document(make_doc([Section("H", None, "One short section of text here.")]))
    assert len(chunks) == 1


def test_invalid_sizes_are_rejected():
    doc = make_doc([Section("H", None, "text")])
    with pytest.raises(ValueError):
        chunk_document(doc, max_words=40, overlap_words=40)


# ---------------------------------------------------------------- overlap


def shared_prefix_length(a, b, limit=200):
    """Largest k where the last k words of `a` equal the first k words of `b`."""
    aw, bw = a.split(), b.split()
    for k in range(min(limit, len(aw), len(bw)), 0, -1):
        if aw[-k:] == bw[:k]:
            return k
    return 0


def test_consecutive_chunks_overlap_by_about_overlap_words():
    doc = make_doc([Section("H", None, paragraphs("A", 30))])
    chunks = chunk_document(doc, max_words=100, overlap_words=40)
    assert len(chunks) > 3
    for before, after in zip(chunks, chunks[1:]):
        overlap = shared_prefix_length(before.text, after.text)
        assert 20 <= overlap <= 40


def test_overlap_can_be_turned_off():
    doc = make_doc([Section("H", None, paragraphs("A", 30))])
    chunks = chunk_document(doc, max_words=100, overlap_words=0)
    for before, after in zip(chunks, chunks[1:]):
        assert shared_prefix_length(before.text, after.text) == 0


def test_no_text_is_lost_between_chunks():
    source = sentences("Keep", 200)
    doc = make_doc([Section("H", None, " ".join(source))])
    chunks = chunk_document(doc, max_words=100, overlap_words=20)
    combined = " ".join(c.text for c in chunks)
    for sentence in source:
        assert sentence in combined


def test_no_text_is_lost_with_paragraphs():
    text = paragraphs("Para", 25)
    chunks = chunk_document(make_doc([Section("H", None, text)]), 100, 20)
    combined = " ".join(c.text for c in chunks)
    found = set(re.findall(r"Parap\d+s\d+", combined))
    assert found == set(re.findall(r"Parap\d+s\d+", text))


def test_fenced_code_block_is_kept_whole_when_it_fits():
    code = "```python\nx = 1\n\ny = 2\n```"  # contains a blank line
    text = paragraphs("A", 12) + "\n\n" + code + "\n\n" + paragraphs("B", 12)
    chunks = chunk_document(make_doc([Section("H", None, text)]), 100, 20)
    holders = [c for c in chunks if "x = 1" in c.text]
    assert holders and all("y = 2" in c.text for c in holders)


# ------------------------------------------------------------- boundaries


def test_chunks_never_cross_section_or_page_boundaries():
    sections = [
        Section("Intro", 1, paragraphs("ALPHA", 12)),
        Section("Methods", 2, paragraphs("BETA", 12)),
        Section("Results", 3, paragraphs("GAMMA", 12)),
    ]
    chunks = chunk_document(make_doc(sections, "pdf"), max_words=100, overlap_words=20)
    expected = {1: "ALPHA", 2: "BETA", 3: "GAMMA"}
    headings = {1: "Intro", 2: "Methods", 3: "Results"}
    assert {c.page_number for c in chunks} == {1, 2, 3}
    for chunk in chunks:
        markers = set(re.findall(r"ALPHA|BETA|GAMMA", chunk.text))
        assert markers == {expected[chunk.page_number]}
        assert chunk.section_heading == headings[chunk.page_number]


def test_sections_with_same_page_but_different_headings_stay_separate():
    sections = [
        Section("One", None, paragraphs("ONE", 8)),
        Section("Two", None, paragraphs("TWO", 8)),
    ]
    for chunk in chunk_document(make_doc(sections), max_words=100, overlap_words=20):
        assert ("ONE" in chunk.text) != ("TWO" in chunk.text)


def test_chunk_index_runs_from_zero_across_the_whole_document():
    sections = [Section("A", 1, paragraphs("A", 12)), Section("B", 2, paragraphs("B", 12))]
    chunks = chunk_document(make_doc(sections, "pdf"), max_words=100, overlap_words=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# --------------------------------------------------- empty and tiny sections


def test_empty_sections_are_skipped():
    sections = [
        Section("Empty", 1, ""),
        Section("Spaces", 2, "  \n\n  "),
        Section("Real", 3, "Some real content in this section."),
    ]
    chunks = chunk_document(make_doc(sections, "pdf"))
    assert [(c.section_heading, c.page_number) for c in chunks] == [("Real", 3)]
    assert chunks[0].chunk_index == 0


def test_tiny_section_merges_into_previous_with_same_context():
    sections = [
        Section("Top", None, "Main body text. " * 20),
        Section("Top > Sub", None, "Tiny sub text."),
    ]
    chunks = chunk_document(make_doc(sections))
    assert len(chunks) == 1
    assert chunks[0].section_heading == "Top"
    assert "Tiny sub text." in chunks[0].text
    assert "Sub" in chunks[0].text  # the sub-heading name is kept as context


def test_tiny_section_is_not_merged_across_pages():
    sections = [Section("Top", 1, "Main body text. " * 20), Section("Top", 2, "Tiny page.")]
    chunks = chunk_document(make_doc(sections, "pdf"))
    assert [(c.page_number, "Tiny page." in c.text) for c in chunks] == [(1, False), (2, True)]


def test_tiny_section_is_not_merged_into_unrelated_heading():
    sections = [Section("First", None, "Body words. " * 20), Section("Second", None, "Tiny.")]
    chunks = chunk_document(make_doc(sections))
    assert [c.section_heading for c in chunks] == ["First", "Second"]


def test_first_tiny_section_stays_its_own_chunk():
    chunks = chunk_document(make_doc([Section("Only", 1, "Tiny cover page.")], "pdf"))
    assert len(chunks) == 1 and chunks[0].page_number == 1
