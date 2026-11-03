import hashlib
import os
from datetime import date, datetime

import pymupdf
import pytest

from src.ingestion.parsers import parse_file, parse_markdown, parse_pdf


# ---------------------------------------------------------------- helpers


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def make_pdf(path, bodies, header="Acme Handbook", title=None):
    """Write a small PDF: a repeated header, a body per page and a "Page N" footer."""
    pdf = pymupdf.open()
    for number, body in enumerate(bodies, start=1):
        page = pdf.new_page()
        page.insert_text((72, 40), header, fontsize=10)
        y = 100
        for line in body.split("\n"):
            if line:
                page.insert_text((72, y), line, fontsize=11)
            y += 14 if line else 20  # a blank line leaves a gap, so a new block starts
        page.insert_text((72, 800), f"Page {number}", fontsize=10)
    if title:
        pdf.set_metadata({"title": title})
    pdf.save(path)
    pdf.close()
    return path


MARKDOWN = """\
---
title: My Notes
date: 2025-03-04
---

# Top Heading

Intro text.

## Rate Limiting

Body one.

### Token Bucket

Bucket body.

```python
# this is a comment, not a heading
x = 1
```

After the code.

## Other Topic

Last words.
"""


# --------------------------------------------------------------- markdown


def test_markdown_headings_become_nested_section_headings(tmp_path):
    doc = parse_markdown(write(tmp_path, "notes.md", MARKDOWN))
    assert [s.heading for s in doc.sections] == [
        "Top Heading",
        "Top Heading > Rate Limiting",
        "Top Heading > Rate Limiting > Token Bucket",
        "Top Heading > Other Topic",
    ]
    assert doc.source_type == "markdown"
    assert all(s.page_number is None for s in doc.sections)


def test_hash_lines_inside_code_fences_are_not_headings(tmp_path):
    doc = parse_markdown(write(tmp_path, "notes.md", MARKDOWN))
    headings = [s.heading for s in doc.sections]
    assert not any("comment" in h for h in headings)
    bucket = next(s for s in doc.sections if s.heading.endswith("Token Bucket"))
    assert "# this is a comment, not a heading" in bucket.text  # code is kept
    assert "After the code." in bucket.text


def test_tilde_fences_and_longer_fences_are_respected(tmp_path):
    text = "# Title\n\n~~~\n# not a heading\n~~~\n\n````\n```\n# still code\n```\n````\n\n## Real\n\nx\n"
    doc = parse_markdown(write(tmp_path, "f.md", text))
    assert [s.heading for s in doc.sections] == ["Title", "Title > Real"]
    assert "# still code" in doc.sections[0].text


def test_front_matter_title_and_date_are_used_and_removed(tmp_path):
    doc = parse_markdown(write(tmp_path, "notes.md", MARKDOWN))
    assert doc.title == "My Notes"
    assert doc.doc_date == date(2025, 3, 4)
    all_text = "\n".join(s.text for s in doc.sections)
    assert "title:" not in all_text
    assert "date:" not in all_text
    assert "---" not in all_text


def test_title_falls_back_to_first_h1_then_filename(tmp_path):
    with_h1 = parse_markdown(write(tmp_path, "a.md", "## Sub\n\n# Real Title\n\ntext\n"))
    assert with_h1.title == "Real Title"
    no_h1 = parse_markdown(write(tmp_path, "my-file.md", "just text\n"))
    assert no_h1.title == "my-file"


def test_date_falls_back_to_file_modified_date(tmp_path):
    path = write(tmp_path, "a.md", "# T\n\ntext\n")
    stamp = datetime(2020, 1, 2, 12, 0).timestamp()
    os.utime(path, (stamp, stamp))
    assert parse_markdown(path).doc_date == date(2020, 1, 2)


def test_unclosed_front_matter_marker_is_treated_as_text(tmp_path):
    doc = parse_markdown(write(tmp_path, "a.md", "---\n\n# Title\n\nbody\n"))
    assert doc.title == "Title"


def test_level_four_headings_are_not_split_points(tmp_path):
    doc = parse_markdown(write(tmp_path, "a.md", "# T\n\n#### deep\n\ntext\n"))
    assert len(doc.sections) == 1
    assert "#### deep" in doc.sections[0].text


# -------------------------------------------------------------------- pdf


def test_pdf_page_numbers_are_preserved(tmp_path):
    bodies = [f"Unique content for page {n}." for n in range(1, 5)]
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", bodies))
    assert doc.source_type == "pdf"
    assert [s.page_number for s in doc.sections] == [1, 2, 3, 4]
    for number, section in enumerate(doc.sections, start=1):
        assert f"Unique content for page {number}." in section.text


def test_repeated_headers_and_footers_are_stripped(tmp_path):
    bodies = [f"Unique content for page {n}." for n in range(1, 6)]
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", bodies, header="Acme Handbook"))
    for section in doc.sections:
        assert "Acme Handbook" not in section.text
        assert "Page " not in section.text  # "Page 1", "Page 2"... are one footer


def test_repeated_lines_in_the_page_body_are_not_treated_as_headers(tmp_path):
    # Only margin text can be a header/footer, even if a body line repeats on every page.
    bodies = [f"Figure {n} shows the results.\nSame sentence on every page." for n in range(1, 5)]
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", bodies))
    for number, section in enumerate(doc.sections, start=1):
        assert f"Figure {number} shows the results." in section.text
        assert "Same sentence on every page." in section.text


def test_hyphenated_line_breaks_are_joined(tmp_path):
    bodies = ["This is an exam-\nple of a split word.", "Page two body.", "Page three body."]
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", bodies))
    assert "example of a split word." in doc.sections[0].text
    assert "exam-" not in doc.sections[0].text


def test_real_hyphenated_words_inside_a_line_are_kept(tmp_path):
    bodies = ["A state-of-the-art result.", "Body two.", "Body three."]
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", bodies))
    assert "state-of-the-art" in doc.sections[0].text


def test_numbered_first_line_becomes_the_page_heading(tmp_path):
    bodies = [
        "1. Introduction\n\nSome body text here.",
        "2. Methods\n\nMore body text.",
        "Plain body.",
    ]
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", bodies))
    assert [s.heading for s in doc.sections] == ["1. Introduction", "2. Methods", None]
    assert doc.sections[0].text == "Some body text here."


def test_pdf_title_from_metadata_else_first_line_else_filename(tmp_path):
    bodies = ["Body one.", "Body two.", "Body three."]
    with_meta = parse_pdf(make_pdf(tmp_path / "a.pdf", bodies, title="Metadata Title"))
    assert with_meta.title == "Metadata Title"
    no_meta = parse_pdf(make_pdf(tmp_path / "b.pdf", ["First Line Title\n\nrest"] * 3))
    assert no_meta.title == "First Line Title"
    empty = parse_pdf(make_pdf(tmp_path / "scan-2024.pdf", [""] * 3))
    assert empty.title == "scan-2024"


def test_pdf_with_no_text_keeps_empty_pages(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", ["", "", ""]))
    assert [s.page_number for s in doc.sections] == [1, 2, 3]
    assert all(s.text == "" for s in doc.sections)


def test_pdf_date_comes_from_metadata_when_present(tmp_path):
    path = make_pdf(tmp_path / "a.pdf", ["x", "y", "z"])
    pdf = pymupdf.open(path)
    pdf.set_metadata({"creationDate": "D:20190507101500+00'00'"})
    pdf.saveIncr()
    pdf.close()
    assert parse_pdf(path).doc_date == date(2019, 5, 7)


# ------------------------------------------------------- hash & dispatcher


def test_content_hash_is_stable_and_changes_with_content(tmp_path):
    path = write(tmp_path, "a.md", "# T\n\none\n")
    first = parse_file(path).content_hash
    assert parse_file(path).content_hash == first
    assert first == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("# T\n\ntwo\n", encoding="utf-8")
    assert parse_file(path).content_hash != first


def test_pdf_content_hash_changes_when_file_changes(tmp_path):
    path = make_pdf(tmp_path / "a.pdf", ["one", "two", "three"])
    first = parse_file(path).content_hash
    assert parse_file(path).content_hash == first
    make_pdf(path, ["one", "two", "different"])
    assert parse_file(path).content_hash != first


def test_parse_file_picks_parser_by_extension(tmp_path):
    assert parse_file(write(tmp_path, "a.md", "# T\n")).source_type == "markdown"
    assert parse_file(write(tmp_path, "b.MARKDOWN", "# T\n")).source_type == "markdown"
    assert parse_file(make_pdf(tmp_path / "c.pdf", ["x", "y", "z"])).source_type == "pdf"


def test_parse_file_rejects_unsupported_types_clearly(tmp_path):
    path = write(tmp_path, "notes.txt", "hello")
    with pytest.raises(ValueError, match=r"Unsupported file type '\.txt'.*\.pdf"):
        parse_file(path)
