from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass
class Section:
    """A piece of a document with one heading and (for PDFs) one page."""

    heading: str | None
    page_number: int | None
    text: str


@dataclass
class ParsedDocument:
    source_type: Literal["pdf", "markdown"]
    path: str
    title: str
    content_hash: str  # sha256 of the file bytes, used to detect changed files
    doc_date: date | None
    sections: list[Section] = field(default_factory=list)


@dataclass
class Chunk:
    """A searchable piece of text that keeps the info needed for citations."""

    text: str
    chunk_index: int
    section_heading: str | None
    page_number: int | None
