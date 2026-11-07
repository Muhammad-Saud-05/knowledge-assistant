"""The shapes of the JSON the API sends and receives."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..retrieval.filters import SearchFilters

# ----------------------------------------------------------------------- health


class DatabaseHealth(BaseModel):
    ok: bool
    documents: int | None = None
    chunks: int | None = None
    detail: str | None = None


class OllamaHealth(BaseModel):
    ok: bool  # the Ollama server answers
    model: str
    model_installed: bool | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: DatabaseHealth
    ollama: OllamaHealth


# -------------------------------------------------------------------- documents


class DocumentOut(BaseModel):
    id: int
    title: str | None
    path: str
    source_type: str
    doc_date: date | None
    ingested_at: datetime
    chunks: int


class DeleteResult(BaseModel):
    deleted: bool
    file_removed: bool  # the uploaded copy under data/uploads was also deleted


class IngestResult(BaseModel):
    status: Literal["new", "updated", "unchanged", "failed"]
    chunks: int
    document_id: int | None
    filename: str  # the sanitized name the file was saved under
    error: str | None = None


# -------------------------------------------------------------------------- ask


class FiltersIn(BaseModel):
    source_types: list[Literal["pdf", "markdown"]] | None = Field(default=None, min_length=1)
    date_from: date | None = None
    date_to: date | None = None
    path_prefix: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _dates_in_order(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self

    def to_search_filters(self) -> SearchFilters | None:
        if not any((self.source_types, self.date_from, self.date_to, self.path_prefix)):
            return None
        return SearchFilters(
            source_types=self.source_types,
            date_from=self.date_from,
            date_to=self.date_to,
            path_prefix=self.path_prefix or None,
        )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    filters: FiltersIn | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value
