"""Reorder retrieved chunks with a cross-encoder.

Vector and keyword search compare a question with each chunk separately. A cross-encoder reads the
question and one chunk together and outputs a single relevance score, which is slower but usually
more precise, so it is used only on a short list of candidates.

About the score: the default model (cross-encoder/ms-marco-MiniLM-L-6-v2) has a single output that
is an unbounded logit (measured here: +5.2 for a relevant passage, -11.4 for an irrelevant one).
This module asks for the raw output explicitly (Identity activation) and keeps it as `rerank_score`.
No sigmoid is applied. (The installed sentence-transformers 6.2.0 also defaults to Identity for this
model; other versions or models may default to a sigmoid, which is why it is set explicitly.) A
sigmoid would never change the ranking; it would only squash the scale into 0..1.
"""

import os
from collections.abc import Callable
from dataclasses import replace

from .models import RetrievedChunk

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INPUT_STYLES = ("plain", "with_context")

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]


def build_text(chunk: RetrievedChunk, style: str) -> str:
    """The passage text the model reads for a chunk."""
    if style == "plain":
        return chunk.text
    if style == "with_context":
        parts = [chunk.title, chunk.section_heading, chunk.text]
        return "\n".join(p for p in parts if p)
    raise ValueError(f"unknown input style {style!r}; use one of: {', '.join(INPUT_STYLES)}")


class Reranker:
    def __init__(
        self,
        model_name: str | None = None,
        input_style: str = "plain",
        max_length: int = 512,
        batch_size: int = 16,
        score_fn: ScoreFn | None = None,
        cache: dict | None = None,
    ):
        if input_style not in INPUT_STYLES:
            raise ValueError(f"unknown input style {input_style!r}; use one of: {', '.join(INPUT_STYLES)}")
        self.model_name = model_name or os.environ.get("RERANKER_MODEL") or DEFAULT_RERANKER_MODEL
        self.input_style = input_style
        self.max_length = max_length
        self.batch_size = batch_size
        self._score_fn = score_fn  # injected in tests so no model is needed
        self._model = None
        # Optional: (model, style, query, chunk_id) -> raw score, so repeated pairs are not recomputed.
        self.cache = cache

    @property
    def model(self):
        # Loaded on first use, on the CPU, so importing this module stays fast.
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)
        return self._model

    def load(self) -> None:
        """Load the model now (used to time the load and to warm up)."""
        self._score([("warm up", "warm up")])

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._score_fn is not None:
            return [float(s) for s in self._score_fn(pairs)]
        import torch

        scores = self.model.predict(
            pairs, batch_size=self.batch_size, activation_fn=torch.nn.Identity(), show_progress_bar=False
        )
        return [float(s) for s in scores]

    def rerank(self, query: str, chunks: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """The best k chunks for the query, best first, with rerank_score and rerank_rank set.

        `score` becomes the raw rerank score. Everything else about a chunk (ranks from the earlier
        stages, title, page ...) is kept. Equal scores keep their incoming order, so the result is
        always the same for the same input.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if not chunks:
            return []

        scores: list[float | None] = [None] * len(chunks)
        missing: list[int] = []
        for i, chunk in enumerate(chunks):
            key = (self.model_name, self.input_style, query, chunk.chunk_id)
            if self.cache is not None and key in self.cache:
                scores[i] = self.cache[key]
            else:
                missing.append(i)
        if missing:
            pairs = [(query, build_text(chunks[i], self.input_style)) for i in missing]
            fresh = self._score(pairs)
            if len(fresh) != len(pairs):
                raise ValueError("the scoring function returned the wrong number of scores")
            for i, value in zip(missing, fresh):
                scores[i] = value
                if self.cache is not None:
                    self.cache[(self.model_name, self.input_style, query, chunks[i].chunk_id)] = value

        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        results = []
        for rank, i in enumerate(order[:k], start=1):
            results.append(replace(chunks[i], score=scores[i], rerank_score=scores[i], rerank_rank=rank))
        return results
