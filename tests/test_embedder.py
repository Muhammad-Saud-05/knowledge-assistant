import numpy as np
import pytest

from src.ingestion.embedder import QUERY_INSTRUCTION, Embedder


class StubModel:
    """Stands in for SentenceTransformer: records inputs, returns fixed-size vectors."""

    def __init__(self, dims=384):
        self.dims = dims
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.append((list(texts), kwargs))
        return np.ones((len(texts), self.dims), dtype="float32")


def make_embedder(model):
    embedder = Embedder()
    embedder._model = model  # skip the real model download
    return embedder


def test_documents_get_no_prefix_and_use_batching_and_normalisation():
    model = StubModel()
    vectors = make_embedder(model).embed_documents(["first passage", "second passage"])
    texts, options = model.seen[0]
    assert texts == ["first passage", "second passage"]
    assert options["batch_size"] == 32 and options["normalize_embeddings"] is True
    assert len(vectors) == 2 and len(vectors[0]) == 384


def test_query_gets_the_bge_instruction_prefix():
    model = StubModel()
    vector = make_embedder(model).embed_query("what is rag")
    assert model.seen[0][0] == [QUERY_INSTRUCTION + "what is rag"]
    assert QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "
    assert len(vector) == 384


def test_wrong_dimension_raises_a_clear_error():
    with pytest.raises(ValueError, match="384 dimensions"):
        make_embedder(StubModel(dims=768)).embed_documents(["text"])


def test_empty_input_returns_empty_list_without_loading_the_model():
    assert Embedder().embed_documents([]) == []


@pytest.mark.slow
def test_real_model_gives_384_dims_and_sensible_similarity():
    embedder = Embedder()
    car, automobile, banana = embedder.embed_documents(["car", "automobile", "banana"])
    assert len(car) == 384

    def similarity(a, b):  # vectors are normalised, so the dot product is the cosine similarity
        return sum(x * y for x, y in zip(a, b))

    assert similarity(car, automobile) > similarity(car, banana)
