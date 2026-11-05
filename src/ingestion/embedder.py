"""Turn text into embedding vectors with the bge-small-en-v1.5 model.

About normalized vectors and cosine distance (plain language):

An embedding is a list of 384 numbers that acts like an arrow pointing in a
direction that stands for the text's meaning. Texts with similar meanings
point in similar directions. "Normalized" means every arrow is scaled to the
same length (1), so only the direction is left. The angle between two arrows
then tells us how alike the texts are: cosine distance is 0 when they point
the same way and grows as they drift apart. The database index uses this
distance, so similarity = 1 - distance (1.0 means identical direction).
"""

EXPECTED_DIMENSIONS = 384

# bge models were trained so that a short search question gets this prefix,
# while the passages being searched get none.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        # Loaded on first use: importing this module stays fast, and code that
        # never embeds anything never downloads or loads the model.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Model {self.model_name} returned vectors of shape {tuple(vectors.shape)}, "
                f"but the database expects {EXPECTED_DIMENSIONS} dimensions."
            )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages to store. No instruction prefix is added."""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a search question, with the bge query instruction in front."""
        return self._encode([QUERY_INSTRUCTION + text])[0]
