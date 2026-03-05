from sentence_transformers import SentenceTransformer
from neuroseek.core.vector import Vector


DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be a str, not {type(model_name).__name__}"
            )
        if not model_name.strip():
            raise ValueError("model_name must not be empty or whitespace")

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Dimension is fixed for the lifetime of this Embedder instance
        self.dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError(
                f"text must be a str, not {type(text).__name__}"
            )
        if not text.strip():
            raise ValueError("text must not be empty or whitespace")

        embedding = self._model.encode(text, convert_to_numpy=True)
        v = Vector(self.dimension)
        v.data = embedding.tolist()
        return v

    def encode_batch(self, texts):
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"texts must be a list or tuple, not {type(texts).__name__}"
            )
        if len(texts) == 0:
            raise ValueError("texts must not be empty")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"texts[{i}] must be a str, not {type(t).__name__}"
                )
            if not t.strip():
                raise ValueError(
                    f"texts[{i}] must not be empty or whitespace"
                )

        embeddings = self._model.encode(list(texts), convert_to_numpy=True)
        vectors = []
        for emb in embeddings:
            v = Vector(self.dimension)
            v.data = emb.tolist()
            vectors.append(v)
        return vectors
