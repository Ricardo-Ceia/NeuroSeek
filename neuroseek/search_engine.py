"""
SearchEngine — high-level semantic search over raw text documents.

Ties together:
- ``Embedder``      — converts text → Vector
- ``HNSWIndex``     — approximate nearest-neighbour index over Vectors
- ``DocumentStore`` — maps integer IDs → raw text

The same integer ID is used in both the ``HNSWIndex`` and the
``DocumentStore``, so every search result can be enriched with the original
text without a secondary lookup table.
"""

from __future__ import annotations

from typing import Optional

from neuroseek.embedder import Embedder, DEFAULT_MODEL
from neuroseek.hnsw_index import HNSWIndex
from neuroseek.document_store import DocumentStore


class SearchEngine:
    """Text-to-text semantic search engine backed by HNSW.

    Parameters
    ----------
    model_name:
        Name of the sentence-transformers model used for embedding.
        Defaults to ``multi-qa-MiniLM-L6-cos-v1``.
    M:
        HNSW ``M`` parameter — number of bi-directional edges per node.
        Higher values improve recall at the cost of memory and index-build time.
    efConstruction:
        HNSW ``efConstruction`` parameter — search width during graph construction.
        Higher values improve recall at the cost of index-build time.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        M: int = 16,
        efConstruction: int = 200,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be a str, got {type(model_name).__name__}"
            )
        if not model_name.strip():
            raise ValueError("model_name must not be empty or whitespace-only")
        if not isinstance(M, int):
            raise TypeError(f"M must be an int, got {type(M).__name__}")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if not isinstance(efConstruction, int):
            raise TypeError(
                f"efConstruction must be an int, got {type(efConstruction).__name__}"
            )
        if efConstruction < 1:
            raise ValueError(f"efConstruction must be >= 1, got {efConstruction}")

        self.model_name = model_name
        self.M = M
        self.efConstruction = efConstruction

        self._embedder = Embedder(model_name)
        self._index = HNSWIndex(M=M, efConstruction=efConstruction)
        self._store = DocumentStore()

    @classmethod
    def _from_embedder(
        cls,
        embedder: Embedder,
        M: int = 16,
        efConstruction: int = 200,
    ) -> "SearchEngine":
        """Create a SearchEngine that reuses an already-loaded *embedder*.

        This avoids re-loading the model when many engines share the same
        model (e.g. inside NamespaceManager).  For internal use only.
        """
        instance = object.__new__(cls)
        instance.model_name = embedder.model_name
        instance.M = M
        instance.efConstruction = efConstruction
        instance._embedder = embedder
        instance._index = HNSWIndex(M=M, efConstruction=efConstruction)
        instance._store = DocumentStore()
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, id: Optional[int] = None) -> int:  # noqa: A002
        """Embed *text* and add it to the engine.

        Parameters
        ----------
        text:
            Non-empty document string to store and index.
        id:
            Optional explicit integer ID. If omitted, one is auto-assigned.
            Providing an existing ID raises ``ValueError`` (the HNSW index does
            not support in-place upsert; call ``delete`` first).

        Returns
        -------
        int
            The ID under which the document is stored.
        """
        # Validation is delegated to DocumentStore and HNSWIndex — they raise
        # with clear messages on bad input.
        vector = self._embedder.encode(text)
        # Add to the HNSW index first so that, if an explicit id collides, we
        # fail before mutating the document store.
        assigned_id = self._index.add_vector(vector, id=id)
        # Use the exact same id in the document store (it accepts explicit ids).
        self._store.add(text, id=assigned_id)
        return assigned_id

    def add_batch(
        self,
        texts: list[str] | tuple[str, ...],
        ids: Optional[list[int] | tuple[int, ...]] = None,
    ) -> list[int]:
        """Embed and index multiple documents.

        Parameters
        ----------
        texts:
            List or tuple of non-empty document strings.
        ids:
            Optional list/tuple of explicit IDs (same length as *texts*).

        Returns
        -------
        list[int]
            IDs in the same order as *texts*.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"texts must be a list or tuple, got {type(texts).__name__}"
            )
        if len(texts) == 0:
            raise ValueError("texts must not be empty")

        if ids is not None:
            if not isinstance(ids, (list, tuple)):
                raise TypeError(
                    f"ids must be a list or tuple, got {type(ids).__name__}"
                )
            if len(ids) != len(texts):
                raise ValueError(
                    f"ids and texts must have the same length "
                    f"(got {len(ids)} ids and {len(texts)} texts)"
                )

        # Embed all texts first (validates text types / empty strings)
        vectors = self._embedder.encode_batch(texts)

        ids_list = list(ids) if ids is not None else [None] * len(texts)

        # Add vectors to the HNSW index
        assigned_ids = self._index.add_vectors(vectors, ids=ids_list)

        # Mirror every document into the store using the assigned id
        for text, assigned_id in zip(texts, assigned_ids):
            self._store.add(text, id=assigned_id)

        return assigned_ids

    def delete(self, id: int) -> None:  # noqa: A002
        """Remove the document with *id* from both the index and the store.

        Parameters
        ----------
        id:
            Integer ID of the document to remove.

        Raises
        ------
        TypeError
            If *id* is not an int.
        ValueError
            If *id* does not exist in the index.
        """
        # Delete from HNSW first (stricter; raises ValueError on missing id)
        self._index.delete_vector(id)
        # Delete from document store (should always succeed if index deletion did)
        self._store.delete(id)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the *top_k* most semantically similar documents to *query*.

        Parameters
        ----------
        query:
            Non-empty query string.
        top_k:
            Number of results to return. Defaults to 5.

        Returns
        -------
        list[dict]
            Each dict has the keys:
            - ``"id"``    (int)   — document ID
            - ``"text"``  (str)   — original document text
            - ``"score"`` (float) — cosine similarity in [0, 1], higher = more similar
            Sorted by score descending (best match first).

        Raises
        ------
        TypeError
            If *query* is not a str or *top_k* is not an int.
        ValueError
            If *query* is empty/whitespace or *top_k* < 1.
        """
        # encode() validates text type and emptiness
        query_vector = self._embedder.encode(query)
        raw_results = self._index.search(query_vector, top_k=top_k)
        return [
            {
                "id": doc_id,
                "text": self._store.get(doc_id),
                "score": score,
            }
            for doc_id, score in raw_results
        ]

    def __len__(self) -> int:
        """Return the number of documents currently in the engine."""
        return len(self._index)
