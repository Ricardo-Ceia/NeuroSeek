"""
SearchEngine — high-level semantic search over raw text documents.

Ties together:
- ``Embedder``      — converts text → Vector
- ``HNSWIndex``     — approximate nearest-neighbour index over Vectors
- ``DocumentStore`` — maps integer IDs → raw text + metadata

The same integer ID is used in both the ``HNSWIndex`` and the
``DocumentStore``, so every search result can be enriched with the original
text and metadata without a secondary lookup table.
"""

from __future__ import annotations

from typing import Optional

from neuroseek.embedder import Embedder, DEFAULT_MODEL
from neuroseek.core.hnsw_index import HNSWIndex
from neuroseek.document_store import DocumentStore, _validate_metadata

# How many extra candidates to fetch from HNSW when a metadata filter is
# active.  We fetch top_k * _FILTER_OVERSAMPLE results then trim to top_k
# after filtering.  If fewer than top_k pass the filter we return what we have.
_FILTER_OVERSAMPLE = 10


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

    def add(
        self,
        text: str,
        id: Optional[int] = None,  # noqa: A002
        metadata: Optional[dict] = None,
    ) -> int:
        """Embed *text* and add it to the engine.

        Parameters
        ----------
        text:
            Non-empty document string to store and index.
        id:
            Optional explicit integer ID. If omitted, one is auto-assigned.
            Providing an existing ID raises ``ValueError`` (the HNSW index does
            not support in-place upsert; call ``delete`` first).
        metadata:
            Optional flat dict of str → str/int/float/bool pairs attached to
            the document and used for filtering at search time.

        Returns
        -------
        int
            The ID under which the document is stored.
        """
        _validate_metadata(metadata)
        vector = self._embedder.encode(text)
        assigned_id = self._index.add_vector(vector, id=id)
        self._store.add(text, id=assigned_id, metadata=metadata)
        return assigned_id

    def add_batch(
        self,
        texts: list[str] | tuple[str, ...],
        ids: Optional[list[int] | tuple[int, ...]] = None,
        metadata_list: Optional[list[Optional[dict]]] = None,
    ) -> list[int]:
        """Embed and index multiple documents.

        Parameters
        ----------
        texts:
            List or tuple of non-empty document strings.
        ids:
            Optional list/tuple of explicit IDs (same length as *texts*).
        metadata_list:
            Optional list of metadata dicts (or None entries), one per text.

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

        if metadata_list is not None:
            if not isinstance(metadata_list, (list, tuple)):
                raise TypeError(
                    f"metadata_list must be a list or tuple, "
                    f"got {type(metadata_list).__name__}"
                )
            if len(metadata_list) != len(texts):
                raise ValueError(
                    f"metadata_list and texts must have the same length "
                    f"(got {len(metadata_list)} and {len(texts)})"
                )
            for i, meta in enumerate(metadata_list):
                _validate_metadata(meta, context=f"index {i}")

        # Embed all texts (validates text types / empty strings)
        vectors = self._embedder.encode_batch(texts)

        ids_list = list(ids) if ids is not None else [None] * len(texts)
        assigned_ids = self._index.add_vectors(vectors, ids=ids_list)

        for i, (text, assigned_id) in enumerate(zip(texts, assigned_ids)):
            meta = metadata_list[i] if metadata_list is not None else None
            self._store.add(text, id=assigned_id, metadata=meta)

        return assigned_ids

    def delete_by_query(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[dict] = None,  # noqa: A002
    ) -> list[dict]:
        """Search for *query* and delete all matching documents.

        Runs ``search(query, top_k, filter)``, deletes every returned
        document, and returns the list of result dicts that were deleted
        (same structure as :meth:`search`).

        Parameters
        ----------
        query:
            Non-empty query string.
        top_k:
            Maximum number of chunks to delete. Defaults to 5.
        filter:
            Optional metadata filter applied before deleting.

        Returns
        -------
        list[dict]
            The result dicts that were deleted (id, text, score, metadata),
            sorted by score descending.  Empty list if nothing matched.

        Raises
        ------
        TypeError / ValueError
            Same as :meth:`search`.
        """
        results = self.search(query, top_k=top_k, filter=filter)
        for result in results:
            self._index.delete_vector(result["id"])
            self._store.delete(result["id"])
        return results

    def delete_by_source(self, filename: str) -> int:
        """Remove all documents whose ``filename`` metadata value equals *filename*.

        Parameters
        ----------
        filename:
            The filename to match against the ``"filename"`` metadata field.

        Returns
        -------
        int
            The number of documents (chunks) that were deleted.

        Raises
        ------
        TypeError
            If *filename* is not a str.
        ValueError
            If *filename* is empty or whitespace-only.
        """
        if not isinstance(filename, str):
            raise TypeError(
                f"filename must be a str, got {type(filename).__name__}"
            )
        if not filename.strip():
            raise ValueError("filename must not be empty or whitespace-only")

        ids_to_delete = self._store.ids_for_metadata("filename", filename)
        for doc_id in ids_to_delete:
            self._index.delete_vector(doc_id)
            self._store.delete(doc_id)
        return len(ids_to_delete)

    def delete(self, id: int) -> None:  # noqa: A002
        """Remove the document with *id* from both the index and the store."""
        self._index.delete_vector(id)
        self._store.delete(id)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[dict] = None,  # noqa: A002
    ) -> list[dict]:
        """Return the *top_k* most semantically similar documents to *query*.

        Parameters
        ----------
        query:
            Non-empty query string.
        top_k:
            Number of results to return. Defaults to 5.
        filter:
            Optional metadata filter. Only documents whose metadata contains
            all key-value pairs in *filter* are returned. ``None`` means no
            filtering.

        Returns
        -------
        list[dict]
            Each dict has the keys:
            - ``"id"``       (int)   — document ID
            - ``"text"``     (str)   — original document text
            - ``"score"``    (float) — cosine similarity in [0, 1]
            - ``"metadata"`` (dict)  — document metadata (may be empty)
            Sorted by score descending (best match first).

        Raises
        ------
        TypeError
            If *query* is not a str, *top_k* is not an int, or *filter* is
            not a dict.
        ValueError
            If *query* is empty/whitespace or *top_k* < 1.
        """
        _validate_metadata(filter)

        query_vector = self._embedder.encode(query)

        if filter:
            # Fetch more candidates so we have enough after filtering
            fetch_k = top_k * _FILTER_OVERSAMPLE
            raw_results = self._index.search(query_vector, top_k=fetch_k)
            results = []
            for doc_id, score in raw_results:
                if self._store.matches_filter(doc_id, filter):
                    results.append({
                        "id": doc_id,
                        "text": self._store.get(doc_id),
                        "score": score,
                        "metadata": self._store.get_metadata(doc_id),
                    })
                    if len(results) == top_k:
                        break
            return results
        else:
            raw_results = self._index.search(query_vector, top_k=top_k)
            return [
                {
                    "id": doc_id,
                    "text": self._store.get(doc_id),
                    "score": score,
                    "metadata": self._store.get_metadata(doc_id),
                }
                for doc_id, score in raw_results
            ]

    def list_sources(self, key: str = "filename") -> set:
        """Return the set of distinct values for metadata *key* across all documents.

        Parameters
        ----------
        key:
            The metadata key to aggregate. Defaults to ``"filename"``.

        Returns
        -------
        set
            Unique values for *key*. Documents without *key* are ignored.
        """
        return self._store.list_sources(key)

    def __len__(self) -> int:
        """Return the number of documents currently in the engine."""
        return len(self._index)
