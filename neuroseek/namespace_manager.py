"""
NamespaceManager — multiple isolated SearchEngine instances keyed by name.

Each namespace has its own HNSW graph, document store, and ID counter.
Documents in different namespaces never interfere with each other.
"""

from __future__ import annotations

from typing import Optional

from neuroseek.search_engine import SearchEngine
from neuroseek.embedder import Embedder, DEFAULT_MODEL


class NamespaceManager:
    """Manages a collection of named, isolated :class:`SearchEngine` instances.

    Parameters
    ----------
    model_name:
        Sentence-transformers model used by every namespace.
    M:
        HNSW ``M`` parameter applied to every namespace engine.
    efConstruction:
        HNSW ``efConstruction`` parameter applied to every namespace engine.
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
        self._embedder = Embedder(model_name)   # shared across all namespaces
        self._namespaces: dict[str, SearchEngine] = {}

    @classmethod
    def _from_embedder(
        cls,
        embedder: Embedder,
        M: int = 16,
        efConstruction: int = 200,
    ) -> "NamespaceManager":
        """Create a NamespaceManager that reuses an already-loaded *embedder*.

        Avoids re-loading the model when a shared embedder already exists
        (e.g. in tests or when building multiple managers from the same model).
        For internal / testing use only.
        """
        instance = object.__new__(cls)
        instance.model_name = embedder.model_name
        instance.M = M
        instance.efConstruction = efConstruction
        instance._embedder = embedder
        instance._namespaces = {}
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_namespace(self, name: object) -> None:
        if not isinstance(name, str):
            raise TypeError(
                f"namespace must be a str, got {type(name).__name__}"
            )
        if not name.strip():
            raise ValueError("namespace must not be empty or whitespace-only")

    def _get_namespace(self, name: str) -> SearchEngine:
        """Return the engine for *name*, raising KeyError if it doesn't exist."""
        self._validate_namespace(name)
        if name not in self._namespaces:
            raise KeyError(f"namespace {name!r} does not exist")
        return self._namespaces[name]

    def _make_engine(self) -> SearchEngine:
        return SearchEngine._from_embedder(
            embedder=self._embedder,
            M=self.M,
            efConstruction=self.efConstruction,
        )

    # ------------------------------------------------------------------
    # Namespace management
    # ------------------------------------------------------------------

    def create_namespace(self, name: str) -> None:
        """Create a new empty namespace called *name*.

        Raises
        ------
        TypeError / ValueError
            On invalid *name*.
        ValueError
            If *name* already exists.
        """
        self._validate_namespace(name)
        if name in self._namespaces:
            raise ValueError(f"namespace {name!r} already exists")
        self._namespaces[name] = self._make_engine()

    def delete_namespace(self, name: str) -> None:
        """Delete namespace *name* and all its documents.

        Raises
        ------
        KeyError
            If *name* does not exist.
        """
        self._get_namespace(name)  # validates + raises KeyError if missing
        del self._namespaces[name]

    def list_namespaces(self) -> list[str]:
        """Return a sorted list of all namespace names."""
        return sorted(self._namespaces.keys())

    # ------------------------------------------------------------------
    # Document operations (auto-create namespace on first use)
    # ------------------------------------------------------------------

    def _ensure_namespace(self, name: str) -> SearchEngine:
        """Return the engine for *name*, creating it if it doesn't exist."""
        self._validate_namespace(name)
        if name not in self._namespaces:
            self._namespaces[name] = self._make_engine()
        return self._namespaces[name]

    def add(
        self,
        text: str,
        namespace: str,
        id: Optional[int] = None,  # noqa: A002
        metadata: Optional[dict] = None,
    ) -> int:
        """Embed *text* and add it to *namespace* (auto-created if absent).

        Parameters
        ----------
        text:
            Non-empty document string to store and index.
        namespace:
            Target namespace (auto-created if absent).
        id:
            Optional explicit integer ID.
        metadata:
            Optional flat dict of str → str/int/float/bool pairs.

        Returns
        -------
        int
            The ID assigned to the document within *namespace*.
        """
        engine = self._ensure_namespace(namespace)
        return engine.add(text, id=id, metadata=metadata)

    def add_batch(
        self,
        texts: list[str] | tuple[str, ...],
        namespace: str,
        ids: Optional[list[int] | tuple[int, ...]] = None,
        metadata_list: Optional[list[Optional[dict]]] = None,
    ) -> list[int]:
        """Embed and index multiple documents into *namespace*.

        Parameters
        ----------
        texts:
            List or tuple of non-empty document strings.
        namespace:
            Target namespace (auto-created if absent).
        ids:
            Optional list/tuple of explicit IDs.
        metadata_list:
            Optional list of metadata dicts (or None entries), one per text.
        """
        engine = self._ensure_namespace(namespace)
        return engine.add_batch(texts, ids=ids, metadata_list=metadata_list)

    def delete(self, id: int, namespace: str) -> None:  # noqa: A002
        """Remove document *id* from *namespace*.

        Raises
        ------
        KeyError
            If *namespace* does not exist or *id* is not found.
        """
        engine = self._get_namespace(namespace)
        engine.delete(id)

    def search(
        self,
        query: str,
        namespace: str,
        top_k: int = 5,
        filter: Optional[dict] = None,  # noqa: A002
    ) -> list[dict]:
        """Search *namespace* for *query*.

        Parameters
        ----------
        query:
            Non-empty query string.
        namespace:
            Namespace to search. Must already exist.
        top_k:
            Number of results to return.
        filter:
            Optional metadata filter dict. Only documents whose metadata
            contains all key-value pairs in *filter* are returned.

        Returns
        -------
        list[dict]
            Same structure as :meth:`SearchEngine.search` —
            each dict has ``id``, ``text``, ``score``, and ``metadata``.

        Raises
        ------
        KeyError
            If *namespace* does not exist.
        """
        engine = self._get_namespace(namespace)
        return engine.search(query, top_k=top_k, filter=filter)

    # ------------------------------------------------------------------
    # Aggregate info
    # ------------------------------------------------------------------

    def namespace_len(self, name: str) -> int:
        """Return the number of documents in *name*."""
        return len(self._get_namespace(name))

    def __len__(self) -> int:
        """Return the total number of documents across all namespaces."""
        return sum(len(e) for e in self._namespaces.values())
