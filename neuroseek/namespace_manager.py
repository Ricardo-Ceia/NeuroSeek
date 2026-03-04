"""
NamespaceManager — multiple isolated SearchEngine instances keyed by name.

Each namespace has its own HNSW graph, document store, and ID counter.
Documents in different namespaces never interfere with each other.
"""

from __future__ import annotations

from typing import Optional

from neuroseek.search_engine import SearchEngine
from neuroseek.embedder import DEFAULT_MODEL


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
        self._namespaces: dict[str, SearchEngine] = {}

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
        return SearchEngine(
            model_name=self.model_name,
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
    ) -> int:
        """Embed *text* and add it to *namespace* (auto-created if absent).

        Returns
        -------
        int
            The ID assigned to the document within *namespace*.
        """
        engine = self._ensure_namespace(namespace)
        return engine.add(text, id=id)

    def add_batch(
        self,
        texts: list[str] | tuple[str, ...],
        namespace: str,
        ids: Optional[list[int] | tuple[int, ...]] = None,
    ) -> list[int]:
        """Embed and index multiple documents into *namespace*."""
        engine = self._ensure_namespace(namespace)
        return engine.add_batch(texts, ids=ids)

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
    ) -> list[dict]:
        """Search *namespace* for *query*.

        Returns
        -------
        list[dict]
            Same structure as :meth:`SearchEngine.search` —
            each dict has ``id``, ``text``, and ``score``.

        Raises
        ------
        KeyError
            If *namespace* does not exist.
        """
        engine = self._get_namespace(namespace)
        return engine.search(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Aggregate info
    # ------------------------------------------------------------------

    def namespace_len(self, name: str) -> int:
        """Return the number of documents in *name*."""
        return len(self._get_namespace(name))

    def __len__(self) -> int:
        """Return the total number of documents across all namespaces."""
        return sum(len(e) for e in self._namespaces.values())
