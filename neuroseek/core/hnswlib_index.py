"""HNSWLibIndex — hnswlib-backed drop-in replacement for HNSWIndex.

Requires the optional ``hnswlib`` package::

    pip install neuroseek[fast]

Exposes the same public interface as ``HNSWIndex`` so that ``SearchEngine``
and all persistence code can treat both backends identically:

    add_vector(vector, id=None) -> int
    add_vectors(vectors, ids=None) -> list[int]
    get_vector(id) -> Vector
    delete_vector(id) -> Vector
    search(query, top_k=5, ef=10) -> list[tuple[int, float]]
    __len__() -> int

Key differences from the pure-Python implementation:

- Build is ~120× faster (C++ HNSW with SIMD distances).
- Deletion uses hnswlib's ``mark_deleted`` (soft delete — the slot is
  excluded from search but the memory is not freed until the index is
  rebuilt or resized).  The ``Vector`` stored at insertion time is kept
  in a Python dict so ``get_vector`` / ``delete_vector`` can still
  return it.
- ``_dim`` and ``_next_id`` are exposed as attributes so the persistence
  layer can save/restore them without special-casing.
"""

from __future__ import annotations

import numpy as np

from neuroseek.core.vector import Vector

try:
    import hnswlib as _hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _hnswlib = None  # type: ignore[assignment]
    _HNSWLIB_AVAILABLE = False

# hnswlib requires a fixed max_elements at init time.  We start here and
# double when full — matches the numpy matrix growth strategy in HNSWIndex.
_INITIAL_CAPACITY = 256


def _require_hnswlib() -> None:
    if not _HNSWLIB_AVAILABLE:
        raise ImportError(
            "hnswlib is required for the 'hnswlib' backend.\n"
            "Install it with:  pip install neuroseek[fast]"
        )


class HNSWLibIndex:
    """hnswlib-backed HNSW index with the same interface as HNSWIndex.

    Parameters
    ----------
    M:
        Number of bi-directional edges per node.  Default 16.
    efConstruction:
        Beam width during graph construction.  Default 200.
    maxLayers:
        Unused — kept for API parity; hnswlib determines layer count
        internally via its own formula.
    """

    def __init__(
        self,
        M: int = 16,
        efConstruction: int = 200,
        maxLayers: int = 16,  # noqa: ARG002 — API parity only
    ) -> None:
        _require_hnswlib()

        self.M = M
        self.efConstruction = efConstruction
        self.maxLayers = maxLayers  # stored for persistence round-trips

        self._dim: int = 0
        self._capacity: int = _INITIAL_CAPACITY
        self._next_id: int = 0
        self.num_vectors: int = 0

        # hnswlib index — allocated on first insertion once dim is known.
        self._index: "_hnswlib.Index | None" = None  # type: ignore[name-defined]

        # id -> Vector: keep original Vector objects so get_vector works
        # and delete_vector can return the original vector.
        self._id_to_vector: dict[int, Vector] = {}

        # ids that have been soft-deleted via mark_deleted.
        # hnswlib keeps them in memory but excludes from search.
        self._deleted_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_index(self, dim: int) -> None:
        """Create the underlying hnswlib index for dimension *dim*."""
        self._dim = dim
        self._index = _hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=_INITIAL_CAPACITY,
            ef_construction=self.efConstruction,
            M=self.M,
        )
        self._index.set_ef(max(self.efConstruction, 50))

    def _grow_if_needed(self) -> None:
        """Double the index capacity when it is full."""
        assert self._index is not None
        if self.num_vectors >= self._capacity:
            self._capacity *= 2
            self._index.resize_index(self._capacity)

    def _vector_to_np(self, vector: Vector) -> np.ndarray:
        """Convert a Vector to a normalised float32 row vector."""
        arr = np.array(vector.data, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Cannot insert a zero-norm vector into HNSWLibIndex.")
        return (arr / norm).reshape(1, -1)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_vector(self, vector: Vector, id: int | None = None) -> int:  # noqa: A002
        """Embed *vector* and insert it into the index.

        Parameters
        ----------
        vector:
            The ``Vector`` to insert.
        id:
            Optional explicit integer ID.  Auto-assigned if omitted.

        Returns
        -------
        int
            The ID under which the vector is stored.
        """
        if not isinstance(vector, Vector):
            raise TypeError(
                f"vector must be a Vector, not {type(vector).__name__}"
            )

        if id is None:
            while self._next_id in self._id_to_vector or self._next_id in self._deleted_ids:
                self._next_id += 1
            id = self._next_id
            self._next_id += 1
        else:
            if not isinstance(id, int):
                raise TypeError(f"id must be an int, not {type(id).__name__}")
            if id in self._id_to_vector:
                raise ValueError(f"ID {id} already exists")
            if id >= self._next_id:
                self._next_id = id + 1

        dim = len(vector)

        # Lazy init: first insertion fixes the dimension.
        if self._index is None:
            self._init_index(dim)
        elif dim != self._dim:
            raise ValueError(
                f"Vector dimension {dim} does not match index dimension {self._dim}"
            )

        self._grow_if_needed()

        row = self._vector_to_np(vector)
        self._index.add_items(row, [id])
        self._id_to_vector[id] = vector
        self.num_vectors += 1

        return id

    def add_vectors(
        self,
        vectors: list[Vector],
        ids: list[int | None] | None = None,
    ) -> list[int]:
        """Insert multiple vectors, optionally with explicit IDs.

        Parameters
        ----------
        vectors:
            List of ``Vector`` objects.
        ids:
            Optional list of integer IDs, one per vector.  ``None`` entries
            are auto-assigned.

        Returns
        -------
        list[int]
            Assigned IDs in the same order as *vectors*.
        """
        if not isinstance(vectors, (list, tuple)):
            raise TypeError(
                f"vectors must be a list or tuple, not {type(vectors).__name__}"
            )
        if ids is None:
            ids = [None] * len(vectors)
        if not isinstance(ids, (list, tuple)):
            raise TypeError(
                f"ids must be a list, tuple, or None, not {type(ids).__name__}"
            )
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        for id in ids:
            if id is not None and not isinstance(id, int):
                raise TypeError(
                    f"id must be an int or None, not {type(id).__name__}"
                )

        assigned: list[int] = []
        for vec, id in zip(vectors, ids):
            assigned.append(self.add_vector(vec, id))
        return assigned

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_vector(self, id: int) -> Vector:  # noqa: A002
        """Return the original ``Vector`` stored under *id*.

        Parameters
        ----------
        id:
            Integer ID of the stored vector.

        Raises
        ------
        TypeError
            If *id* is not an int.
        ValueError
            If *id* does not exist (or has been deleted).
        """
        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")
        if id not in self._id_to_vector:
            raise ValueError(f"ID {id} does not exist")
        return self._id_to_vector[id]

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_vector(self, id: int) -> Vector:  # noqa: A002
        """Remove the vector with *id* from the index.

        Uses hnswlib's ``mark_deleted`` (soft delete).  The vector slot
        is excluded from all future searches.  The memory is reclaimed
        when the index is next resized or reloaded.

        Parameters
        ----------
        id:
            Integer ID of the vector to delete.

        Returns
        -------
        Vector
            The original ``Vector`` that was stored under *id*.

        Raises
        ------
        TypeError
            If *id* is not an int.
        ValueError
            If *id* does not exist.
        """
        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")
        if id not in self._id_to_vector:
            raise ValueError(f"ID {id} does not exist")

        assert self._index is not None
        self._index.mark_deleted(id)
        vector = self._id_to_vector.pop(id)
        self._deleted_ids.add(id)
        self.num_vectors -= 1
        return vector

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: Vector,
        top_k: int = 5,
        ef: int = 10,
    ) -> list[tuple[int, float]]:
        """Return the *top_k* nearest neighbours to *query*.

        Parameters
        ----------
        query:
            ``Vector`` to search with.
        top_k:
            Number of results to return.  Defaults to 5.
        ef:
            Search beam width.  Higher values improve recall at the cost
            of latency.  Automatically raised to ``top_k`` if lower.

        Returns
        -------
        list[tuple[int, float]]
            ``(id, cosine_similarity)`` pairs sorted by similarity
            descending (best match first).  Scores are in ``[0, 1]``.
        """
        if not isinstance(query, Vector):
            raise TypeError(
                f"query must be a Vector, not {type(query).__name__}"
            )
        if not isinstance(top_k, int):
            raise TypeError(
                f"top_k must be an int, not {type(top_k).__name__}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if self._index is None or self.num_vectors == 0:
            return []

        effective_ef = max(ef, top_k)
        self._index.set_ef(effective_ef)

        # Clamp top_k to the number of live (non-deleted) vectors.
        k = min(top_k, self.num_vectors)

        arr = np.array(query.data, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Query vector has zero norm.")
        arr = (arr / norm).reshape(1, -1)

        labels, distances = self._index.knn_query(arr, k=k)
        # labels/distances are shape (1, k); flatten.
        # hnswlib cosine space: distance = 1 - cosine_similarity
        results = []
        for label, dist in zip(labels[0], distances[0]):
            score = float(1.0 - dist)
            results.append((int(label), score))

        # Already sorted best-first by hnswlib, but sort explicitly for safety.
        results.sort(key=lambda x: -x[1])
        return results

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_vectors
