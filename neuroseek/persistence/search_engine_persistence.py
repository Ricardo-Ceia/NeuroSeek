"""
Persistence for SearchEngine — save to and load from disk using pickle.

The serialised payload captures everything needed to reconstruct the engine
exactly:
- model_name / M / efConstruction / backend  (constructor params)
- the full index internal state (backend-specific)
- the DocumentStore internal state
- _next_id counters for both, so auto-ID assignment continues without gaps

Backend-specific index serialisation
-------------------------------------
``"hnsw"`` (pure Python)
    Pickles the graph structures directly (layers, id_to_node, entry_point).

``"hnswlib"`` (C++ via hnswlib)
    Uses hnswlib's own ``save_index`` / ``load_index`` binary format, stored
    as a bytes blob inside the pickle payload, alongside the Python-side
    ``_id_to_vector`` dict and ``_deleted_ids`` set that hnswlib does not
    manage itself.

``"auto"``
    Saved under whichever backend was resolved at construction time; the
    actual resolved backend name is stored so load re-creates the same index
    type.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

from neuroseek.search_engine import SearchEngine

try:
    from neuroseek.core.hnswlib_index import HNSWLibIndex as _HNSWLibIndex
    _HNSWLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HNSWLibIndex = None  # type: ignore[assignment,misc]
    _HNSWLIB_AVAILABLE = False


def _resolved_backend(engine: SearchEngine) -> str:
    """Return the concrete backend name actually in use (never 'auto')."""
    if _HNSWLIB_AVAILABLE and isinstance(engine._index, _HNSWLibIndex):
        return "hnswlib"
    return "hnsw"


def save_search_engine(engine: SearchEngine, path: str | Path) -> None:
    """Serialise *engine* to *path*.

    Parameters
    ----------
    engine:
        A ``SearchEngine`` instance to persist.
    path:
        Destination file path (created or overwritten).

    Raises
    ------
    TypeError
        If *engine* is not a ``SearchEngine``.
    """
    if not isinstance(engine, SearchEngine):
        raise TypeError(
            f"engine must be a SearchEngine, got {type(engine).__name__}"
        )

    store = engine._store
    resolved = _resolved_backend(engine)

    payload: dict = {
        # Constructor params
        "model_name": engine.model_name,
        "M": engine.M,
        "efConstruction": engine.efConstruction,
        "backend": resolved,   # always concrete, never "auto"
        # DocumentStore internals
        "store_data": dict(store._store),
        "store_next_id": store._next_id,
    }

    if resolved == "hnswlib":
        idx = engine._index  # HNSWLibIndex
        # Serialise the hnswlib C++ index to bytes via a temp file
        if idx._index is not None:
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp_path = tmp.name
            idx._index.save_index(tmp_path)
            with open(tmp_path, "rb") as fh:
                index_bytes = fh.read()
            Path(tmp_path).unlink(missing_ok=True)
        else:
            index_bytes = None

        payload.update({
            "hnswlib_index_bytes": index_bytes,
            "hnswlib_dim": idx._dim,
            "hnswlib_capacity": idx._capacity,
            "hnswlib_num_vectors": idx.num_vectors,
            "hnswlib_next_id": idx._next_id,
            "hnswlib_id_to_vector": dict(idx._id_to_vector),
            "hnswlib_deleted_ids": set(idx._deleted_ids),
        })
    else:
        # Pure-Python HNSWIndex
        idx = engine._index
        payload.update({
            "hnsw_maxLayers": idx.maxLayers,
            "hnsw_layers": idx.layers,
            "hnsw_id_to_node": idx.id_to_node,
            "hnsw_entry_point_id": idx.entry_point.id if idx.entry_point else None,
            "hnsw_num_vectors": idx.num_vectors,
            "hnsw_next_id": idx._next_id,
        })

    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_search_engine(path: str | Path) -> SearchEngine:
    """Deserialise a ``SearchEngine`` from *path*.

    The model is **not** re-downloaded; the ``SentenceTransformer`` is
    initialised from the cached weights just like a fresh ``SearchEngine()``.
    The index and document store are restored without re-embedding.

    Parameters
    ----------
    path:
        File previously written by :func:`save_search_engine`.

    Returns
    -------
    SearchEngine
        Fully restored engine ready to search and accept new documents.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    resolved = payload.get("backend", "hnsw")  # backwards compat with old files

    # Reconstruct the engine with the concrete backend (never "auto")
    engine = SearchEngine(
        model_name=payload["model_name"],
        M=payload["M"],
        efConstruction=payload["efConstruction"],
        backend=resolved,
    )

    # Restore DocumentStore state
    store = engine._store
    store._store = payload["store_data"]
    store._next_id = payload["store_next_id"]

    if resolved == "hnswlib":
        idx = engine._index  # HNSWLibIndex
        idx._dim = payload["hnswlib_dim"]
        idx._capacity = payload["hnswlib_capacity"]
        idx.num_vectors = payload["hnswlib_num_vectors"]
        idx._next_id = payload["hnswlib_next_id"]
        idx._id_to_vector = payload["hnswlib_id_to_vector"]
        idx._deleted_ids = payload["hnswlib_deleted_ids"]

        index_bytes = payload["hnswlib_index_bytes"]
        if index_bytes is not None and idx._dim > 0:
            import hnswlib as _hnswlib
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp.write(index_bytes)
                tmp_path = tmp.name
            hnswlib_idx = _hnswlib.Index(space="cosine", dim=idx._dim)
            hnswlib_idx.load_index(tmp_path, max_elements=idx._capacity)
            Path(tmp_path).unlink(missing_ok=True)
            hnswlib_idx.set_ef(max(engine.efConstruction, 50))
            idx._index = hnswlib_idx
    else:
        # Pure-Python HNSWIndex
        idx = engine._index
        idx.maxLayers = payload["hnsw_maxLayers"]
        idx.layers = payload["hnsw_layers"]
        idx.id_to_node = payload["hnsw_id_to_node"]
        idx.num_vectors = payload["hnsw_num_vectors"]
        idx._next_id = payload["hnsw_next_id"]
        ep_id = payload["hnsw_entry_point_id"]
        idx.entry_point = idx.id_to_node[ep_id] if ep_id is not None else None

    return engine
