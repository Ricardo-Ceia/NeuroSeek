"""
Persistence for NamespaceManager — save all namespaces to a single file
and restore them without re-embedding any documents.

Backend-specific index serialisation
--------------------------------------
Each namespace engine is saved with the same backend-aware logic used by
``save_search_engine`` / ``load_search_engine``:

``"hnsw"``    — pickles the pure-Python graph structures directly.
``"hnswlib"`` — uses hnswlib's own ``save_index`` / ``load_index`` binary
                format, stored as a bytes blob inside the pickle payload.
``"auto"``    — saved under whichever backend was resolved at construction
                time (never stored as ``"auto"``).

Format
------
A pickle file containing:

.. code-block:: python

    {
        "persistence_version": "2",
        "model_name":          str,
        "M":                   int,
        "efConstruction":      int,
        "namespaces": {
            "<name>": { ... per-engine payload ... },
            ...
        }
    }

The payload always includes a ``"persistence_version"`` key whose value is
:data:`PERSISTENCE_VERSION`.  :func:`load_namespace_manager` raises a clear
:exc:`ValueError` if the stored version is absent or does not match.

Changelog
---------
``"1"`` — original; only supported the pure-Python ``hnsw`` backend.
``"2"`` — added ``"backend"`` key per namespace; full ``hnswlib`` support.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

from neuroseek.namespace_manager import NamespaceManager
from neuroseek.search_engine import SearchEngine
from neuroseek.embedder import Embedder

try:
    from neuroseek.core.hnswlib_index import HNSWLibIndex as _HNSWLibIndex
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLibIndex = None
    _HNSWLIB_AVAILABLE = False

#: Current binary persistence format version.  Bump whenever the pickle
#: schema changes in a backwards-incompatible way.
PERSISTENCE_VERSION = "2"


def _resolved_backend(engine: SearchEngine) -> str:
    """Return the concrete backend name actually in use (never ``'auto'``)."""
    if _HNSWLIB_AVAILABLE and isinstance(engine._index, _HNSWLibIndex):
        return "hnswlib"
    return "hnsw"


def _save_engine_payload(engine: SearchEngine) -> dict:
    """Serialise one SearchEngine to a plain dict (no file I/O)."""
    store    = engine._store
    resolved = _resolved_backend(engine)

    payload: dict = {
        "model_name":     engine.model_name,
        "M":              engine.M,
        "efConstruction": engine.efConstruction,
        "backend":        resolved,
        "store_data":     dict(store._store),
        "store_next_id":  store._next_id,
    }

    if resolved == "hnswlib":
        idx = engine._index  # HNSWLibIndex
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
            "hnswlib_index_bytes":  index_bytes,
            "hnswlib_dim":          idx._dim,
            "hnswlib_capacity":     idx._capacity,
            "hnswlib_num_vectors":  idx.num_vectors,
            "hnswlib_next_id":      idx._next_id,
            "hnswlib_id_to_vector": dict(idx._id_to_vector),
            "hnswlib_deleted_ids":  set(idx._deleted_ids),
        })
    else:
        # Pure-Python HNSWIndex
        idx = engine._index
        payload.update({
            "hnsw_maxLayers":      idx.maxLayers,
            "hnsw_layers":         idx.layers,
            "hnsw_id_to_node":     idx.id_to_node,
            "hnsw_entry_point_id": idx.entry_point.id if idx.entry_point else None,
            "hnsw_num_vectors":    idx.num_vectors,
            "hnsw_next_id":        idx._next_id,
        })

    return payload


def _load_engine_payload(engine_payload: dict, shared_embedder: Embedder) -> SearchEngine:
    """Restore one SearchEngine from a serialised dict."""
    resolved = engine_payload.get("backend", "hnsw")  # backwards-compat: v1 had no "backend"

    engine = SearchEngine._from_embedder(
        embedder=shared_embedder,
        M=engine_payload["M"],
        efConstruction=engine_payload["efConstruction"],
        backend=resolved,
    )

    store = engine._store
    store._store   = engine_payload["store_data"]
    store._next_id = engine_payload["store_next_id"]

    if resolved == "hnswlib":
        idx = engine._index  # HNSWLibIndex
        idx._dim          = engine_payload["hnswlib_dim"]
        idx._capacity     = engine_payload["hnswlib_capacity"]
        idx.num_vectors   = engine_payload["hnswlib_num_vectors"]
        idx._next_id      = engine_payload["hnswlib_next_id"]
        idx._id_to_vector = engine_payload["hnswlib_id_to_vector"]
        idx._deleted_ids  = engine_payload["hnswlib_deleted_ids"]

        index_bytes = engine_payload["hnswlib_index_bytes"]
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
        idx.maxLayers   = engine_payload["hnsw_maxLayers"]
        idx.layers      = engine_payload["hnsw_layers"]
        idx.id_to_node  = engine_payload["hnsw_id_to_node"]
        idx.num_vectors = engine_payload["hnsw_num_vectors"]
        idx._next_id    = engine_payload["hnsw_next_id"]
        ep_id           = engine_payload["hnsw_entry_point_id"]
        idx.entry_point = idx.id_to_node[ep_id] if ep_id is not None else None
        # Rebuild the transient numpy matrix and row-mapping dicts from the
        # restored id_to_node.  These are not pickled and must be reconstructed.
        idx._rebuild_matrix()

    return engine


# ── SAVE ────────────────────────────────────────────────────────────────────

def save_namespace_manager(manager: NamespaceManager, path: str | Path) -> None:
    """Save *manager* to *path* as a pickle file.

    Both ``"hnsw"`` (pure-Python) and ``"hnswlib"`` (C++) backends are
    supported.  The concrete backend in use is resolved per namespace at save
    time and stored in the payload.

    Parameters
    ----------
    manager:
        The :class:`~neuroseek.namespace_manager.NamespaceManager` to save.
    path:
        Destination file path.  Will be created or overwritten.

    Raises
    ------
    TypeError
        If *manager* is not a :class:`~neuroseek.namespace_manager.NamespaceManager`.
    """
    if not isinstance(manager, NamespaceManager):
        raise TypeError(
            f"manager must be a NamespaceManager, got {type(manager).__name__}"
        )

    namespaces_payload: dict[str, dict] = {
        name: _save_engine_payload(engine)
        for name, engine in manager._namespaces.items()
    }

    payload = {
        "persistence_version": PERSISTENCE_VERSION,
        "model_name":          manager.model_name,
        "M":                   manager.M,
        "efConstruction":      manager.efConstruction,
        "namespaces":          namespaces_payload,
    }

    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


# ── LOAD ────────────────────────────────────────────────────────────────────

def load_namespace_manager(path: str | Path) -> NamespaceManager:
    """Load a :class:`~neuroseek.namespace_manager.NamespaceManager` from *path*.

    Supports files saved with the current persistence format (version
    :data:`PERSISTENCE_VERSION`).  Raises :exc:`ValueError` for missing or
    mismatched version headers so callers get a clear error rather than
    cryptic unpickling failures.

    Parameters
    ----------
    path:
        Path to the file previously created by :func:`save_namespace_manager`.

    Returns
    -------
    NamespaceManager
        Fully restored manager with all namespaces and documents.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file's persistence version is missing or does not match
        :data:`PERSISTENCE_VERSION`.
    """
    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    stored_version = payload.get("persistence_version")
    if stored_version != PERSISTENCE_VERSION:
        if stored_version is None:
            raise ValueError(
                f"The index file at {str(path)!r} was created by an older version of "
                f"NeuroSeek that does not include a persistence version header. "
                f"Please re-index your data with the current version."
            )
        raise ValueError(
            f"The index file at {str(path)!r} was created with persistence_version "
            f"{stored_version!r}, but this version of NeuroSeek requires "
            f"{PERSISTENCE_VERSION!r}. Please re-index your data."
        )

    manager = NamespaceManager(
        model_name=payload["model_name"],
        M=payload["M"],
        efConstruction=payload["efConstruction"],
    )
    # Reuse the single embedder instance loaded by __init__ — no extra model load.
    shared_embedder = manager._embedder

    for name, engine_payload in payload["namespaces"].items():
        engine = _load_engine_payload(engine_payload, shared_embedder)
        manager._namespaces[name] = engine

    return manager
