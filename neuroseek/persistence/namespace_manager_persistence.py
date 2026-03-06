"""
Persistence for NamespaceManager — save all namespaces to a single file
and restore them without re-embedding any documents.

Format: a pickle file containing a list of per-namespace payloads, each
identical in structure to what save_search_engine / load_search_engine use,
plus the top-level NamespaceManager constructor params.

The payload always includes a ``"persistence_version"`` key whose value is
:data:`PERSISTENCE_VERSION`.  :func:`load_namespace_manager` raises a clear
:exc:`ValueError` if the stored version is absent or does not match, so users
see a useful error instead of a cryptic ``KeyError`` or attribute crash.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from neuroseek.namespace_manager import NamespaceManager
from neuroseek.search_engine import SearchEngine
from neuroseek.embedder import Embedder

#: Current binary persistence format version.  Bump this whenever the pickle
#: schema changes in a backwards-incompatible way.
PERSISTENCE_VERSION = "1"


def save_namespace_manager(manager: NamespaceManager, path: str | Path) -> None:
    """Serialise *manager* (all namespaces) to *path*.

    Parameters
    ----------
    manager:
        A :class:`NamespaceManager` instance to persist.
    path:
        Destination file path (created or overwritten).

    Raises
    ------
    TypeError
        If *manager* is not a :class:`NamespaceManager`.
    """
    if not isinstance(manager, NamespaceManager):
        raise TypeError(
            f"manager must be a NamespaceManager, got {type(manager).__name__}"
        )

    namespaces_payload: dict[str, dict] = {}
    for name, engine in manager._namespaces.items():
        # Reuse save_search_engine by capturing its output in memory
        idx = engine._index
        store = engine._store
        engine_payload = {
            "model_name": engine.model_name,
            "M": engine.M,
            "efConstruction": engine.efConstruction,
            "hnsw_maxLayers": idx.maxLayers,
            "hnsw_layers": idx.layers,
            "hnsw_id_to_node": idx.id_to_node,
            "hnsw_entry_point_id": idx.entry_point.id if idx.entry_point else None,
            "hnsw_num_vectors": idx.num_vectors,
            "hnsw_next_id": idx._next_id,
            "store_data": dict(store._store),
            "store_next_id": store._next_id,
        }
        namespaces_payload[name] = engine_payload

    payload = {
        "persistence_version": PERSISTENCE_VERSION,
        "model_name": manager.model_name,
        "M": manager.M,
        "efConstruction": manager.efConstruction,
        "namespaces": namespaces_payload,
    }

    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_namespace_manager(path: str | Path) -> NamespaceManager:
    """Deserialise a :class:`NamespaceManager` from *path*.

    Parameters
    ----------
    path:
        File previously written by :func:`save_namespace_manager`.

    Returns
    -------
    NamespaceManager
        Fully restored manager with all namespaces and documents intact.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
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
    # manager._embedder was loaded once during NamespaceManager.__init__ above.
    # Reuse it for all namespace engines so the model is only loaded once.
    shared_embedder = manager._embedder

    for name, engine_payload in payload["namespaces"].items():
        # Build a SearchEngine reusing the shared embedder (no extra model load)
        engine = SearchEngine._from_embedder(
            embedder=shared_embedder,
            M=engine_payload["M"],
            efConstruction=engine_payload["efConstruction"],
        )
        idx = engine._index
        idx.maxLayers = engine_payload["hnsw_maxLayers"]
        idx.layers = engine_payload["hnsw_layers"]
        idx.id_to_node = engine_payload["hnsw_id_to_node"]
        idx.num_vectors = engine_payload["hnsw_num_vectors"]
        idx._next_id = engine_payload["hnsw_next_id"]
        ep_id = engine_payload["hnsw_entry_point_id"]
        idx.entry_point = idx.id_to_node[ep_id] if ep_id is not None else None

        store = engine._store
        store._store = engine_payload["store_data"]
        store._next_id = engine_payload["store_next_id"]

        manager._namespaces[name] = engine

    return manager
