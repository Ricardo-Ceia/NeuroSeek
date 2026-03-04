"""
Persistence for SearchEngine — save to and load from disk using pickle.

The serialised payload captures everything needed to reconstruct the engine
exactly:
- model_name / M / efConstruction  (constructor params)
- the full HNSWIndex internal state
- the DocumentStore internal state
- _next_id counters for both, so auto-ID assignment continues without gaps
"""

from __future__ import annotations

import pickle
from pathlib import Path

from neuroseek.search_engine import SearchEngine


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

    idx = engine._index
    store = engine._store

    payload = {
        # Constructor params — needed to rebuild Embedder + HNSWIndex
        "model_name": engine.model_name,
        "M": engine.M,
        "efConstruction": engine.efConstruction,
        # HNSWIndex internals
        "hnsw_maxLayers": idx.maxLayers,
        "hnsw_layers": idx.layers,
        "hnsw_id_to_node": idx.id_to_node,
        "hnsw_entry_point_id": idx.entry_point.id if idx.entry_point else None,
        "hnsw_num_vectors": idx.num_vectors,
        "hnsw_next_id": idx._next_id,
        # DocumentStore internals
        "store_data": dict(store._store),
        "store_next_id": store._next_id,
    }

    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_search_engine(path: str | Path) -> SearchEngine:
    """Deserialise a ``SearchEngine`` from *path*.

    The model is **not** re-downloaded; the ``SentenceTransformer`` is
    initialised from the cached weights just like a fresh ``SearchEngine()``.
    The HNSW graph and document store are restored without re-embedding.

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

    # Reconstruct the engine (loads the embedding model)
    engine = SearchEngine(
        model_name=payload["model_name"],
        M=payload["M"],
        efConstruction=payload["efConstruction"],
    )

    # Restore HNSWIndex state
    idx = engine._index
    idx.maxLayers = payload["hnsw_maxLayers"]
    idx.layers = payload["hnsw_layers"]
    idx.id_to_node = payload["hnsw_id_to_node"]
    idx.num_vectors = payload["hnsw_num_vectors"]
    idx._next_id = payload["hnsw_next_id"]
    ep_id = payload["hnsw_entry_point_id"]
    idx.entry_point = idx.id_to_node[ep_id] if ep_id is not None else None

    # Restore DocumentStore state
    store = engine._store
    store._store = payload["store_data"]
    store._next_id = payload["store_next_id"]

    return engine
