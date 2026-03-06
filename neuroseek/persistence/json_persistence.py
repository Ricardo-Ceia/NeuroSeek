"""
JSON export / import for NeuroSeek.

The on-disk schema is::

    {
        "neuroseek_version": "1",
        "namespaces": {
            "<namespace_name>": [
                {"id": <int>, "text": "<str>", "metadata": {<str>: <value>, ...}},
                ...
            ],
            ...
        }
    }

Public API
----------
export_namespace_manager(manager, path, namespace=None)
    Write the index (or a single namespace) to a JSON file.

import_from_json(path, manager, namespace_map=None) -> dict[str, int]
    Read a JSON export and ingest all documents into *manager*, re-embedding
    every chunk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from neuroseek.namespace_manager import NamespaceManager

_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_namespace_manager(
    manager: NamespaceManager,
    path: str | Path,
    namespace: Optional[str] = None,
) -> None:
    """Write *manager* (or a single *namespace*) to a JSON file at *path*.

    Parameters
    ----------
    manager:
        The :class:`~neuroseek.namespace_manager.NamespaceManager` to export.
    path:
        Destination file path.  Parent directories must already exist.
    namespace:
        If given, export only this namespace.  Raises :exc:`KeyError` if the
        namespace doesn't exist.

    Raises
    ------
    KeyError
        If *namespace* is specified but not present in *manager*.
    """
    path = Path(path)

    if namespace is not None:
        # Validate: raises KeyError if missing
        engine = manager._namespaces.get(namespace)
        if engine is None:
            raise KeyError(f"namespace {namespace!r} does not exist")
        namespaces_to_export = [namespace]
    else:
        namespaces_to_export = manager.list_namespaces()

    ns_data: dict[str, list[dict]] = {}
    for ns in namespaces_to_export:
        engine = manager._namespaces[ns]
        ns_data[ns] = engine._store.all_documents()

    payload = {
        "neuroseek_version": _SCHEMA_VERSION,
        "namespaces": ns_data,
    }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_from_json(
    path: str | Path,
    manager: NamespaceManager,
    namespace_map: Optional[dict[str, str]] = None,
) -> dict[str, int]:
    """Read a JSON export and ingest all documents into *manager*.

    Every text chunk is re-embedded using *manager*'s embedder so the vectors
    are consistent with the target index.

    Parameters
    ----------
    path:
        Path to the JSON file produced by :func:`export_namespace_manager`.
    manager:
        Target :class:`~neuroseek.namespace_manager.NamespaceManager`.
    namespace_map:
        Optional dict that remaps source namespace names to target names.
        E.g. ``{"default": "imported"}`` will import the ``"default"``
        namespace from the file into the ``"imported"`` namespace of
        *manager*.  Source names not in the map are used as-is.

    Returns
    -------
    dict[str, int]
        Mapping from target namespace name → number of chunks imported.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON file has an unrecognised ``neuroseek_version``.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    version = payload.get("neuroseek_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported neuroseek_version {version!r} "
            f"(expected {_SCHEMA_VERSION!r})"
        )

    ns_data: dict[str, list[dict]] = payload.get("namespaces", {})
    result: dict[str, int] = {}

    for src_ns, docs in ns_data.items():
        target_ns = (namespace_map or {}).get(src_ns, src_ns)

        if not docs:
            result[target_ns] = 0
            continue

        texts = [d["text"] for d in docs]
        metadata_list = [d.get("metadata") or {} for d in docs]

        manager.add_batch(texts, namespace=target_ns, metadata_list=metadata_list)
        result[target_ns] = len(texts)

    return result
