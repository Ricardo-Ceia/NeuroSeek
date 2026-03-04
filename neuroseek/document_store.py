"""
DocumentStore — lightweight in-memory store that maps integer IDs to text
documents and optional key-value metadata.

Design contract:
- IDs are non-negative integers.
- If the caller supplies an explicit id, it is used as-is; duplicates replace
  the existing entry (upsert semantics).
- If no id is supplied, the store auto-assigns the next available integer using
  an internal monotonically-increasing counter that is never decremented, so
  IDs from deleted documents are never re-used.
- `add_batch` is atomic in the sense that it validates *all* inputs before
  persisting any of them — a single bad item leaves the store unchanged.
- metadata must be a flat dict with str keys and str/int/float/bool values,
  or None (treated as an empty dict).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


# Types allowed as metadata values
_METADATA_VALUE_TYPES = (str, int, float, bool)


def _validate_metadata(meta: object, context: str = "") -> None:
    prefix = f"{context}: " if context else ""
    if meta is None:
        return
    if not isinstance(meta, dict):
        raise TypeError(
            f"{prefix}metadata must be a dict or None, got {type(meta).__name__}"
        )
    for k, v in meta.items():
        if not isinstance(k, str):
            raise TypeError(
                f"{prefix}metadata keys must be str, got {type(k).__name__!r}"
            )
        if not isinstance(v, _METADATA_VALUE_TYPES):
            raise TypeError(
                f"{prefix}metadata values must be str, int, float, or bool, "
                f"got {type(v).__name__!r} for key {k!r}"
            )


class DocumentStore:
    """In-memory mapping from integer IDs to text documents with metadata."""

    def __init__(self) -> None:
        # Each entry: {"text": str, "metadata": dict}
        self._store: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_text(self, text: object, context: str = "") -> None:
        if not isinstance(text, str):
            prefix = f"{context}: " if context else ""
            raise TypeError(f"{prefix}text must be a str, got {type(text).__name__}")
        if not text.strip():
            prefix = f"{context}: " if context else ""
            raise ValueError(f"{prefix}text must not be empty or whitespace-only")

    def _validate_id(self, doc_id: object, context: str = "") -> None:
        if not isinstance(doc_id, int):
            prefix = f"{context}: " if context else ""
            raise TypeError(
                f"{prefix}id must be an int, got {type(doc_id).__name__}"
            )
        if doc_id < 0:
            prefix = f"{context}: " if context else ""
            raise ValueError(f"{prefix}id must be >= 0, got {doc_id}")

    def _allocate_id(self) -> int:
        doc_id = self._next_id
        self._next_id += 1
        return doc_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        id: Optional[int] = None,  # noqa: A002
        metadata: Optional[dict] = None,
    ) -> int:
        """Store *text* (with optional *metadata*) and return its integer ID.

        Parameters
        ----------
        text:
            The document text. Must be a non-empty, non-whitespace string.
        id:
            Optional explicit ID. If omitted, the next auto-assigned ID is used.
            Providing an existing ID overwrites the previous document (upsert).
        metadata:
            Optional flat dict of str → str/int/float/bool pairs.

        Returns
        -------
        int
            The ID under which the document is stored.
        """
        self._validate_text(text)
        _validate_metadata(metadata)
        meta = dict(metadata) if metadata else {}

        if id is not None:
            self._validate_id(id)
            if id >= self._next_id:
                self._next_id = id + 1
            self._store[id] = {"text": text, "metadata": meta}
            return id

        doc_id = self._allocate_id()
        self._store[doc_id] = {"text": text, "metadata": meta}
        return doc_id

    def add_batch(
        self,
        texts: Iterable[str],
        ids: Optional[Iterable[int]] = None,
        metadata_list: Optional[list[Optional[dict]]] = None,
    ) -> list[int]:
        """Store multiple documents atomically and return their IDs.

        Parameters
        ----------
        texts:
            An iterable of document strings.
        ids:
            Optional iterable of explicit IDs, one per text.
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

        texts_list: list[str] = list(texts)
        ids_list: Optional[list[int]] = None

        if ids is not None:
            if not isinstance(ids, (list, tuple)):
                raise TypeError(
                    f"ids must be a list or tuple, got {type(ids).__name__}"
                )
            ids_list = list(ids)
            if len(ids_list) != len(texts_list):
                raise ValueError(
                    f"ids and texts must have the same length "
                    f"(got {len(ids_list)} ids and {len(texts_list)} texts)"
                )

        if metadata_list is not None:
            if not isinstance(metadata_list, (list, tuple)):
                raise TypeError(
                    f"metadata_list must be a list or tuple, "
                    f"got {type(metadata_list).__name__}"
                )
            if len(metadata_list) != len(texts_list):
                raise ValueError(
                    f"metadata_list and texts must have the same length "
                    f"(got {len(metadata_list)} and {len(texts_list)})"
                )

        # --- Validate everything before mutating state ---
        for i, text in enumerate(texts_list):
            self._validate_text(text, context=f"index {i}")
        if ids_list is not None:
            for i, doc_id in enumerate(ids_list):
                self._validate_id(doc_id, context=f"index {i}")
        if metadata_list is not None:
            for i, meta in enumerate(metadata_list):
                _validate_metadata(meta, context=f"index {i}")

        # --- Commit ---
        result: list[int] = []
        for i, text in enumerate(texts_list):
            explicit_id = ids_list[i] if ids_list is not None else None
            meta = metadata_list[i] if metadata_list is not None else None
            result.append(self.add(text, id=explicit_id, metadata=meta))
        return result

    def get(self, id: int) -> str:  # noqa: A002
        """Return the document text for *id*.

        Raises
        ------
        KeyError
            If *id* is not in the store.
        """
        self._validate_id(id)
        if id not in self._store:
            raise KeyError(id)
        return self._store[id]["text"]

    def get_metadata(self, id: int) -> dict:  # noqa: A002
        """Return the metadata dict for *id* (empty dict if none was set).

        Raises
        ------
        KeyError
            If *id* is not in the store.
        """
        self._validate_id(id)
        if id not in self._store:
            raise KeyError(id)
        return dict(self._store[id]["metadata"])

    def delete(self, id: int) -> None:  # noqa: A002
        """Remove the document with *id* from the store."""
        self._validate_id(id)
        if id not in self._store:
            raise KeyError(id)
        del self._store[id]

    def matches_filter(self, id: int, filter: Optional[dict]) -> bool:  # noqa: A002
        """Return True if the document's metadata satisfies *filter*.

        A document matches if every key-value pair in *filter* is present and
        equal in the document's metadata.  ``None`` or empty filter matches
        everything.
        """
        if not filter:
            return True
        if id not in self._store:
            return False
        meta = self._store[id]["metadata"]
        return all(meta.get(k) == v for k, v in filter.items())

    def __len__(self) -> int:
        """Return the number of documents currently in the store."""
        return len(self._store)

    def __contains__(self, id: object) -> bool:  # noqa: A002
        """Support ``id in store`` syntax."""
        return id in self._store
