"""
DocumentStore — lightweight in-memory store that maps integer IDs to raw text.

Design contract:
- IDs are non-negative integers.
- If the caller supplies an explicit id, it is used as-is; duplicates replace the
  existing entry (upsert semantics).
- If no id is supplied, the store auto-assigns the next available integer using an
  internal monotonically-increasing counter that is never decremented, so IDs
  from deleted documents are never re-used.
- `add_batch` is atomic in the sense that it validates *all* inputs before
  persisting any of them — a single bad item leaves the store unchanged.
"""

from __future__ import annotations

from typing import Iterable, Optional


class DocumentStore:
    """In-memory mapping from integer IDs to text documents."""

    def __init__(self) -> None:
        self._store: dict[int, str] = {}
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

    def add(self, text: str, id: Optional[int] = None) -> int:  # noqa: A002
        """Store *text* and return its integer ID.

        Parameters
        ----------
        text:
            The document text. Must be a non-empty, non-whitespace string.
        id:
            Optional explicit ID. If omitted, the next auto-assigned ID is used.
            Providing an existing ID overwrites the previous document (upsert).

        Returns
        -------
        int
            The ID under which the document is stored.
        """
        self._validate_text(text)
        if id is not None:
            self._validate_id(id)
            # Keep _next_id ahead of any manually-supplied id so automatic
            # assignments never collide with explicit ones.
            if id >= self._next_id:
                self._next_id = id + 1
            self._store[id] = text
            return id

        doc_id = self._allocate_id()
        self._store[doc_id] = text
        return doc_id

    def add_batch(
        self,
        texts: Iterable[str],
        ids: Optional[Iterable[int]] = None,
    ) -> list[int]:
        """Store multiple documents atomically and return their IDs.

        Parameters
        ----------
        texts:
            An iterable of document strings.
        ids:
            Optional iterable of explicit IDs, one per text. If provided, the
            two iterables must have the same length.

        Returns
        -------
        list[int]
            IDs in the same order as *texts*.

        Raises
        ------
        TypeError
            If *texts* is not a list or tuple, or if any element is not a str,
            or if any supplied ID is not an int.
        ValueError
            If *texts* is empty, if any element is blank/whitespace, if any
            supplied ID is negative, or if *ids* and *texts* have different
            lengths.
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

        # --- Validate everything before mutating state ---
        for i, text in enumerate(texts_list):
            self._validate_text(text, context=f"index {i}")
        if ids_list is not None:
            for i, doc_id in enumerate(ids_list):
                self._validate_id(doc_id, context=f"index {i}")

        # --- Commit ---
        result: list[int] = []
        for i, text in enumerate(texts_list):
            explicit_id = ids_list[i] if ids_list is not None else None
            result.append(self.add(text, id=explicit_id))
        return result

    def get(self, id: int) -> str:  # noqa: A002
        """Return the document text for *id*.

        Raises
        ------
        TypeError
            If *id* is not an int.
        ValueError
            If *id* is negative.
        KeyError
            If *id* is not in the store.
        """
        self._validate_id(id)
        if id not in self._store:
            raise KeyError(id)
        return self._store[id]

    def delete(self, id: int) -> None:  # noqa: A002
        """Remove the document with *id* from the store.

        Raises
        ------
        TypeError
            If *id* is not an int.
        ValueError
            If *id* is negative.
        KeyError
            If *id* is not in the store.
        """
        self._validate_id(id)
        if id not in self._store:
            raise KeyError(id)
        del self._store[id]

    def __len__(self) -> int:
        """Return the number of documents currently in the store."""
        return len(self._store)

    def __contains__(self, id: object) -> bool:  # noqa: A002
        """Support ``id in store`` syntax."""
        return id in self._store
