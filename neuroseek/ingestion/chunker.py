"""
Chunker — splits a plain-text string into overlapping fixed-size word chunks.

Design contract:
- Chunking is word-based (whitespace-split tokens).
- ``chunk_size``   : number of words per chunk (must be >= 1).
- ``chunk_overlap``: number of words shared between consecutive chunks
                     (must be >= 0 and < chunk_size).
- Each chunk is returned as a dict with keys:
    - ``"text"``        : the reconstructed chunk string (words joined by single space)
    - ``"chunk_index"`` : zero-based position of this chunk in the sequence
  plus any caller-supplied metadata fields merged in (caller fields take
  precedence over the built-in keys if there is a collision — callers are
  responsible for avoiding that).
- If the input text is empty or whitespace-only, an empty list is returned.
- The last chunk may contain fewer than ``chunk_size`` words; it is always
  included as long as it has at least one word.
- Metadata values must follow the same rules as DocumentStore metadata:
  flat dict with str keys and str/int/float/bool values, or None.
"""

from __future__ import annotations

from typing import Optional

from neuroseek.document_store import _validate_metadata


def chunk_text(
    text: str,
    *,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Split *text* into overlapping word chunks and return them as dicts.

    Parameters
    ----------
    text:
        The source text to split.  Empty / whitespace-only input returns ``[]``.
    chunk_size:
        Number of words per chunk.  Must be a positive integer.
    chunk_overlap:
        Number of words carried over from the previous chunk into the next.
        Must be >= 0 and strictly less than ``chunk_size``.
    metadata:
        Optional flat dict (str → str/int/float/bool) merged into every chunk
        dict alongside the built-in ``"text"`` and ``"chunk_index"`` keys.
        ``None`` is treated as an empty dict.

    Returns
    -------
    list[dict]
        Ordered list of chunk dicts.  Each dict contains at minimum:
        ``{"text": str, "chunk_index": int, ...metadata_fields}``.

    Raises
    ------
    TypeError
        If *text* is not a str, or *metadata* fails type validation.
    ValueError
        If *chunk_size* < 1, or *chunk_overlap* < 0, or
        *chunk_overlap* >= *chunk_size*.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError(
            f"chunk_overlap must be an int, got {type(chunk_overlap).__name__}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must be < chunk_size "
            f"(got chunk_overlap={chunk_overlap}, chunk_size={chunk_size})"
        )

    _validate_metadata(metadata)
    extra = dict(metadata) if metadata else {}

    words = text.split()
    if not words:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[dict] = []
    start = 0
    index = 0

    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        entry = {"text": " ".join(chunk_words), "chunk_index": index}
        entry.update(extra)
        chunks.append(entry)
        start += stride
        index += 1

    return chunks
