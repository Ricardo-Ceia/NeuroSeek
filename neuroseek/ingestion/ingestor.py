"""
Ingestor — reads plain-text content from files and directories.

Design contract:
- ``ingest_file(path)`` reads a single file and returns its text content as a
  string, together with a metadata dict describing the file.
- ``ingest_directory(path)`` walks a directory recursively and returns a list
  of (text, metadata) pairs, one per supported file found.
- Supported extensions: .txt, .md, .py, .json, .csv
  (binary formats such as .pdf and .docx are explicitly out of scope for now).
- Files are always read as UTF-8.  If a file cannot be decoded as UTF-8 it is
  skipped with a warning (directory ingestion) or raises UnicodeDecodeError
  (single-file ingestion).
- Metadata returned per file:
    {
        "path":     absolute path string,
        "filename": basename of the file,
        "filetype": lowercase extension without the leading dot (e.g. "txt"),
    }
- Unsupported extensions raise ``ValueError`` for single-file ingestion and
  are silently skipped in directory ingestion.
- Non-existent paths raise ``FileNotFoundError``.
- Passing a directory path to ``ingest_file`` raises ``IsADirectoryError``.
- Passing a file path to ``ingest_directory`` raises ``NotADirectoryError``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {"txt", "md", "py", "json", "csv"}
)


def _file_metadata(path: Path) -> dict:
    """Build the standard metadata dict for a file path."""
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "filetype": path.suffix.lstrip(".").lower(),
    }


def ingest_file(path: str | Path) -> tuple[str, dict]:
    """Read a single supported file and return ``(text, metadata)``.

    Parameters
    ----------
    path:
        Path to the file to ingest.

    Returns
    -------
    tuple[str, dict]
        ``(text_content, metadata)`` where *metadata* contains
        ``"path"``, ``"filename"``, and ``"filetype"`` keys.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    IsADirectoryError
        If *path* is a directory.
    ValueError
        If the file extension is not in ``SUPPORTED_EXTENSIONS``.
    UnicodeDecodeError
        If the file content cannot be decoded as UTF-8.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"No such file: {str(p)!r}")
    if p.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {str(p)!r}")

    ext = p.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension {p.suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    text = p.read_text(encoding="utf-8")
    return text, _file_metadata(p)


def ingest_directory(
    path: str | Path,
    *,
    extensions: Optional[frozenset[str]] = None,
) -> list[tuple[str, dict]]:
    """Recursively ingest all supported files under *path*.

    Parameters
    ----------
    path:
        Root directory to walk.
    extensions:
        Optional subset of ``SUPPORTED_EXTENSIONS`` to restrict ingestion to.
        Defaults to all supported extensions.

    Returns
    -------
    list[tuple[str, dict]]
        Ordered list of ``(text, metadata)`` pairs, sorted by absolute path
        for deterministic output.  Files that cannot be decoded as UTF-8 are
        skipped with a ``UserWarning``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    NotADirectoryError
        If *path* is a file, not a directory.
    ValueError
        If *extensions* contains a value not in ``SUPPORTED_EXTENSIONS``.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"No such directory: {str(p)!r}")
    if not p.is_dir():
        raise NotADirectoryError(f"Path is a file, not a directory: {str(p)!r}")

    allowed = extensions if extensions is not None else SUPPORTED_EXTENSIONS
    invalid = allowed - SUPPORTED_EXTENSIONS
    if invalid:
        raise ValueError(
            f"Unsupported extension(s): {sorted(invalid)}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    results: list[tuple[str, dict]] = []

    for file_path in sorted(p.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lstrip(".").lower()
        if ext not in allowed:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.warn(
                f"Skipping {str(file_path)!r}: could not decode as UTF-8",
                UserWarning,
                stacklevel=2,
            )
            continue
        results.append((text, _file_metadata(file_path)))

    return results
