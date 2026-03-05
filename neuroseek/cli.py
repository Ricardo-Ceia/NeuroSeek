"""
NeuroSeek CLI — semantic content search from the command line.

Commands
--------
neuroseek index <path> [--namespace NS] [--chunk-size N] [--chunk-overlap N]
    Ingest a file or directory, chunk the text, embed and index every chunk.
    Appends to the existing index (upsert not required — chunks from the same
    file re-indexed will be added as new entries).

neuroseek search "<query>" [--namespace NS] [--top-k N]
    Search the index and print the top-k most relevant chunks.

neuroseek list
    List all namespaces and their document counts.

Index location
--------------
Default store: ~/.neuroseek/index.pkl
Override with the NEUROSEEK_INDEX environment variable or --index flag.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from neuroseek.chunker import chunk_text
from neuroseek.ingestor import ingest_directory, ingest_file
from neuroseek.namespace_manager import NamespaceManager
from neuroseek.namespace_manager_persistence import (
    load_namespace_manager,
    save_namespace_manager,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INDEX_PATH = Path.home() / ".neuroseek" / "index.pkl"
DEFAULT_NAMESPACE = "default"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Index I/O helpers
# ---------------------------------------------------------------------------


def _resolve_index_path(override: str | None) -> Path:
    """Return the resolved index file path.

    Priority: CLI --index flag > NEUROSEEK_INDEX env var > default.
    """
    if override:
        return Path(override)
    env = os.environ.get("NEUROSEEK_INDEX")
    if env:
        return Path(env)
    return DEFAULT_INDEX_PATH


def _load_manager(index_path: Path) -> NamespaceManager:
    """Load an existing NamespaceManager from *index_path*, or create a fresh one."""
    if index_path.exists():
        return load_namespace_manager(index_path)
    return NamespaceManager()


def _save_manager(manager: NamespaceManager, index_path: Path) -> None:
    """Persist *manager* to *index_path*, creating parent directories as needed."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    save_namespace_manager(manager, index_path)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    """Handle ``neuroseek index <path>``."""
    target = Path(args.path)
    index_path = _resolve_index_path(args.index)
    namespace = args.namespace
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    # --- gather (text, metadata) pairs ---
    pairs: list[tuple[str, dict]] = []
    if target.is_dir():
        pairs = ingest_directory(target)
        if not pairs:
            print(f"No supported files found in {str(target)!r}.")
            return 0
    elif target.is_file():
        try:
            pairs = [ingest_file(target)]
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"Error: path does not exist: {str(target)!r}", file=sys.stderr)
        return 1

    # --- chunk ---
    all_chunks: list[str] = []
    all_metadata: list[dict] = []
    for text, file_meta in pairs:
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=file_meta,
        )
        for chunk in chunks:
            chunk_text_str = chunk.pop("text")
            all_chunks.append(chunk_text_str)
            all_metadata.append(chunk)

    if not all_chunks:
        print("No text content found to index.")
        return 0

    # --- load, add, save ---
    manager = _load_manager(index_path)
    manager.add_batch(all_chunks, namespace=namespace, metadata_list=all_metadata)
    _save_manager(manager, index_path)

    print(
        f"Indexed {len(all_chunks)} chunk(s) from {len(pairs)} file(s) "
        f"into namespace {namespace!r}."
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Handle ``neuroseek search "<query>"``."""
    index_path = _resolve_index_path(args.index)
    namespace = args.namespace
    top_k = args.top_k
    query = args.query

    if not index_path.exists():
        print("No index found. Run `neuroseek index <path>` first.", file=sys.stderr)
        return 1

    manager = _load_manager(index_path)

    if namespace not in manager.list_namespaces():
        print(
            f"Namespace {namespace!r} not found. "
            f"Available: {manager.list_namespaces()}",
            file=sys.stderr,
        )
        return 1

    results = manager.search(query, namespace=namespace, top_k=top_k)

    if not results:
        print("No results found.")
        return 0

    for i, result in enumerate(results, start=1):
        meta = result["metadata"]
        source = meta.get("filename", meta.get("path", "unknown"))
        chunk_idx = meta.get("chunk_index", "?")
        score = result["score"]
        text = result["text"]
        print(f"[{i}] {source} (chunk {chunk_idx}) — score: {score:.4f}")
        print(f"    {text[:200]}{'...' if len(text) > 200 else ''}")
        print()

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Handle ``neuroseek list``."""
    index_path = _resolve_index_path(args.index)

    if not index_path.exists():
        print("No index found. Run `neuroseek index <path>` first.")
        return 0

    manager = _load_manager(index_path)
    namespaces = manager.list_namespaces()

    if not namespaces:
        print("Index is empty — no namespaces found.")
        return 0

    print(f"{'Namespace':<30} {'Documents':>10}")
    print("-" * 42)
    for name in namespaces:
        count = manager.namespace_len(name)
        print(f"{name:<30} {count:>10}")

    total = len(manager)
    print("-" * 42)
    print(f"{'TOTAL':<30} {total:>10}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroseek",
        description="Semantic content search engine.",
    )
    parser.add_argument(
        "--index",
        metavar="PATH",
        default=None,
        help=(
            f"Path to the index file "
            f"(default: ~/.neuroseek/index.pkl or $NEUROSEEK_INDEX)"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # -- index --
    p_index = subparsers.add_parser(
        "index",
        help="Ingest and index a file or directory.",
    )
    p_index.add_argument("path", metavar="PATH", help="File or directory to index.")
    p_index.add_argument(
        "--namespace",
        metavar="NS",
        default=DEFAULT_NAMESPACE,
        help=f"Target namespace (default: {DEFAULT_NAMESPACE!r}).",
    )
    p_index.add_argument(
        "--chunk-size",
        metavar="N",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    p_index.add_argument(
        "--chunk-overlap",
        metavar="N",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlapping words between chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    p_index.set_defaults(func=cmd_index)

    # -- search --
    p_search = subparsers.add_parser(
        "search",
        help="Search the index with a natural-language query.",
    )
    p_search.add_argument("query", metavar="QUERY", help="Search query string.")
    p_search.add_argument(
        "--namespace",
        metavar="NS",
        default=DEFAULT_NAMESPACE,
        help=f"Namespace to search (default: {DEFAULT_NAMESPACE!r}).",
    )
    p_search.add_argument(
        "--top-k",
        metavar="N",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K}).",
    )
    p_search.set_defaults(func=cmd_search)

    # -- list --
    p_list = subparsers.add_parser(
        "list",
        help="List all namespaces and document counts.",
    )
    p_list.set_defaults(func=cmd_list)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* and dispatch to the appropriate command.

    Returns the exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
