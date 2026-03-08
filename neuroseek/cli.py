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
import json
import os
import sys
from pathlib import Path

from neuroseek.ingestion.chunker import chunk_text
from neuroseek.ingestion.ingestor import ingest_directory, ingest_file
from neuroseek.namespace_manager import NamespaceManager
from neuroseek.persistence.namespace_manager_persistence import (
    load_namespace_manager,
    save_namespace_manager,
)
from neuroseek.persistence.json_persistence import (
    export_namespace_manager,
    import_from_json,
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

    # --- load manager and detect already-indexed files ---
    manager = _load_manager(index_path)

    if manager.list_namespaces() and namespace in manager.list_namespaces():
        already_indexed = manager.list_sources(namespace, key="filename")
    else:
        already_indexed = set()

    skipped = [
        (text, meta) for text, meta in pairs
        if meta.get("filename") in already_indexed
    ]
    pairs = [
        (text, meta) for text, meta in pairs
        if meta.get("filename") not in already_indexed
    ]

    if skipped:
        skipped_names = sorted({meta.get("filename", "?") for _, meta in skipped})
        for name in skipped_names:
            print(f"Skipping (already indexed): {name}")

    if not pairs:
        print("All files are already indexed. Nothing to do.")
        return 0

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
    ef_search = args.ef_search  # int or None
    filter_str = getattr(args, "filter", None)
    metadata_filter: dict | None = None
    if filter_str:
        try:
            metadata_filter = json.loads(filter_str)
        except json.JSONDecodeError as exc:
            print(f"Error: --filter is not valid JSON: {exc}", file=sys.stderr)
            return 1

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

    results = manager.search(query, namespace=namespace, top_k=top_k, ef_search=ef_search, filter=metadata_filter)

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


def cmd_delete(args: argparse.Namespace) -> int:
    """Handle ``neuroseek delete <filename>`` or ``neuroseek delete --query "<q>"``."""
    index_path = _resolve_index_path(args.index)
    namespace = args.namespace

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

    # ---- validate: must supply exactly one of filename or --query ----
    if not args.query and not args.filename:
        print(
            "Error: provide either a FILENAME positional argument or --query.",
            file=sys.stderr,
        )
        return 1

    # ---- parse --filter (only used with --query) ----
    filter_str = getattr(args, "filter", None)
    metadata_filter: dict | None = None
    if filter_str:
        try:
            metadata_filter = json.loads(filter_str)
        except json.JSONDecodeError as exc:
            print(f"Error: --filter is not valid JSON: {exc}", file=sys.stderr)
            return 1

    # ---- delete by query ----
    if args.query:
        top_k = args.top_k
        deleted = manager.delete_by_query(args.query, namespace=namespace, top_k=top_k, filter=metadata_filter)

        if not deleted:
            print(
                f"No chunks matched query {args.query!r} in namespace {namespace!r}.",
                file=sys.stderr,
            )
            return 1

        print(f"Matched {len(deleted)} chunk(s) for query {args.query!r}:")
        for i, result in enumerate(deleted, start=1):
            meta = result["metadata"]
            source = meta.get("filename", meta.get("path", "unknown"))
            chunk_idx = meta.get("chunk_index", "?")
            score = result["score"]
            text = result["text"]
            print(f"  [{i}] {source} (chunk {chunk_idx}) — score: {score:.4f}")
            print(f"      {text[:120]}{'...' if len(text) > 120 else ''}")

        if getattr(args, "dry_run", False):
            print("Dry run — no changes written.")
            return 0

        _save_manager(manager, index_path)
        print(f"Deleted {len(deleted)} chunk(s) from namespace {namespace!r}.")
        return 0

    # ---- delete by filename ----
    filename = args.filename
    deleted_count = manager.delete_source(filename, namespace=namespace)

    if deleted_count == 0:
        print(
            f"No chunks found for {filename!r} in namespace {namespace!r}.",
            file=sys.stderr,
        )
        return 1

    _save_manager(manager, index_path)
    print(f"Deleted {deleted_count} chunk(s) for {filename!r} from namespace {namespace!r}.")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Handle ``neuroseek update <path>``."""
    target = Path(args.path)
    index_path = _resolve_index_path(args.index)
    namespace = args.namespace
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    if not index_path.exists():
        print("No index found. Run `neuroseek index <path>` first.", file=sys.stderr)
        return 1

    if not target.is_file():
        print(f"Error: path is not a file: {str(target)!r}", file=sys.stderr)
        return 1

    manager = _load_manager(index_path)

    if namespace not in manager.list_namespaces():
        print(
            f"Namespace {namespace!r} not found. "
            f"Available: {manager.list_namespaces()}",
            file=sys.stderr,
        )
        return 1

    try:
        text, file_meta = ingest_file(target)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    filename = file_meta.get("filename", target.name)

    chunks_data = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=file_meta,
    )
    chunk_texts = []
    chunk_metas = []
    for chunk in chunks_data:
        chunk_texts.append(chunk.pop("text"))
        chunk_metas.append(chunk)

    if not chunk_texts:
        print("No text content found to index.")
        return 0

    count = manager.update_source(filename, namespace, chunk_texts, metadata_list=chunk_metas)
    _save_manager(manager, index_path)
    print(
        f"Updated {filename!r}: indexed {count} chunk(s) into namespace {namespace!r}."
    )
    return 0


def cmd_list_sources(args: argparse.Namespace) -> int:
    """Handle ``neuroseek list-sources [--namespace NS]``."""
    index_path = _resolve_index_path(args.index)
    namespace = args.namespace

    if not index_path.exists():
        print("No index found. Run `neuroseek index <path>` first.")
        return 0

    manager = _load_manager(index_path)

    if namespace not in manager.list_namespaces():
        print(
            f"Namespace {namespace!r} not found. "
            f"Available: {manager.list_namespaces()}",
            file=sys.stderr,
        )
        return 1

    sources = sorted(manager.list_sources(namespace))

    if not sources:
        print(f"No sources found in namespace {namespace!r}.")
        return 0

    for source in sources:
        print(source)
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


def cmd_export(args: argparse.Namespace) -> int:
    """Handle ``neuroseek export <output.json> [--namespace NS]``."""
    index_path = _resolve_index_path(args.index)
    output_path = Path(args.output)
    namespace = getattr(args, "namespace", None)

    if not index_path.exists():
        print("No index found. Run `neuroseek index <path>` first.", file=sys.stderr)
        return 1

    manager = _load_manager(index_path)

    if namespace is not None and namespace not in manager.list_namespaces():
        print(
            f"Namespace {namespace!r} not found. "
            f"Available: {manager.list_namespaces()}",
            file=sys.stderr,
        )
        return 1

    try:
        export_namespace_manager(manager, output_path, namespace=namespace)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Count total chunks exported
    if namespace is not None:
        ns_list = [namespace]
    else:
        ns_list = manager.list_namespaces()

    total_chunks = sum(manager.namespace_len(ns) for ns in ns_list)
    print(
        f"Exported {total_chunks} chunk(s) from {len(ns_list)} namespace(s) "
        f"to {str(output_path)!r}."
    )
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Handle ``neuroseek import <input.json> [--namespace NS]``."""
    index_path = _resolve_index_path(args.index)
    input_path = Path(args.input)
    target_namespace = getattr(args, "namespace", None)

    manager = _load_manager(index_path)

    # Build namespace_map if --namespace was given: every source namespace
    # is redirected to the single target namespace.
    namespace_map: dict[str, str] | None = None
    if target_namespace is not None:
        import json as _json
        try:
            with input_path.open("r", encoding="utf-8") as fh:
                payload = _json.load(fh)
            src_namespaces = list(payload.get("namespaces", {}).keys())
        except FileNotFoundError:
            print(f"Error: file not found: {str(input_path)!r}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error reading JSON: {exc}", file=sys.stderr)
            return 1
        namespace_map = {ns: target_namespace for ns in src_namespaces}

    try:
        counts = import_from_json(input_path, manager, namespace_map=namespace_map)
    except FileNotFoundError:
        print(f"Error: file not found: {str(input_path)!r}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not counts:
        print("Nothing imported (empty file).")
        return 0

    _save_manager(manager, index_path)
    for ns, count in sorted(counts.items()):
        print(f"Imported {count} chunk(s) into namespace {ns!r}.")
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
    p_search.add_argument(
        "--ef-search",
        metavar="N",
        type=int,
        default=None,
        dest="ef_search",
        help=(
            "HNSW search beam width. Higher values improve recall at the cost "
            "of latency. Defaults to the index's efConstruction value."
        ),
    )
    p_search.add_argument(
        "--filter",
        metavar="JSON",
        default=None,
        dest="filter",
        help=(
            "Metadata filter as a JSON object (e.g. '{\"category\": \"science\"}')."
            " Only chunks whose metadata contains all key-value pairs are returned."
        ),
    )
    p_search.set_defaults(func=cmd_search)

    # -- list --
    p_list = subparsers.add_parser(
        "list",
        help="List all namespaces and document counts.",
    )
    p_list.set_defaults(func=cmd_list)

    # -- delete --
    p_delete = subparsers.add_parser(
        "delete",
        help="Remove chunks by filename or by semantic query.",
    )
    p_delete.add_argument(
        "filename",
        metavar="FILENAME",
        nargs="?",
        default=None,
        help="Filename to delete (as stored in metadata, e.g. 'report.txt').",
    )
    p_delete.add_argument(
        "--query",
        metavar="QUERY",
        default=None,
        help="Delete chunks that semantically match this query instead of a filename.",
    )
    p_delete.add_argument(
        "--top-k",
        metavar="N",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Max chunks to delete when using --query (default: {DEFAULT_TOP_K}).",
    )
    p_delete.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print matching chunks but do not delete them (only with --query).",
    )
    p_delete.add_argument(
        "--namespace",
        metavar="NS",
        default=DEFAULT_NAMESPACE,
        help=f"Namespace to delete from (default: {DEFAULT_NAMESPACE!r}).",
    )
    p_delete.add_argument(
        "--filter",
        metavar="JSON",
        default=None,
        dest="filter",
        help=(
            "Metadata filter as a JSON object (e.g. '{\"category\": \"science\"}')."
            " Only chunks whose metadata matches are considered for deletion (only with --query)."
        ),
    )
    p_delete.set_defaults(func=cmd_delete)

    # -- update --
    p_update = subparsers.add_parser(
        "update",
        help="Re-index a file, replacing its existing chunks.",
    )
    p_update.add_argument("path", metavar="PATH", help="File to re-index.")
    p_update.add_argument(
        "--namespace",
        metavar="NS",
        default=DEFAULT_NAMESPACE,
        help=f"Namespace to update (default: {DEFAULT_NAMESPACE!r}).",
    )
    p_update.add_argument(
        "--chunk-size",
        metavar="N",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    p_update.add_argument(
        "--chunk-overlap",
        metavar="N",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlapping words between chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    p_update.set_defaults(func=cmd_update)

    # -- list-sources --
    p_list_sources = subparsers.add_parser(
        "list-sources",
        help="List all indexed source filenames in a namespace.",
    )
    p_list_sources.add_argument(
        "--namespace",
        metavar="NS",
        default=DEFAULT_NAMESPACE,
        help=f"Namespace to inspect (default: {DEFAULT_NAMESPACE!r}).",
    )
    p_list_sources.set_defaults(func=cmd_list_sources)

    # -- export --
    p_export = subparsers.add_parser(
        "export",
        help="Export the index (or a namespace) to a JSON file.",
    )
    p_export.add_argument(
        "output",
        metavar="OUTPUT",
        help="Destination JSON file path.",
    )
    p_export.add_argument(
        "--namespace",
        metavar="NS",
        default=None,
        help="Export only this namespace (default: export all).",
    )
    p_export.set_defaults(func=cmd_export)

    # -- import --
    p_import = subparsers.add_parser(
        "import",
        help="Import chunks from a JSON export file into the index.",
    )
    p_import.add_argument(
        "input",
        metavar="INPUT",
        help="Source JSON file path.",
    )
    p_import.add_argument(
        "--namespace",
        metavar="NS",
        default=None,
        help=(
            "Import all data into this single target namespace "
            "(default: use source namespace names)."
        ),
    )
    p_import.set_defaults(func=cmd_import)

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
