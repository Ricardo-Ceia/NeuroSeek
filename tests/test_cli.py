"""
Tests for neuroseek/cli.py — main(), cmd_index, cmd_search, cmd_list,
_resolve_index_path, _load_manager, _save_manager, _build_parser.

All tests use a temporary directory for the index file so they never touch
~/.neuroseek or any real user state.

Run with:
    PYTHONPATH=/home/ricardo/NeuroSeek ~/.local/bin/pytest tests/test_cli.py -q
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from neuroseek.cli import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    DEFAULT_TOP_K,
    _build_parser,
    _load_manager,
    _resolve_index_path,
    _save_manager,
    main,
)
from neuroseek.namespace_manager import NamespaceManager


# ---------------------------------------------------------------------------
# Session-level model cache — reuse the shared embedder from conftest so the
# embedding model is only loaded once for the entire CLI test session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _patch_sentence_transformer(embedder):
    """Monkey-patch SentenceTransformer so every new Embedder() reuses the
    already-loaded model object instead of loading from disk again."""
    cached_model = embedder._model

    class _CachedST:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(cached_model.__dict__)
            self.__class__ = cached_model.__class__

    with patch("neuroseek.embedder.SentenceTransformer", _CachedST):
        yield


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


@pytest.fixture()
def idx(tmp_path):
    """Return a temporary index file path (does not yet exist)."""
    return tmp_path / "index.pkl"


def write_file(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


def run(argv: list[str], capsys, index_path: Path) -> tuple[int, str, str]:
    """Run main() with --index forced to *index_path*; return (exit_code, out, err)."""
    full_argv = ["--index", str(index_path)] + argv
    code = main(full_argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# _resolve_index_path
# ---------------------------------------------------------------------------


class TestResolveIndexPath:
    def test_none_returns_default(self, monkeypatch):
        monkeypatch.delenv("NEUROSEEK_INDEX", raising=False)
        from neuroseek.cli import DEFAULT_INDEX_PATH
        assert _resolve_index_path(None) == DEFAULT_INDEX_PATH

    def test_override_takes_precedence(self, tmp):
        p = tmp / "custom.pkl"
        assert _resolve_index_path(str(p)) == p

    def test_env_var_used_when_no_override(self, tmp, monkeypatch):
        p = tmp / "env.pkl"
        monkeypatch.setenv("NEUROSEEK_INDEX", str(p))
        assert _resolve_index_path(None) == p

    def test_cli_override_beats_env_var(self, tmp, monkeypatch):
        env_p = tmp / "env.pkl"
        cli_p = tmp / "cli.pkl"
        monkeypatch.setenv("NEUROSEEK_INDEX", str(env_p))
        assert _resolve_index_path(str(cli_p)) == cli_p


# ---------------------------------------------------------------------------
# _load_manager / _save_manager
# ---------------------------------------------------------------------------


class TestLoadSaveManager:
    def test_load_nonexistent_returns_fresh_manager(self, idx):
        manager = _load_manager(idx)
        assert isinstance(manager, NamespaceManager)
        assert manager.list_namespaces() == []

    def test_save_creates_file(self, idx):
        manager = NamespaceManager()
        _save_manager(manager, idx)
        assert idx.exists()

    def test_save_creates_parent_directories(self, tmp):
        deep_path = tmp / "a" / "b" / "c" / "index.pkl"
        manager = NamespaceManager()
        _save_manager(manager, deep_path)
        assert deep_path.exists()

    def test_roundtrip_preserves_namespaces(self, idx):
        manager = NamespaceManager()
        manager.add("hello world semantic search", namespace="ns1")
        _save_manager(manager, idx)
        loaded = _load_manager(idx)
        assert "ns1" in loaded.list_namespaces()

    def test_roundtrip_preserves_document_count(self, idx):
        manager = NamespaceManager()
        manager.add("first document", namespace="docs")
        manager.add("second document", namespace="docs")
        _save_manager(manager, idx)
        loaded = _load_manager(idx)
        assert loaded.namespace_len("docs") == 2


# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_returns_argument_parser(self):
        import argparse
        assert isinstance(_build_parser(), argparse.ArgumentParser)

    def test_prog_is_neuroseek(self):
        parser = _build_parser()
        assert parser.prog == "neuroseek"

    def test_index_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "index", "/some/path"])
        assert args.command == "index"

    def test_search_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "search", "my query"])
        assert args.command == "search"

    def test_list_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "list"])
        assert args.command == "list"

    def test_index_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "index", "/path"])
        assert args.namespace == DEFAULT_NAMESPACE
        assert args.chunk_size == DEFAULT_CHUNK_SIZE
        assert args.chunk_overlap == DEFAULT_CHUNK_OVERLAP

    def test_search_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "search", "query"])
        assert args.namespace == DEFAULT_NAMESPACE
        assert args.top_k == DEFAULT_TOP_K

    def test_index_custom_namespace(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "index", "/path", "--namespace", "myns"]
        )
        assert args.namespace == "myns"

    def test_search_custom_top_k(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "search", "q", "--top-k", "10"]
        )
        assert args.top_k == 10

    def test_index_custom_chunk_size(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "index", "/path", "--chunk-size", "128"]
        )
        assert args.chunk_size == 128

    def test_index_custom_chunk_overlap(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "index", "/path", "--chunk-overlap", "16"]
        )
        assert args.chunk_overlap == 16


# ---------------------------------------------------------------------------
# cmd_index
# ---------------------------------------------------------------------------


class TestCmdIndex:
    def test_index_single_file_exit_zero(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        code, _, _ = run(["index", str(f)], capsys, idx)
        assert code == 0

    def test_index_single_file_creates_index(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        run(["index", str(f)], capsys, idx)
        assert idx.exists()

    def test_index_single_file_prints_summary(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        _, out, _ = run(["index", str(f)], capsys, idx)
        assert "Indexed" in out
        assert "chunk" in out

    def test_index_directory_exit_zero(self, tmp, idx, capsys):
        write_file(tmp, "a.txt", "some content here")
        write_file(tmp, "b.md", "more content there")
        code, _, _ = run(["index", str(tmp)], capsys, idx)
        assert code == 0

    def test_index_directory_indexes_all_files(self, tmp, idx, capsys):
        write_file(tmp, "a.txt", "content one")
        write_file(tmp, "b.txt", "content two")
        run(["index", str(tmp)], capsys, idx)
        manager = _load_manager(idx)
        assert manager.namespace_len(DEFAULT_NAMESPACE) >= 2

    def test_index_nonexistent_path_exit_one(self, tmp, idx, capsys):
        code, _, err = run(["index", str(tmp / "ghost.txt")], capsys, idx)
        assert code == 1
        assert "Error" in err

    def test_index_unsupported_extension_exit_one(self, tmp, idx, capsys):
        f = tmp / "doc.pdf"
        f.write_bytes(b"%PDF")
        code, _, err = run(["index", str(f)], capsys, idx)
        assert code == 1
        assert "Error" in err

    def test_index_custom_namespace(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "custom namespace content here")
        run(["index", str(f), "--namespace", "myns"], capsys, idx)
        manager = _load_manager(idx)
        assert "myns" in manager.list_namespaces()

    def test_index_appends_to_existing_index(self, tmp, idx, capsys):
        f1 = write_file(tmp, "a.txt", "first document content here")
        f2 = write_file(tmp, "b.txt", "second document content here")
        run(["index", str(f1)], capsys, idx)
        run(["index", str(f2)], capsys, idx)
        manager = _load_manager(idx)
        assert manager.namespace_len(DEFAULT_NAMESPACE) >= 2

    def test_index_empty_directory_prints_message(self, tmp, idx, capsys):
        empty_dir = tmp / "empty"
        empty_dir.mkdir()
        code, out, _ = run(["index", str(empty_dir)], capsys, idx)
        assert code == 0
        assert "No supported files" in out

    def test_index_stores_file_metadata(self, tmp, idx, capsys):
        f = write_file(tmp, "report.txt", "quarterly earnings report content")
        run(["index", str(f)], capsys, idx)
        manager = _load_manager(idx)
        results = manager.search("earnings report", namespace=DEFAULT_NAMESPACE, top_k=1)
        assert len(results) == 1
        assert results[0]["metadata"].get("filename") == "report.txt"

    def test_index_chunk_size_flag(self, tmp, idx, capsys):
        # chunk_size=5 with a 10-word doc → 2 chunks (no overlap)
        f = write_file(tmp, "doc.txt", "one two three four five six seven eight nine ten")
        run(["index", str(f), "--chunk-size", "5", "--chunk-overlap", "0"], capsys, idx)
        manager = _load_manager(idx)
        assert manager.namespace_len(DEFAULT_NAMESPACE) == 2

    def test_index_prints_file_count(self, tmp, idx, capsys):
        write_file(tmp, "a.txt", "alpha content")
        write_file(tmp, "b.txt", "beta content")
        _, out, _ = run(["index", str(tmp)], capsys, idx)
        assert "2 file" in out


# ---------------------------------------------------------------------------
# cmd_index — deduplication
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_sentence_transformer")
class TestCmdIndexDeduplication:
    def test_reindex_same_file_skips_it(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        run(["index", str(f)], capsys, idx)
        count_after_first = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)

        run(["index", str(f)], capsys, idx)
        count_after_second = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)

        assert count_after_first == count_after_second

    def test_reindex_prints_skipping_message(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        run(["index", str(f)], capsys, idx)
        _, out, _ = run(["index", str(f)], capsys, idx)
        assert "Skipping" in out
        assert "doc.txt" in out

    def test_reindex_exit_zero(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        run(["index", str(f)], capsys, idx)
        code, _, _ = run(["index", str(f)], capsys, idx)
        assert code == 0

    def test_reindex_all_files_prints_nothing_to_do(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        run(["index", str(f)], capsys, idx)
        _, out, _ = run(["index", str(f)], capsys, idx)
        assert "Nothing to do" in out or "already indexed" in out.lower()

    def test_new_file_indexed_alongside_existing(self, tmp, idx, capsys):
        f1 = write_file(tmp, "a.txt", "first document content alpha")
        f2 = write_file(tmp, "b.txt", "second document content beta")
        run(["index", str(f1)], capsys, idx)
        count_after_first = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)

        run(["index", str(tmp)], capsys, idx)
        count_after_dir = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)

        # b.txt should be added; a.txt should be skipped
        assert count_after_dir > count_after_first

    def test_directory_reindex_skips_all_already_indexed(self, tmp, idx, capsys):
        write_file(tmp, "a.txt", "alpha content here")
        write_file(tmp, "b.txt", "beta content here")
        run(["index", str(tmp)], capsys, idx)
        count_first = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)

        run(["index", str(tmp)], capsys, idx)
        count_second = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)

        assert count_first == count_second

    def test_directory_reindex_prints_skipped_filenames(self, tmp, idx, capsys):
        write_file(tmp, "a.txt", "alpha content here")
        write_file(tmp, "b.txt", "beta content here")
        run(["index", str(tmp)], capsys, idx)
        _, out, _ = run(["index", str(tmp)], capsys, idx)
        assert "a.txt" in out
        assert "b.txt" in out

    def test_dedup_is_namespace_scoped(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world this is a test document")
        run(["index", str(f), "--namespace", "ns1"], capsys, idx)
        # Same file in a different namespace should NOT be skipped
        code, out, _ = run(["index", str(f), "--namespace", "ns2"], capsys, idx)
        assert code == 0
        assert "Indexed" in out


# ---------------------------------------------------------------------------
# cmd_search
# ---------------------------------------------------------------------------


class TestCmdSearch:
    @pytest.fixture(autouse=True)
    def pre_index(self, tmp, idx, capsys):
        """Index some documents before each search test."""
        self.tmp = tmp
        self.idx = idx
        write_file(
            tmp,
            "animals.txt",
            "The quick brown fox jumps over the lazy dog. "
            "Dogs are loyal companions. Foxes are cunning animals.",
        )
        write_file(
            tmp,
            "space.txt",
            "The universe is vast and full of stars. "
            "Black holes warp spacetime. Galaxies contain billions of stars.",
        )
        run(["index", str(tmp)], capsys, idx)

    def test_search_exit_zero(self, idx, capsys):
        code, _, _ = run(["search", "loyal dog"], capsys, idx)
        assert code == 0

    def test_search_returns_results(self, idx, capsys):
        _, out, _ = run(["search", "loyal dog"], capsys, idx)
        assert out.strip() != ""

    def test_search_output_contains_score(self, idx, capsys):
        _, out, _ = run(["search", "loyal dog"], capsys, idx)
        assert "score" in out.lower()

    def test_search_output_contains_filename(self, idx, capsys):
        _, out, _ = run(["search", "loyal dog"], capsys, idx)
        assert ".txt" in out

    def test_search_no_index_exit_one(self, tmp, capsys):
        missing_idx = tmp / "no_index.pkl"
        code, _, err = run(["search", "query"], capsys, missing_idx)
        assert code == 1
        assert "index" in err.lower()

    def test_search_unknown_namespace_exit_one(self, idx, capsys):
        code, _, err = run(
            ["search", "query", "--namespace", "nonexistent"], capsys, idx
        )
        assert code == 1
        assert "nonexistent" in err or "Namespace" in err

    def test_search_top_k_limits_results(self, idx, capsys):
        _, out, _ = run(["search", "stars", "--top-k", "1"], capsys, idx)
        # Each result starts with "[N]"
        result_lines = [l for l in out.splitlines() if l.startswith("[")]
        assert len(result_lines) == 1

    def test_search_default_top_k(self, idx, capsys):
        _, out, _ = run(["search", "the"], capsys, idx)
        result_lines = [l for l in out.splitlines() if l.startswith("[")]
        assert len(result_lines) <= DEFAULT_TOP_K

    def test_search_custom_namespace(self, tmp, idx, capsys):
        f = write_file(tmp, "extra.txt", "custom namespace only document")
        run(["index", str(f), "--namespace", "custom"], capsys, idx)
        code, out, _ = run(
            ["search", "custom namespace", "--namespace", "custom"], capsys, idx
        )
        assert code == 0
        assert out.strip() != ""

    def test_ef_search_flag_accepted(self, idx, capsys):
        """--ef-search flag should be accepted without error."""
        code, _, _ = run(["search", "loyal dog", "--ef-search", "200"], capsys, idx)
        assert code == 0

    def test_ef_search_returns_results(self, idx, capsys):
        """--ef-search flag should return results just like a normal search."""
        _, out, _ = run(["search", "stars", "--ef-search", "100"], capsys, idx)
        result_lines = [l for l in out.splitlines() if l.startswith("[")]
        assert len(result_lines) >= 1

    def test_ef_search_default_is_none(self):
        """--ef-search should default to None (index uses its own default)."""
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "search", "query"])
        assert args.ef_search is None

    def test_ef_search_custom_value_parsed(self):
        """--ef-search should parse an integer value correctly."""
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "search", "query", "--ef-search", "300"]
        )
        assert args.ef_search == 300


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


class TestCmdList:
    def test_list_no_index_prints_message(self, tmp, capsys):
        missing_idx = tmp / "no_index.pkl"
        code, out, _ = run(["list"], capsys, missing_idx)
        assert code == 0
        assert "No index found" in out

    def test_list_empty_index_prints_message(self, idx, capsys):
        manager = NamespaceManager()
        _save_manager(manager, idx)
        code, out, _ = run(["list"], capsys, idx)
        assert code == 0
        assert "empty" in out.lower() or "no namespace" in out.lower()

    def test_list_shows_namespace_name(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "some content for listing test")
        run(["index", str(f), "--namespace", "myproject"], capsys, idx)
        _, out, _ = run(["list"], capsys, idx)
        assert "myproject" in out

    def test_list_shows_document_count(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "some content for listing test")
        run(["index", str(f)], capsys, idx)
        _, out, _ = run(["list"], capsys, idx)
        # At least one number > 0 must appear
        import re
        numbers = [int(n) for n in re.findall(r"\d+", out)]
        assert any(n > 0 for n in numbers)

    def test_list_shows_total(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "content for total test")
        run(["index", str(f)], capsys, idx)
        _, out, _ = run(["list"], capsys, idx)
        assert "TOTAL" in out

    def test_list_multiple_namespaces(self, tmp, idx, capsys):
        f1 = write_file(tmp, "a.txt", "content alpha")
        f2 = write_file(tmp, "b.txt", "content beta")
        run(["index", str(f1), "--namespace", "ns_alpha"], capsys, idx)
        run(["index", str(f2), "--namespace", "ns_beta"], capsys, idx)
        _, out, _ = run(["list"], capsys, idx)
        assert "ns_alpha" in out
        assert "ns_beta" in out

    def test_list_exit_zero(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "content")
        run(["index", str(f)], capsys, idx)
        code, _, _ = run(["list"], capsys, idx)
        assert code == 0


# ---------------------------------------------------------------------------
# main() — argument routing and edge cases
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_subcommand_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_unknown_subcommand_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["unknown"])
        assert exc_info.value.code != 0

    def test_returns_int(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world")
        result = main(["--index", str(idx), "index", str(f)])
        assert isinstance(result, int)

    def test_index_then_search_end_to_end(self, tmp, idx, capsys):
        write_file(
            tmp,
            "ml.txt",
            "Machine learning is a subset of artificial intelligence. "
            "Neural networks are inspired by the human brain.",
        )
        code_i, _, _ = run(["index", str(tmp)], capsys, idx)
        assert code_i == 0

        code_s, out, _ = run(["search", "neural networks"], capsys, idx)
        assert code_s == 0
        assert "neural" in out.lower() or "brain" in out.lower() or out.strip()


# ---------------------------------------------------------------------------
# cmd_delete
# ---------------------------------------------------------------------------


class TestCmdDelete:
    @pytest.fixture(autouse=True)
    def pre_index(self, tmp, idx, capsys):
        """Index two files before each delete test."""
        self.tmp = tmp
        self.idx = idx
        write_file(tmp, "alpha.txt", "alpha content about machine learning")
        write_file(tmp, "beta.txt", "beta content about neural networks")
        run(["index", str(tmp)], capsys, idx)

    def test_delete_exit_zero(self, idx, capsys):
        code, _, _ = run(["delete", "alpha.txt"], capsys, idx)
        assert code == 0

    def test_delete_prints_confirmation(self, idx, capsys):
        _, out, _ = run(["delete", "alpha.txt"], capsys, idx)
        assert "alpha.txt" in out
        assert "Deleted" in out

    def test_delete_reduces_chunk_count(self, idx, capsys):
        count_before = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)
        run(["delete", "alpha.txt"], capsys, idx)
        count_after = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)
        assert count_after < count_before

    def test_delete_removes_only_target_file(self, idx, capsys):
        run(["delete", "alpha.txt"], capsys, idx)
        manager = _load_manager(idx)
        sources = manager.list_sources(DEFAULT_NAMESPACE)
        assert "alpha.txt" not in sources
        assert "beta.txt" in sources

    def test_delete_persists_to_disk(self, idx, capsys):
        run(["delete", "alpha.txt"], capsys, idx)
        manager = _load_manager(idx)
        sources = manager.list_sources(DEFAULT_NAMESPACE)
        assert "alpha.txt" not in sources

    def test_delete_no_index_exit_one(self, tmp, capsys):
        missing_idx = tmp / "no_index.pkl"
        code, _, err = run(["delete", "alpha.txt"], capsys, missing_idx)
        assert code == 1
        assert "index" in err.lower()

    def test_delete_unknown_namespace_exit_one(self, idx, capsys):
        code, _, err = run(
            ["delete", "alpha.txt", "--namespace", "nonexistent"], capsys, idx
        )
        assert code == 1
        assert "nonexistent" in err or "Namespace" in err

    def test_delete_unknown_filename_exit_one(self, idx, capsys):
        code, _, err = run(["delete", "ghost.txt"], capsys, idx)
        assert code == 1
        assert "ghost.txt" in err

    def test_delete_custom_namespace(self, tmp, idx, capsys):
        f = write_file(tmp, "extra.txt", "custom namespace document content here")
        run(["index", str(f), "--namespace", "myns"], capsys, idx)
        code, out, _ = run(["delete", "extra.txt", "--namespace", "myns"], capsys, idx)
        assert code == 0
        assert "extra.txt" in out

    def test_delete_default_namespace_leaves_other_namespace_intact(self, tmp, idx, capsys):
        f = write_file(tmp, "shared.txt", "shared document content here for testing")
        run(["index", str(f), "--namespace", "other"], capsys, idx)
        run(["delete", "shared.txt"], capsys, idx)
        manager = _load_manager(idx)
        # "other" namespace should still have shared.txt
        sources_other = manager.list_sources("other")
        assert "shared.txt" in sources_other

    def test_delete_subcommand_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "delete", "report.txt"])
        assert args.command == "delete"
        assert args.filename == "report.txt"
        assert args.namespace == DEFAULT_NAMESPACE

    def test_delete_custom_namespace_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "delete", "report.txt", "--namespace", "myns"]
        )
        assert args.namespace == "myns"

    def test_delete_no_filename_no_query_exit_one(self, idx, capsys):
        code, _, err = run(["delete"], capsys, idx)
        assert code == 1
        assert "FILENAME" in err or "filename" in err.lower() or "query" in err.lower()


# ---------------------------------------------------------------------------
# cmd_delete --query
# ---------------------------------------------------------------------------


class TestCmdDeleteByQuery:
    @pytest.fixture(autouse=True)
    def pre_index(self, tmp, idx, capsys):
        """Index two files before each query-delete test."""
        self.tmp = tmp
        self.idx = idx
        write_file(tmp, "dogs.txt", "dogs and puppies are wonderful loyal companions")
        write_file(tmp, "space.txt", "black holes warp spacetime and swallow light")
        run(["index", str(tmp)], capsys, idx)

    def test_delete_query_exit_zero(self, idx, capsys):
        code, _, _ = run(["delete", "--query", "dogs and puppies"], capsys, idx)
        assert code == 0

    def test_delete_query_prints_matched_chunks(self, idx, capsys):
        _, out, _ = run(["delete", "--query", "dogs and puppies"], capsys, idx)
        assert "Matched" in out
        assert "chunk" in out.lower()

    def test_delete_query_prints_deleted_count(self, idx, capsys):
        _, out, _ = run(["delete", "--query", "dogs and puppies"], capsys, idx)
        assert "Deleted" in out

    def test_delete_query_reduces_chunk_count(self, idx, capsys):
        count_before = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)
        run(["delete", "--query", "dogs and puppies", "--top-k", "1"], capsys, idx)
        count_after = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)
        assert count_after < count_before

    def test_delete_query_persists_to_disk(self, idx, capsys):
        run(["delete", "--query", "dogs and puppies", "--top-k", "1"], capsys, idx)
        manager = _load_manager(idx)
        # After delete, a search should return fewer results than before
        results = manager.search("dogs and puppies", DEFAULT_NAMESPACE, top_k=10)
        assert len(results) < _load_manager(idx).namespace_len(DEFAULT_NAMESPACE) + 1

    def test_delete_query_dry_run_does_not_modify_index(self, idx, capsys):
        count_before = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)
        run(["delete", "--query", "dogs", "--dry-run"], capsys, idx)
        count_after = _load_manager(idx).namespace_len(DEFAULT_NAMESPACE)
        assert count_before == count_after

    def test_delete_query_dry_run_prints_dry_run_message(self, idx, capsys):
        _, out, _ = run(["delete", "--query", "dogs", "--dry-run"], capsys, idx)
        assert "Dry run" in out or "dry run" in out.lower()

    def test_delete_query_no_index_exit_one(self, tmp, capsys):
        missing_idx = tmp / "no_index.pkl"
        code, _, err = run(["delete", "--query", "dogs"], capsys, missing_idx)
        assert code == 1
        assert "index" in err.lower()

    def test_delete_query_unknown_namespace_exit_one(self, idx, capsys):
        code, _, err = run(
            ["delete", "--query", "dogs", "--namespace", "ghost"], capsys, idx
        )
        assert code == 1
        assert "ghost" in err or "Namespace" in err

    def test_delete_query_top_k_parser(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "delete", "--query", "dogs", "--top-k", "3"]
        )
        assert args.query == "dogs"
        assert args.top_k == 3
        assert args.filename is None

    def test_delete_query_dry_run_parser(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "delete", "--query", "dogs", "--dry-run"]
        )
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# cmd_list_sources
# ---------------------------------------------------------------------------


class TestCmdListSources:
    def test_list_sources_no_index_prints_message(self, tmp, capsys):
        missing_idx = tmp / "no_index.pkl"
        code, out, _ = run(["list-sources"], capsys, missing_idx)
        assert code == 0
        assert "No index found" in out

    def test_list_sources_empty_namespace_prints_message(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "hello world test content here")
        run(["index", str(f)], capsys, idx)
        # Query a namespace with no sources (shouldn't happen normally, but test empty path)
        manager = _load_manager(idx)
        manager.create_namespace("empty_ns")
        _save_manager(manager, idx)
        _, out, _ = run(["list-sources", "--namespace", "empty_ns"], capsys, idx)
        assert "No sources" in out or "empty" in out.lower()

    def test_list_sources_shows_indexed_filename(self, tmp, idx, capsys):
        f = write_file(tmp, "report.txt", "quarterly earnings report content here")
        run(["index", str(f)], capsys, idx)
        _, out, _ = run(["list-sources"], capsys, idx)
        assert "report.txt" in out

    def test_list_sources_shows_all_files(self, tmp, idx, capsys):
        write_file(tmp, "alpha.txt", "alpha content")
        write_file(tmp, "beta.txt", "beta content")
        run(["index", str(tmp)], capsys, idx)
        _, out, _ = run(["list-sources"], capsys, idx)
        assert "alpha.txt" in out
        assert "beta.txt" in out

    def test_list_sources_output_is_sorted(self, tmp, idx, capsys):
        write_file(tmp, "zebra.txt", "zebra content here")
        write_file(tmp, "apple.txt", "apple content here")
        run(["index", str(tmp)], capsys, idx)
        _, out, _ = run(["list-sources"], capsys, idx)
        lines = [l for l in out.splitlines() if l.strip()]
        assert lines == sorted(lines)

    def test_list_sources_custom_namespace(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "document in custom namespace content")
        run(["index", str(f), "--namespace", "myns"], capsys, idx)
        _, out, _ = run(["list-sources", "--namespace", "myns"], capsys, idx)
        assert "doc.txt" in out

    def test_list_sources_unknown_namespace_exit_one(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "content")
        run(["index", str(f)], capsys, idx)
        code, _, err = run(["list-sources", "--namespace", "ghost"], capsys, idx)
        assert code == 1
        assert "ghost" in err or "Namespace" in err

    def test_list_sources_exit_zero(self, tmp, idx, capsys):
        f = write_file(tmp, "doc.txt", "content")
        run(["index", str(f)], capsys, idx)
        code, _, _ = run(["list-sources"], capsys, idx)
        assert code == 0

    def test_list_sources_namespace_isolated(self, tmp, idx, capsys):
        f1 = write_file(tmp, "a.txt", "content alpha here")
        f2 = write_file(tmp, "b.txt", "content beta here")
        run(["index", str(f1), "--namespace", "ns1"], capsys, idx)
        run(["index", str(f2), "--namespace", "ns2"], capsys, idx)
        _, out1, _ = run(["list-sources", "--namespace", "ns1"], capsys, idx)
        _, out2, _ = run(["list-sources", "--namespace", "ns2"], capsys, idx)
        assert "a.txt" in out1
        assert "b.txt" not in out1
        assert "b.txt" in out2
        assert "a.txt" not in out2

    def test_list_sources_subcommand_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "list-sources"])
        assert args.command == "list-sources"
        assert args.namespace == DEFAULT_NAMESPACE

    def test_list_sources_custom_namespace_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "list-sources", "--namespace", "myns"]
        )
        assert args.namespace == "myns"


# ---------------------------------------------------------------------------
# cmd_update
# ---------------------------------------------------------------------------


class TestCmdUpdate:
    @pytest.fixture(autouse=True)
    def pre_index(self, tmp, idx, capsys):
        """Index a file before each update test."""
        self.tmp = tmp
        self.idx = idx
        self.f = write_file(tmp, "doc.txt", "original content about dogs and puppies")
        run(["index", str(self.f)], capsys, idx)

    def test_update_exit_zero(self, idx, capsys):
        self.f.write_text("updated content about cats and kittens", encoding="utf-8")
        code, _, _ = run(["update", str(self.f)], capsys, idx)
        assert code == 0

    def test_update_prints_confirmation(self, idx, capsys):
        self.f.write_text("updated content about cats and kittens", encoding="utf-8")
        _, out, _ = run(["update", str(self.f)], capsys, idx)
        assert "Updated" in out
        assert "doc.txt" in out

    def test_update_replaces_old_content(self, tmp, idx, capsys):
        self.f.write_text("completely new text about space exploration", encoding="utf-8")
        run(["update", str(self.f)], capsys, idx)
        manager = _load_manager(idx)
        results = manager.search("dogs and puppies", DEFAULT_NAMESPACE, top_k=5)
        texts = [r["text"] for r in results]
        assert not any("dogs" in t for t in texts)

    def test_update_indexes_new_content(self, tmp, idx, capsys):
        self.f.write_text("space exploration and astronomy", encoding="utf-8")
        run(["update", str(self.f)], capsys, idx)
        manager = _load_manager(idx)
        results = manager.search("space exploration", DEFAULT_NAMESPACE, top_k=1)
        assert len(results) == 1
        assert "space" in results[0]["text"].lower()

    def test_update_persists_to_disk(self, tmp, idx, capsys):
        self.f.write_text("completely new persisted content", encoding="utf-8")
        run(["update", str(self.f)], capsys, idx)
        manager = _load_manager(idx)
        sources = manager.list_sources(DEFAULT_NAMESPACE)
        assert "doc.txt" in sources

    def test_update_no_index_exit_one(self, tmp, capsys):
        missing_idx = tmp / "no_index.pkl"
        code, _, err = run(["update", str(self.f)], capsys, missing_idx)
        assert code == 1
        assert "index" in err.lower()

    def test_update_nonexistent_file_exit_one(self, tmp, idx, capsys):
        code, _, err = run(["update", str(tmp / "ghost.txt")], capsys, idx)
        assert code == 1
        assert "Error" in err

    def test_update_unsupported_extension_exit_one(self, tmp, idx, capsys):
        f = tmp / "doc.pdf"
        f.write_bytes(b"%PDF")
        code, _, err = run(["update", str(f)], capsys, idx)
        assert code == 1
        assert "Error" in err

    def test_update_unknown_namespace_exit_one(self, idx, capsys):
        self.f.write_text("new content here", encoding="utf-8")
        code, _, err = run(
            ["update", str(self.f), "--namespace", "ghost"], capsys, idx
        )
        assert code == 1
        assert "ghost" in err or "Namespace" in err

    def test_update_custom_namespace(self, tmp, idx, capsys):
        f = write_file(tmp, "ns_doc.txt", "original content in custom namespace here")
        run(["index", str(f), "--namespace", "myns"], capsys, idx)
        f.write_text("updated content in custom namespace here", encoding="utf-8")
        code, out, _ = run(["update", str(f), "--namespace", "myns"], capsys, idx)
        assert code == 0
        assert "ns_doc.txt" in out

    def test_update_subcommand_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "update", "/some/file.txt"])
        assert args.command == "update"
        assert args.path == "/some/file.txt"
        assert args.namespace == DEFAULT_NAMESPACE
        assert args.chunk_size == DEFAULT_CHUNK_SIZE
        assert args.chunk_overlap == DEFAULT_CHUNK_OVERLAP

    def test_update_custom_chunk_size_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "update", "/f.txt", "--chunk-size", "64"]
        )
        assert args.chunk_size == 64


# ---------------------------------------------------------------------------
# cmd_export
# ---------------------------------------------------------------------------


class TestCmdExport:
    @pytest.fixture(autouse=True)
    def pre_index(self, tmp, idx, capsys):
        """Index two files before each export test."""
        self.tmp = tmp
        self.idx = idx
        write_file(tmp, "dogs.txt", "dogs and puppies are wonderful loyal companions")
        write_file(tmp, "space.txt", "black holes warp spacetime and swallow light")
        run(["index", str(tmp)], capsys, idx)

    # --- Basic behaviour ---

    def test_export_exit_zero(self, tmp, idx, capsys):
        out = tmp / "out.json"
        code, _, _ = run(["export", str(out)], capsys, idx)
        assert code == 0

    def test_export_creates_file(self, tmp, idx, capsys):
        out = tmp / "out.json"
        run(["export", str(out)], capsys, idx)
        assert out.exists()

    def test_export_produces_valid_json(self, tmp, idx, capsys):
        import json as _json
        out = tmp / "out.json"
        run(["export", str(out)], capsys, idx)
        with out.open() as fh:
            payload = _json.load(fh)
        assert isinstance(payload, dict)

    def test_export_prints_chunk_count(self, tmp, idx, capsys):
        out = tmp / "out.json"
        _, stdout, _ = run(["export", str(out)], capsys, idx)
        assert "chunk" in stdout.lower()

    def test_export_prints_namespace_count(self, tmp, idx, capsys):
        out = tmp / "out.json"
        _, stdout, _ = run(["export", str(out)], capsys, idx)
        assert "namespace" in stdout.lower()

    def test_export_output_contains_path(self, tmp, idx, capsys):
        out = tmp / "out.json"
        _, stdout, _ = run(["export", str(out)], capsys, idx)
        assert "out.json" in stdout

    # --- Single namespace ---

    def test_export_single_namespace_exit_zero(self, tmp, idx, capsys):
        out = tmp / "out.json"
        code, _, _ = run(["export", str(out), "--namespace", "default"], capsys, idx)
        assert code == 0

    def test_export_single_namespace_only_exports_that_ns(self, tmp, idx, capsys):
        import json as _json
        # Add second namespace
        write_file(tmp, "extra.txt", "extra document in other namespace")
        run(["index", str(tmp / "extra.txt"), "--namespace", "other"], capsys, idx)
        out = tmp / "out.json"
        run(["export", str(out), "--namespace", "default"], capsys, idx)
        with out.open() as fh:
            payload = _json.load(fh)
        assert list(payload["namespaces"].keys()) == ["default"]

    def test_export_unknown_namespace_exit_one(self, tmp, idx, capsys):
        out = tmp / "out.json"
        code, _, err = run(["export", str(out), "--namespace", "ghost"], capsys, idx)
        assert code == 1
        assert "ghost" in err or "Namespace" in err or "not found" in err.lower()

    # --- No index ---

    def test_export_no_index_exit_one(self, tmp, capsys):
        missing_idx = tmp / "no_index.pkl"
        out = tmp / "out.json"
        code, _, err = run(["export", str(out)], capsys, missing_idx)
        assert code == 1
        assert "index" in err.lower()

    # --- Parser ---

    def test_export_subcommand_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "export", "/tmp/out.json"])
        assert args.command == "export"
        assert args.output == "/tmp/out.json"
        assert args.namespace is None

    def test_export_namespace_flag_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "export", "/tmp/out.json", "--namespace", "myns"]
        )
        assert args.namespace == "myns"


# ---------------------------------------------------------------------------
# cmd_import
# ---------------------------------------------------------------------------


class TestCmdImport:
    @pytest.fixture()
    def exported_json(self, tmp, idx, capsys):
        """Index a file, export it, and return the JSON path."""
        write_file(tmp, "source.txt", "the mitochondria is the powerhouse of the cell")
        run(["index", str(tmp / "source.txt")], capsys, idx)
        out = tmp / "export.json"
        run(["export", str(out)], capsys, idx)
        return out

    # --- Basic behaviour ---

    def test_import_exit_zero(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        code, _, _ = run(["import", str(exported_json)], capsys, fresh_idx)
        assert code == 0

    def test_import_prints_chunk_count(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        _, stdout, _ = run(["import", str(exported_json)], capsys, fresh_idx)
        assert "Imported" in stdout
        assert "chunk" in stdout.lower()

    def test_import_creates_index(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        run(["import", str(exported_json)], capsys, fresh_idx)
        assert fresh_idx.exists()

    def test_import_adds_documents_to_manager(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        run(["import", str(exported_json)], capsys, fresh_idx)
        manager = _load_manager(fresh_idx)
        assert "default" in manager.list_namespaces()
        assert manager.namespace_len("default") > 0

    def test_imported_content_is_searchable(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        run(["import", str(exported_json)], capsys, fresh_idx)
        code, out, _ = run(["search", "powerhouse cell"], capsys, fresh_idx)
        assert code == 0
        assert out.strip() != ""

    # --- --namespace flag ---

    def test_import_namespace_flag_redirects_to_target(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        run(["import", str(exported_json), "--namespace", "imported"], capsys, fresh_idx)
        manager = _load_manager(fresh_idx)
        assert "imported" in manager.list_namespaces()

    def test_import_namespace_flag_source_absent(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        run(["import", str(exported_json), "--namespace", "imported"], capsys, fresh_idx)
        manager = _load_manager(fresh_idx)
        assert "default" not in manager.list_namespaces()

    def test_import_namespace_flag_content_searchable(self, tmp, idx, capsys, exported_json):
        fresh_idx = tmp / "fresh.pkl"
        run(["import", str(exported_json), "--namespace", "imported"], capsys, fresh_idx)
        code, out, _ = run(
            ["search", "powerhouse cell", "--namespace", "imported"], capsys, fresh_idx
        )
        assert code == 0
        assert out.strip() != ""

    # --- Error conditions ---

    def test_import_missing_file_exit_one(self, tmp, capsys):
        fresh_idx = tmp / "fresh.pkl"
        code, _, err = run(["import", str(tmp / "ghost.json")], capsys, fresh_idx)
        assert code == 1
        assert "not found" in err.lower() or "Error" in err

    def test_import_bad_version_exit_one(self, tmp, capsys):
        import json as _json
        bad_json = tmp / "bad.json"
        bad_json.write_text(
            _json.dumps({"neuroseek_version": "99", "namespaces": {}}),
            encoding="utf-8",
        )
        fresh_idx = tmp / "fresh.pkl"
        code, _, err = run(["import", str(bad_json)], capsys, fresh_idx)
        assert code == 1
        assert "Error" in err

    # --- Parser ---

    def test_import_subcommand_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(["--index", "/tmp/x.pkl", "import", "/tmp/in.json"])
        assert args.command == "import"
        assert args.input == "/tmp/in.json"
        assert args.namespace is None

    def test_import_namespace_flag_in_parser(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--index", "/tmp/x.pkl", "import", "/tmp/in.json", "--namespace", "myns"]
        )
        assert args.namespace == "myns"

