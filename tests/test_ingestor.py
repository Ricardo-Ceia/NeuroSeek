"""
Tests for neuroseek/ingestor.py — ingest_file() and ingest_directory()

Run with:
    PYTHONPATH=/home/ricardo/NeuroSeek ~/.local/bin/pytest tests/test_ingestor.py -q
"""

import os
import warnings
from pathlib import Path

import pytest

from neuroseek.ingestion.ingestor import (
    SUPPORTED_EXTENSIONS,
    ingest_directory,
    ingest_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp(tmp_path):
    """Alias for pytest's tmp_path for brevity."""
    return tmp_path


def write(directory: Path, name: str, content: str) -> Path:
    """Helper: write a file and return its Path."""
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS constant
# ---------------------------------------------------------------------------


class TestSupportedExtensions:
    def test_is_frozenset(self):
        assert isinstance(SUPPORTED_EXTENSIONS, frozenset)

    def test_contains_expected_types(self):
        assert {"txt", "md", "py", "json", "csv"} <= SUPPORTED_EXTENSIONS

    def test_values_are_lowercase_no_dot(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower()
            assert not ext.startswith(".")


# ---------------------------------------------------------------------------
# ingest_file — happy path
# ---------------------------------------------------------------------------


class TestIngestFileHappyPath:
    def test_returns_tuple(self, tmp):
        f = write(tmp, "doc.txt", "hello world")
        result = ingest_file(f)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_text_content_correct(self, tmp):
        f = write(tmp, "doc.txt", "hello world")
        text, _ = ingest_file(f)
        assert text == "hello world"

    def test_txt_extension(self, tmp):
        f = write(tmp, "file.txt", "text content")
        _, meta = ingest_file(f)
        assert meta["filetype"] == "txt"

    def test_md_extension(self, tmp):
        f = write(tmp, "readme.md", "# Heading")
        text, meta = ingest_file(f)
        assert text == "# Heading"
        assert meta["filetype"] == "md"

    def test_py_extension(self, tmp):
        f = write(tmp, "script.py", "def foo(): pass")
        text, meta = ingest_file(f)
        assert text == "def foo(): pass"
        assert meta["filetype"] == "py"

    def test_json_extension(self, tmp):
        f = write(tmp, "data.json", '{"key": "value"}')
        text, meta = ingest_file(f)
        assert text == '{"key": "value"}'
        assert meta["filetype"] == "json"

    def test_csv_extension(self, tmp):
        f = write(tmp, "data.csv", "a,b,c\n1,2,3")
        text, meta = ingest_file(f)
        assert text == "a,b,c\n1,2,3"
        assert meta["filetype"] == "csv"

    def test_metadata_has_path_key(self, tmp):
        f = write(tmp, "doc.txt", "x")
        _, meta = ingest_file(f)
        assert "path" in meta

    def test_metadata_path_is_absolute(self, tmp):
        f = write(tmp, "doc.txt", "x")
        _, meta = ingest_file(f)
        assert os.path.isabs(meta["path"])

    def test_metadata_filename_correct(self, tmp):
        f = write(tmp, "my_file.txt", "x")
        _, meta = ingest_file(f)
        assert meta["filename"] == "my_file.txt"

    def test_metadata_filetype_correct(self, tmp):
        f = write(tmp, "notes.md", "x")
        _, meta = ingest_file(f)
        assert meta["filetype"] == "md"

    def test_metadata_path_resolves_to_real_path(self, tmp):
        f = write(tmp, "doc.txt", "content")
        _, meta = ingest_file(f)
        assert meta["path"] == str(f.resolve())

    def test_accepts_str_path(self, tmp):
        f = write(tmp, "doc.txt", "hello")
        text, meta = ingest_file(str(f))
        assert text == "hello"
        assert meta["filename"] == "doc.txt"

    def test_accepts_path_object(self, tmp):
        f = write(tmp, "doc.txt", "hello")
        text, meta = ingest_file(f)
        assert text == "hello"

    def test_empty_file_returns_empty_string(self, tmp):
        f = write(tmp, "empty.txt", "")
        text, meta = ingest_file(f)
        assert text == ""

    def test_multiline_content_preserved(self, tmp):
        content = "line one\nline two\nline three"
        f = write(tmp, "multi.txt", content)
        text, _ = ingest_file(f)
        assert text == content

    def test_unicode_content_preserved(self, tmp):
        content = "café résumé naïve"
        f = write(tmp, "unicode.txt", content)
        text, _ = ingest_file(f)
        assert text == content

    def test_extension_case_insensitive(self, tmp):
        # Create a file with uppercase extension manually
        f = tmp / "DOC.TXT"
        f.write_text("hello", encoding="utf-8")
        text, meta = ingest_file(f)
        assert meta["filetype"] == "txt"

    def test_metadata_keys_are_exactly_three(self, tmp):
        f = write(tmp, "doc.txt", "x")
        _, meta = ingest_file(f)
        assert set(meta.keys()) == {"path", "filename", "filetype"}


# ---------------------------------------------------------------------------
# ingest_file — error cases
# ---------------------------------------------------------------------------


class TestIngestFileErrors:
    def test_nonexistent_file_raises_file_not_found(self, tmp):
        with pytest.raises(FileNotFoundError):
            ingest_file(tmp / "ghost.txt")

    def test_directory_path_raises_is_a_directory_error(self, tmp):
        with pytest.raises(IsADirectoryError):
            ingest_file(tmp)

    def test_unsupported_extension_raises_value_error(self, tmp):
        f = tmp / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ingest_file(f)

    def test_docx_extension_raises_value_error(self, tmp):
        f = tmp / "doc.docx"
        f.write_bytes(b"PK\x03\x04")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ingest_file(f)

    def test_no_extension_raises_value_error(self, tmp):
        f = tmp / "README"
        f.write_text("no extension", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ingest_file(f)

    def test_error_message_lists_supported_extensions(self, tmp):
        f = tmp / "archive.zip"
        f.write_bytes(b"PK")
        with pytest.raises(ValueError, match="Supported"):
            ingest_file(f)

    def test_binary_file_utf8_decode_error(self, tmp):
        f = tmp / "binary.txt"
        f.write_bytes(b"\xff\xfe\x00\x01")
        with pytest.raises(UnicodeDecodeError):
            ingest_file(f)


# ---------------------------------------------------------------------------
# ingest_directory — happy path
# ---------------------------------------------------------------------------


class TestIngestDirectoryHappyPath:
    def test_returns_list(self, tmp):
        write(tmp, "a.txt", "hello")
        result = ingest_directory(tmp)
        assert isinstance(result, list)

    def test_empty_directory_returns_empty_list(self, tmp):
        result = ingest_directory(tmp)
        assert result == []

    def test_single_file_returns_one_entry(self, tmp):
        write(tmp, "a.txt", "hello")
        result = ingest_directory(tmp)
        assert len(result) == 1

    def test_each_entry_is_tuple_of_two(self, tmp):
        write(tmp, "a.txt", "hello")
        result = ingest_directory(tmp)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_text_content_correct(self, tmp):
        write(tmp, "a.txt", "hello world")
        text, _ = ingest_directory(tmp)[0]
        assert text == "hello world"

    def test_metadata_correct(self, tmp):
        f = write(tmp, "a.txt", "hello")
        _, meta = ingest_directory(tmp)[0]
        assert meta["filename"] == "a.txt"
        assert meta["filetype"] == "txt"
        assert os.path.isabs(meta["path"])

    def test_all_supported_extensions_ingested(self, tmp):
        for ext in SUPPORTED_EXTENSIONS:
            write(tmp, f"file.{ext}", f"content for {ext}")
        result = ingest_directory(tmp)
        found_exts = {meta["filetype"] for _, meta in result}
        assert found_exts == SUPPORTED_EXTENSIONS

    def test_unsupported_extensions_skipped(self, tmp):
        write(tmp, "doc.txt", "supported")
        (tmp / "image.png").write_bytes(b"\x89PNG")
        (tmp / "archive.zip").write_bytes(b"PK")
        result = ingest_directory(tmp)
        assert len(result) == 1
        assert result[0][1]["filetype"] == "txt"

    def test_recursive_subdirectory_ingestion(self, tmp):
        sub = tmp / "subdir"
        sub.mkdir()
        write(tmp, "root.txt", "root")
        write(sub, "child.txt", "child")
        result = ingest_directory(tmp)
        filenames = {meta["filename"] for _, meta in result}
        assert filenames == {"root.txt", "child.txt"}

    def test_deeply_nested_files_ingested(self, tmp):
        deep = tmp / "a" / "b" / "c"
        deep.mkdir(parents=True)
        write(deep, "deep.md", "deep content")
        result = ingest_directory(tmp)
        assert len(result) == 1
        assert result[0][1]["filename"] == "deep.md"

    def test_results_sorted_by_path(self, tmp):
        write(tmp, "z.txt", "z")
        write(tmp, "a.txt", "a")
        write(tmp, "m.txt", "m")
        result = ingest_directory(tmp)
        paths = [meta["path"] for _, meta in result]
        assert paths == sorted(paths)

    def test_accepts_str_path(self, tmp):
        write(tmp, "a.txt", "hello")
        result = ingest_directory(str(tmp))
        assert len(result) == 1

    def test_accepts_path_object(self, tmp):
        write(tmp, "a.txt", "hello")
        result = ingest_directory(tmp)
        assert len(result) == 1

    def test_multiple_files_all_returned(self, tmp):
        write(tmp, "a.txt", "a")
        write(tmp, "b.md", "b")
        write(tmp, "c.py", "c")
        result = ingest_directory(tmp)
        assert len(result) == 3

    def test_extensions_filter_restricts_types(self, tmp):
        write(tmp, "a.txt", "txt")
        write(tmp, "b.md", "md")
        write(tmp, "c.py", "py")
        result = ingest_directory(tmp, extensions=frozenset({"txt"}))
        assert len(result) == 1
        assert result[0][1]["filetype"] == "txt"

    def test_extensions_filter_multiple_types(self, tmp):
        write(tmp, "a.txt", "txt")
        write(tmp, "b.md", "md")
        write(tmp, "c.py", "py")
        result = ingest_directory(tmp, extensions=frozenset({"txt", "md"}))
        assert len(result) == 2
        exts = {meta["filetype"] for _, meta in result}
        assert exts == {"txt", "md"}

    def test_empty_extensions_filter_returns_nothing(self, tmp):
        write(tmp, "a.txt", "txt")
        result = ingest_directory(tmp, extensions=frozenset())
        assert result == []


# ---------------------------------------------------------------------------
# ingest_directory — UTF-8 decode warning
# ---------------------------------------------------------------------------


class TestIngestDirectoryUnicodeWarning:
    def test_undecodable_file_skipped_with_warning(self, tmp):
        write(tmp, "good.txt", "hello")
        bad = tmp / "bad.txt"
        bad.write_bytes(b"\xff\xfe\x00\x01")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ingest_directory(tmp)
        assert len(result) == 1
        assert result[0][1]["filename"] == "good.txt"
        assert any("UTF-8" in str(w.message) or "UTF" in str(w.message) for w in caught)

    def test_warning_is_user_warning(self, tmp):
        bad = tmp / "bad.txt"
        bad.write_bytes(b"\xff\xfe")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ingest_directory(tmp)
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_other_files_still_returned_after_bad_file(self, tmp):
        bad = tmp / "aaa_bad.txt"
        bad.write_bytes(b"\xff\xfe")
        write(tmp, "zzz_good.txt", "good content")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = ingest_directory(tmp)
        assert len(result) == 1
        assert result[0][1]["filename"] == "zzz_good.txt"


# ---------------------------------------------------------------------------
# ingest_directory — error cases
# ---------------------------------------------------------------------------


class TestIngestDirectoryErrors:
    def test_nonexistent_path_raises_file_not_found(self, tmp):
        with pytest.raises(FileNotFoundError):
            ingest_directory(tmp / "ghost_dir")

    def test_file_path_raises_not_a_directory_error(self, tmp):
        f = write(tmp, "file.txt", "x")
        with pytest.raises(NotADirectoryError):
            ingest_directory(f)

    def test_unsupported_extension_in_filter_raises_value_error(self, tmp):
        with pytest.raises(ValueError, match="Unsupported extension"):
            ingest_directory(tmp, extensions=frozenset({"pdf"}))

    def test_mixed_valid_invalid_extensions_raises_value_error(self, tmp):
        with pytest.raises(ValueError, match="Unsupported extension"):
            ingest_directory(tmp, extensions=frozenset({"txt", "pdf"}))


# ---------------------------------------------------------------------------
# Integration: ingest_file + ingest_directory consistency
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_single_file_same_result_both_ways(self, tmp):
        f = write(tmp, "doc.txt", "consistent content")
        single_text, single_meta = ingest_file(f)
        dir_results = ingest_directory(tmp)
        assert len(dir_results) == 1
        dir_text, dir_meta = dir_results[0]
        assert single_text == dir_text
        assert single_meta == dir_meta
