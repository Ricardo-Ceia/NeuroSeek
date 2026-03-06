"""
Tests for neuroseek/persistence/json_persistence.py.

Covers export_namespace_manager() and import_from_json() using real embeddings
(model is loaded once via the session-scoped `embedder` fixture in conftest.py
and reused through the module-scoped _patch_sentence_transformer fixture).

Run with:
    PYTHONPATH=/home/ricardo/NeuroSeek ~/.local/bin/pytest tests/test_json_persistence.py -q
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neuroseek.namespace_manager import NamespaceManager
from neuroseek.persistence.json_persistence import (
    _SCHEMA_VERSION,
    export_namespace_manager,
    import_from_json,
)


# ---------------------------------------------------------------------------
# Session-level model cache — reuse shared embedder so model loads once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _patch_sentence_transformer(embedder):
    cached_model = embedder._model

    class _CachedST:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(cached_model.__dict__)
            self.__class__ = cached_model.__class__

    with patch("neuroseek.embedder.SentenceTransformer", _CachedST):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager() -> NamespaceManager:
    return NamespaceManager()


# ---------------------------------------------------------------------------
# export_namespace_manager
# ---------------------------------------------------------------------------


class TestExportNamespaceManager:

    @pytest.fixture()
    def manager(self):
        m = _make_manager()
        m.add("The quick brown fox", namespace="default", metadata={"source": "test"})
        m.add("Semantic search is powerful", namespace="default")
        m.add("Another namespace doc", namespace="other")
        return m

    # --- File creation ---

    def test_creates_file(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        assert out.exists()

    def test_creates_file_str_path(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, str(out))
        assert out.exists()

    # --- JSON structure ---

    def test_valid_json(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        assert isinstance(payload, dict)

    def test_schema_version_present(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        assert payload["neuroseek_version"] == _SCHEMA_VERSION

    def test_namespaces_key_present(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        assert "namespaces" in payload

    def test_all_namespaces_exported(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        assert set(payload["namespaces"].keys()) == {"default", "other"}

    def test_document_count_default(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        assert len(payload["namespaces"]["default"]) == 2

    def test_document_count_other(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        assert len(payload["namespaces"]["other"]) == 1

    def test_each_doc_has_id_key(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        for doc in payload["namespaces"]["default"]:
            assert "id" in doc

    def test_each_doc_has_text_key(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        for doc in payload["namespaces"]["default"]:
            assert "text" in doc

    def test_each_doc_has_metadata_key(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        for doc in payload["namespaces"]["default"]:
            assert "metadata" in doc

    def test_text_content_preserved(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        texts = {d["text"] for d in payload["namespaces"]["default"]}
        assert "The quick brown fox" in texts
        assert "Semantic search is powerful" in texts

    def test_metadata_preserved(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        doc_map = {d["text"]: d for d in payload["namespaces"]["default"]}
        assert doc_map["The quick brown fox"]["metadata"]["source"] == "test"

    def test_docs_sorted_by_id(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out)
        with out.open() as fh:
            payload = json.load(fh)
        ids = [d["id"] for d in payload["namespaces"]["default"]]
        assert ids == sorted(ids)

    # --- Single-namespace export ---

    def test_single_namespace_export_only_contains_that_ns(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out, namespace="default")
        with out.open() as fh:
            payload = json.load(fh)
        assert list(payload["namespaces"].keys()) == ["default"]

    def test_single_namespace_excludes_other(self, manager, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(manager, out, namespace="default")
        with out.open() as fh:
            payload = json.load(fh)
        assert "other" not in payload["namespaces"]

    def test_nonexistent_namespace_raises_key_error(self, manager, tmp_path):
        out = tmp_path / "export.json"
        with pytest.raises(KeyError):
            export_namespace_manager(manager, out, namespace="does_not_exist")

    # --- Empty manager ---

    def test_empty_manager_produces_empty_namespaces(self, tmp_path):
        out = tmp_path / "export.json"
        export_namespace_manager(_make_manager(), out)
        with out.open() as fh:
            payload = json.load(fh)
        assert payload["namespaces"] == {}


# ---------------------------------------------------------------------------
# import_from_json
# ---------------------------------------------------------------------------


class TestImportFromJson:

    @pytest.fixture()
    def src_export(self, tmp_path):
        """Return (manager, export_path) — manager has two docs in 'default'."""
        m = _make_manager()
        m.add("The quick brown fox", namespace="default", metadata={"filename": "a.txt"})
        m.add("Semantic search is powerful", namespace="default")
        out = tmp_path / "export.json"
        export_namespace_manager(m, out)
        return m, out

    # --- Return type ---

    def test_returns_dict(self, src_export):
        _, out = src_export
        dst = _make_manager()
        result = import_from_json(out, dst)
        assert isinstance(result, dict)

    # --- Basic counts ---

    def test_imported_count_correct(self, src_export):
        _, out = src_export
        dst = _make_manager()
        counts = import_from_json(out, dst)
        assert counts.get("default") == 2

    def test_namespace_exists_after_import(self, src_export):
        _, out = src_export
        dst = _make_manager()
        import_from_json(out, dst)
        assert "default" in dst.list_namespaces()

    def test_document_count_matches_after_import(self, src_export):
        _, out = src_export
        dst = _make_manager()
        import_from_json(out, dst)
        assert dst.namespace_len("default") == 2

    # --- Multiple namespaces ---

    def test_multiple_namespaces_imported(self, tmp_path):
        src = _make_manager()
        src.add("text a", namespace="alpha")
        src.add("text b", namespace="beta")
        out = tmp_path / "export.json"
        export_namespace_manager(src, out)
        dst = _make_manager()
        counts = import_from_json(out, dst)
        assert "alpha" in counts
        assert "beta" in counts
        assert dst.namespace_len("alpha") == 1
        assert dst.namespace_len("beta") == 1

    # --- Semantic search after import ---

    def test_imported_text_is_searchable(self, tmp_path):
        src = _make_manager()
        src.add("The mitochondria is the powerhouse of the cell", namespace="bio")
        out = tmp_path / "export.json"
        export_namespace_manager(src, out)
        dst = _make_manager()
        import_from_json(out, dst)
        results = dst.search("powerhouse cell", namespace="bio", top_k=1)
        assert len(results) == 1
        assert "mitochondria" in results[0]["text"]

    # --- Metadata preservation ---

    def test_metadata_preserved_after_import(self, src_export):
        _, out = src_export
        dst = _make_manager()
        import_from_json(out, dst)
        results = dst.search("quick fox", namespace="default", top_k=1)
        assert results[0]["metadata"].get("filename") == "a.txt"

    # --- namespace_map ---

    def test_namespace_map_redirects_namespace(self, src_export):
        _, out = src_export
        dst = _make_manager()
        import_from_json(out, dst, namespace_map={"default": "imported"})
        assert "imported" in dst.list_namespaces()

    def test_namespace_map_source_name_absent(self, src_export):
        _, out = src_export
        dst = _make_manager()
        import_from_json(out, dst, namespace_map={"default": "imported"})
        assert "default" not in dst.list_namespaces()

    def test_namespace_map_count_correct(self, src_export):
        _, out = src_export
        dst = _make_manager()
        counts = import_from_json(out, dst, namespace_map={"default": "new"})
        assert counts.get("new") == 2

    def test_unmapped_namespaces_use_source_name(self, tmp_path):
        src = _make_manager()
        src.add("text", namespace="keep")
        out = tmp_path / "export.json"
        export_namespace_manager(src, out)
        dst = _make_manager()
        import_from_json(out, dst, namespace_map={"other": "renamed"})
        assert "keep" in dst.list_namespaces()

    # --- Error conditions ---

    def test_missing_file_raises_file_not_found(self, tmp_path):
        dst = _make_manager()
        with pytest.raises(FileNotFoundError):
            import_from_json(tmp_path / "nonexistent.json", dst)

    def test_wrong_version_raises_value_error(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text(
            json.dumps({"neuroseek_version": "99", "namespaces": {}}),
            encoding="utf-8",
        )
        dst = _make_manager()
        with pytest.raises(ValueError, match="neuroseek_version"):
            import_from_json(bad_json, dst)

    def test_missing_version_raises_value_error(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text(
            json.dumps({"namespaces": {}}),
            encoding="utf-8",
        )
        dst = _make_manager()
        with pytest.raises(ValueError):
            import_from_json(bad_json, dst)

    # --- Roundtrip ---

    def test_roundtrip_preserves_namespace_names(self, tmp_path):
        src = _make_manager()
        src.add("first", namespace="ns_a")
        src.add("second", namespace="ns_b")
        out = tmp_path / "export.json"
        export_namespace_manager(src, out)
        dst = _make_manager()
        import_from_json(out, dst)
        assert set(dst.list_namespaces()) == {"ns_a", "ns_b"}

    def test_roundtrip_total_document_count(self, tmp_path):
        src = _make_manager()
        for i in range(4):
            src.add(f"document number {i}", namespace="default")
        out = tmp_path / "export.json"
        export_namespace_manager(src, out)
        dst = _make_manager()
        import_from_json(out, dst)
        assert dst.namespace_len("default") == 4

    # --- Edge cases ---

    def test_empty_json_namespaces_returns_empty_dict(self, tmp_path):
        payload = {"neuroseek_version": _SCHEMA_VERSION, "namespaces": {}}
        out = tmp_path / "empty.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        dst = _make_manager()
        counts = import_from_json(out, dst)
        assert counts == {}

    def test_str_path_accepted(self, src_export):
        _, out = src_export
        dst = _make_manager()
        counts = import_from_json(str(out), dst)
        assert counts.get("default") == 2
