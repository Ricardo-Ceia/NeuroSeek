"""
Comprehensive tests for save_namespace_manager / load_namespace_manager.

No mocking — real embeddings, real file I/O, real round-trips.
The model is loaded once per session via the conftest fixture.
"""

import os
import tempfile
import unittest
import pytest

from neuroseek.embedder import Embedder
from neuroseek.namespace_manager import NamespaceManager
from neuroseek.persistence.namespace_manager_persistence import (
    load_namespace_manager,
    save_namespace_manager,
)


def _tmp_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".nsmgr")
    os.close(fd)
    return path


class TestSaveLoadRoundTrip(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        mgr = NamespaceManager._from_embedder(embedder)
        mgr.add("Paris is the capital of France", "cities")
        mgr.add("Berlin is the capital of Germany", "cities")
        mgr.add("dogs and puppies are great pets", "animals")
        mgr.add("quantum mechanics and thermodynamics", "science")
        self.mgr = mgr
        self.path = _tmp_path()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    # --- File creation ---

    def test_save_creates_file(self):
        os.remove(self.path)
        save_namespace_manager(self.mgr, self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_saved_file_is_non_empty(self):
        save_namespace_manager(self.mgr, self.path)
        self.assertGreater(os.path.getsize(self.path), 0)

    # --- Basic round-trip ---

    def test_load_returns_namespace_manager(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertIsInstance(loaded, NamespaceManager)

    def test_model_name_preserved(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertEqual(loaded.model_name, self.mgr.model_name)

    def test_M_preserved(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertEqual(loaded.M, self.mgr.M)

    def test_efConstruction_preserved(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertEqual(loaded.efConstruction, self.mgr.efConstruction)

    def test_namespaces_preserved(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertEqual(loaded.list_namespaces(), self.mgr.list_namespaces())

    def test_namespace_doc_counts_preserved(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        for ns in self.mgr.list_namespaces():
            self.assertEqual(loaded.namespace_len(ns), self.mgr.namespace_len(ns))

    def test_total_len_preserved(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertEqual(len(loaded), len(self.mgr))

    # --- Search results survive round-trip ---

    def test_search_results_match_after_load(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        orig = self.mgr.search("capital city", "cities", top_k=2)
        restored = loaded.search("capital city", "cities", top_k=2)
        self.assertEqual(
            [r["id"] for r in orig],
            [r["id"] for r in restored],
        )

    def test_search_text_correct_after_load(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        results = loaded.search("capital of France", "cities", top_k=1)
        self.assertEqual(results[0]["text"], "Paris is the capital of France")

    def test_namespaces_still_isolated_after_load(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        animal_results = loaded.search("dogs", "animals", top_k=1)
        science_results = loaded.search("dogs", "science", top_k=1)
        self.assertEqual(animal_results[0]["text"], "dogs and puppies are great pets")
        self.assertNotEqual(science_results[0]["text"], "dogs and puppies are great pets")

    # --- Auto-ID counter continuity ---

    def test_auto_id_continues_after_load(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        existing_ids = set(loaded._namespaces["cities"]._index.id_to_node.keys())
        new_id = loaded.add("Tokyo is the capital of Japan", "cities")
        self.assertNotIn(new_id, existing_ids)

    def test_new_namespace_addable_after_load(self):
        save_namespace_manager(self.mgr, self.path)
        loaded = load_namespace_manager(self.path)
        loaded.add("neural networks are powerful", "tech")
        results = loaded.search("neural networks", "tech", top_k=1)
        self.assertEqual(results[0]["text"], "neural networks are powerful")

    # --- Empty manager round-trip ---

    def test_empty_manager_round_trip(self):
        empty = NamespaceManager._from_embedder(self.mgr._embedder)
        save_namespace_manager(empty, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertEqual(loaded.list_namespaces(), [])
        self.assertEqual(len(loaded), 0)

    # --- Overwrite ---

    def test_save_overwrites_existing_file(self):
        save_namespace_manager(self.mgr, self.path)
        size1 = os.path.getsize(self.path)
        save_namespace_manager(self.mgr, self.path)
        size2 = os.path.getsize(self.path)
        self.assertEqual(size1, size2)

    # --- Custom params ---

    def test_custom_params_preserved(self):
        mgr = NamespaceManager._from_embedder(self.mgr._embedder, M=8, efConstruction=50)
        mgr.add("hello world", "ns")
        save_namespace_manager(mgr, self.path)
        loaded = load_namespace_manager(self.path)
        self.assertEqual(loaded.M, 8)
        self.assertEqual(loaded.efConstruction, 50)


class TestSaveErrors(unittest.TestCase):

    def setUp(self):
        self.path = _tmp_path()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_save_non_manager_raises_type_error(self):
        with self.assertRaises(TypeError):
            save_namespace_manager("not a manager", self.path)  # type: ignore

    def test_save_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            save_namespace_manager(None, self.path)  # type: ignore


class TestLoadErrors(unittest.TestCase):

    def test_load_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_namespace_manager("/tmp/neuroseek_ns_does_not_exist.nsmgr")


if __name__ == "__main__":
    unittest.main()
