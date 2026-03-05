"""
Comprehensive tests for save_search_engine / load_search_engine.

No mocking — real embeddings, real file I/O, real round-trips.
The model is loaded once per session via the conftest fixture.
"""

import os
import tempfile
import unittest
import pytest

from neuroseek.embedder import Embedder
from neuroseek.search_engine import SearchEngine
from neuroseek.persistence.search_engine_persistence import load_search_engine, save_search_engine


_DOCS = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Madrid is the capital of Spain",
    "Rome is the capital of Italy",
    "dogs and puppies are great pets",
]


def _tmp_path() -> str:
    """Return a fresh temporary file path (file does not yet exist)."""
    fd, path = tempfile.mkstemp(suffix=".neuroseek")
    os.close(fd)
    return path


class TestSaveLoadRoundTrip(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._shared_engine = SearchEngine._from_embedder(embedder)
        for text in _DOCS:
            self._shared_engine.add(text)
        self.path = _tmp_path()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    # --- File is created ---

    def test_save_creates_file(self):
        os.remove(self.path)
        save_search_engine(self._shared_engine, self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_saved_file_is_non_empty(self):
        save_search_engine(self._shared_engine, self.path)
        self.assertGreater(os.path.getsize(self.path), 0)

    # --- Basic round-trip ---

    def test_load_returns_search_engine(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        self.assertIsInstance(loaded, SearchEngine)

    def test_loaded_length_matches_original(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        self.assertEqual(len(loaded), len(self._shared_engine))

    def test_loaded_model_name_preserved(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        self.assertEqual(loaded.model_name, self._shared_engine.model_name)

    def test_loaded_M_preserved(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        self.assertEqual(loaded.M, self._shared_engine.M)

    def test_loaded_efConstruction_preserved(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        self.assertEqual(loaded.efConstruction, self._shared_engine.efConstruction)

    # --- Search results survive round-trip ---

    def test_search_results_same_after_load(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        original_results = self._shared_engine.search("capital city of Europe", top_k=3)
        loaded_results = loaded.search("capital city of Europe", top_k=3)
        self.assertEqual(
            [r["id"] for r in original_results],
            [r["id"] for r in loaded_results],
        )

    def test_search_texts_preserved(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        results = loaded.search("capital of France", top_k=1)
        self.assertEqual(results[0]["text"], "Paris is the capital of France")

    def test_scores_approximately_equal_after_load(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        orig = self._shared_engine.search("dogs", top_k=1)
        load = loaded.search("dogs", top_k=1)
        self.assertAlmostEqual(orig[0]["score"], load[0]["score"], places=4)

    # --- Auto-ID counter continuity ---

    def test_auto_id_continues_after_load(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        existing_ids = set(loaded._index.id_to_node.keys())
        new_id = loaded.add("a brand new document")
        self.assertNotIn(new_id, existing_ids)

    def test_search_includes_new_doc_after_load(self):
        save_search_engine(self._shared_engine, self.path)
        loaded = load_search_engine(self.path)
        loaded.add("cats are independent animals")
        results = loaded.search("feline pets", top_k=5)
        texts = [r["text"] for r in results]
        self.assertIn("cats are independent animals", texts)

    # --- Overwrite ---

    def test_save_overwrites_existing_file(self):
        save_search_engine(self._shared_engine, self.path)
        size1 = os.path.getsize(self.path)
        save_search_engine(self._shared_engine, self.path)
        size2 = os.path.getsize(self.path)
        self.assertEqual(size1, size2)

    # --- Custom M / efConstruction preserved ---

    def test_custom_params_preserved(self):
        engine = SearchEngine._from_embedder(
            self._shared_engine._embedder, M=8, efConstruction=50
        )
        engine.add("hello world")
        save_search_engine(engine, self.path)
        loaded = load_search_engine(self.path)
        self.assertEqual(loaded.M, 8)
        self.assertEqual(loaded.efConstruction, 50)

    # --- Empty engine round-trip ---

    def test_empty_engine_round_trip(self):
        empty = SearchEngine._from_embedder(self._shared_engine._embedder)
        save_search_engine(empty, self.path)
        loaded = load_search_engine(self.path)
        self.assertEqual(len(loaded), 0)
        self.assertEqual(loaded.search("anything"), [])

    # --- Delete before save ---

    def test_deleted_doc_absent_after_load(self):
        engine = SearchEngine._from_embedder(self._shared_engine._embedder)
        engine.add("keep this", id=0)
        engine.add("delete this", id=1)
        engine.delete(1)
        save_search_engine(engine, self.path)
        loaded = load_search_engine(self.path)
        results = loaded.search("delete this", top_k=5)
        ids = [r["id"] for r in results]
        self.assertNotIn(1, ids)


class TestSaveErrors(unittest.TestCase):

    def setUp(self):
        self.path = _tmp_path()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_save_non_engine_raises_type_error(self):
        with self.assertRaises(TypeError):
            save_search_engine("not an engine", self.path)  # type: ignore

    def test_save_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            save_search_engine(None, self.path)  # type: ignore


class TestLoadErrors(unittest.TestCase):

    def test_load_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_search_engine("/tmp/neuroseek_does_not_exist_xyz.neuroseek")


if __name__ == "__main__":
    unittest.main()
