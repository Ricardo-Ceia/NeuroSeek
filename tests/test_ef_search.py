"""Tests for ef_search parameter on SearchEngine.search() (Unit 4 of v0.3.0)."""

import unittest
import pytest

from neuroseek.search_engine import SearchEngine


class TestEfSearchValidation(unittest.TestCase):
    """Validation tests — no embedder needed."""

    def test_ef_search_param_exists_on_search(self):
        import inspect
        sig = inspect.signature(SearchEngine.search)
        self.assertIn("ef_search", sig.parameters)
        self.assertIsNone(sig.parameters["ef_search"].default)


class TestEfSearchBehaviour(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder) -> None:
        self._embedder = embedder

    def _engine(self, backend="hnswlib") -> SearchEngine:
        e = SearchEngine._from_embedder(self._embedder, backend=backend)
        e.add("The cat sat on the mat", id=1)
        e.add("Dogs love to play fetch in the park", id=2)
        e.add("Quantum computing uses qubits", id=3)
        return e

    def test_ef_search_none_works(self):
        e = self._engine()
        results = e.search("cat", top_k=1, ef_search=None)
        self.assertEqual(len(results), 1)

    def test_ef_search_int_works_hnswlib(self):
        e = self._engine("hnswlib")
        results = e.search("cat", top_k=1, ef_search=50)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_ef_search_int_works_hnsw(self):
        e = self._engine("hnsw")
        results = e.search("cat", top_k=1, ef_search=50)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_ef_search_non_int_raises(self):
        e = self._engine()
        with self.assertRaises(TypeError):
            e.search("cat", top_k=1, ef_search="50")

    def test_ef_search_zero_raises(self):
        e = self._engine()
        with self.assertRaises(ValueError):
            e.search("cat", top_k=1, ef_search=0)

    def test_ef_search_negative_raises(self):
        e = self._engine()
        with self.assertRaises(ValueError):
            e.search("cat", top_k=1, ef_search=-1)

    def test_ef_search_with_filter(self):
        e = self._engine()
        # add a doc with metadata so filter is exercised
        e.add("A fluffy Persian cat", id=4, metadata={"type": "cat"})
        results = e.search("cat", top_k=5, filter={"type": "cat"}, ef_search=100)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 4)

    def test_high_ef_search_same_top_result(self):
        """ef_search=1000 should still return the same best match."""
        e = self._engine("hnswlib")
        r_default = e.search("quantum qubits", top_k=1)
        r_high_ef = e.search("quantum qubits", top_k=1, ef_search=1000)
        self.assertEqual(r_default[0]["id"], r_high_ef[0]["id"])


if __name__ == "__main__":
    unittest.main()
