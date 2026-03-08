"""Tests for SearchEngine backend='auto'/'hnswlib'/'hnsw' routing (Unit 2 of v0.3.0).

All tests use SearchEngine._from_embedder() to avoid re-loading the model,
but the backend-routing tests don't need embeddings at all and inspect only
the _index type and the .backend attribute.
"""

import unittest
import inspect
import pytest

from neuroseek.core.hnsw_index import HNSWIndex
from neuroseek.search_engine import SearchEngine, _VALID_BACKENDS

hnswlib = pytest.importorskip("hnswlib")
from neuroseek.core.hnswlib_index import HNSWLibIndex


# ---------------------------------------------------------------------------
# backend attribute — constructor (no model load needed)
# ---------------------------------------------------------------------------

class TestSearchEngineBackendConstructor(unittest.TestCase):

    def test_default_backend_is_auto(self):
        sig = inspect.signature(SearchEngine.__init__)
        self.assertEqual(sig.parameters["backend"].default, "auto")

    def test_invalid_backend_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            SearchEngine(backend=42)  # type: ignore

    def test_invalid_backend_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            SearchEngine(backend="unknown")

    def test_valid_backends_set(self):
        self.assertEqual(_VALID_BACKENDS, {"auto", "hnswlib", "hnsw"})


# ---------------------------------------------------------------------------
# backend routing — via _from_embedder (avoids model reload)
# ---------------------------------------------------------------------------

class TestSearchEngineBackendRouting(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder) -> None:
        self._embedder = embedder

    def test_backend_hnsw_creates_hnsw_index(self):
        e = SearchEngine._from_embedder(self._embedder, backend="hnsw")
        self.assertIsInstance(e._index, HNSWIndex)

    def test_backend_hnswlib_creates_hnswlib_index(self):
        e = SearchEngine._from_embedder(self._embedder, backend="hnswlib")
        self.assertIsInstance(e._index, HNSWLibIndex)

    def test_backend_auto_creates_hnswlib_index_when_available(self):
        # hnswlib is installed on this machine, so auto -> HNSWLibIndex
        e = SearchEngine._from_embedder(self._embedder, backend="auto")
        self.assertIsInstance(e._index, HNSWLibIndex)

    def test_backend_attribute_stored_hnsw(self):
        e = SearchEngine._from_embedder(self._embedder, backend="hnsw")
        self.assertEqual(e.backend, "hnsw")

    def test_backend_attribute_stored_hnswlib(self):
        e = SearchEngine._from_embedder(self._embedder, backend="hnswlib")
        self.assertEqual(e.backend, "hnswlib")

    def test_backend_attribute_stored_auto(self):
        e = SearchEngine._from_embedder(self._embedder, backend="auto")
        self.assertEqual(e.backend, "auto")

    def test_invalid_backend_in_from_embedder_raises(self):
        with self.assertRaises(ValueError):
            SearchEngine._from_embedder(self._embedder, backend="bad")

    def test_m_and_efconstruction_forwarded_to_hnswlib_index(self):
        e = SearchEngine._from_embedder(self._embedder, M=8, efConstruction=50, backend="hnswlib")
        self.assertEqual(e._index.M, 8)
        self.assertEqual(e._index.efConstruction, 50)

    def test_m_and_efconstruction_forwarded_to_hnsw_index(self):
        e = SearchEngine._from_embedder(self._embedder, M=8, efConstruction=50, backend="hnsw")
        self.assertEqual(e._index.M, 8)
        self.assertEqual(e._index.efConstruction, 50)


# ---------------------------------------------------------------------------
# End-to-end add/search with each explicit backend
# ---------------------------------------------------------------------------

class TestSearchEngineBackendEndToEnd(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder) -> None:
        self._embedder = embedder

    def _engine(self, backend: str) -> SearchEngine:
        return SearchEngine._from_embedder(self._embedder, backend=backend)

    def _populate(self, engine: SearchEngine) -> None:
        engine.add("The cat sat on the mat", id=1)
        engine.add("Dogs love to play fetch in the park", id=2)
        engine.add("Quantum computing uses qubits", id=3)

    def test_hnsw_backend_add_and_search(self):
        e = self._engine("hnsw")
        self._populate(e)
        results = e.search("cat mat", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_hnswlib_backend_add_and_search(self):
        e = self._engine("hnswlib")
        self._populate(e)
        results = e.search("cat mat", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_auto_backend_add_and_search(self):
        e = self._engine("auto")
        self._populate(e)
        results = e.search("cat mat", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_both_backends_return_same_top_result(self):
        """hnsw and hnswlib should agree on the best match for a clear query."""
        e_hnsw = self._engine("hnsw")
        e_hnswlib = self._engine("hnswlib")
        self._populate(e_hnsw)
        self._populate(e_hnswlib)
        r_hnsw = e_hnsw.search("quantum qubits", top_k=1)
        r_hnswlib = e_hnswlib.search("quantum qubits", top_k=1)
        self.assertEqual(r_hnsw[0]["id"], r_hnswlib[0]["id"])

    def test_len_works_with_hnswlib_backend(self):
        e = self._engine("hnswlib")
        self._populate(e)
        self.assertEqual(len(e), 3)

    def test_delete_works_with_hnswlib_backend(self):
        e = self._engine("hnswlib")
        self._populate(e)
        e.delete(1)
        self.assertEqual(len(e), 2)
        results = e.search("cat mat", top_k=3)
        ids = [r["id"] for r in results]
        self.assertNotIn(1, ids)


if __name__ == "__main__":
    unittest.main()
