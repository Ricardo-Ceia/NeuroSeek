"""Performance regression and correctness tests for the optimised HNSWIndex.

These tests verify that:
1. Bulk insertion + search complete well within a tight time budget.
2. Search results match brute-force cosine rankings (recall@K check).
"""

import random
import time
import math
import unittest

import numpy as np

from neuroseek.core.hnsw_index import HNSWIndex
from neuroseek.core.vector import Vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_vector(dim: int, rng: random.Random) -> Vector:
    v = Vector(dim)
    for i in range(dim):
        v[i] = rng.gauss(0, 1)
    return v


def _brute_force_top_k(
    query: Vector,
    vectors: list[tuple[int, Vector]],
    k: int,
) -> list[int]:
    """Return top-k IDs by cosine similarity (highest first)."""
    scores = [(vid, query.cosine_similarity(v)) for vid, v in vectors]
    scores.sort(key=lambda x: -x[1])
    return [vid for vid, _ in scores[:k]]


# ---------------------------------------------------------------------------
# Performance regression
# ---------------------------------------------------------------------------

class TestHNSWPerformance(unittest.TestCase):
    """Wall-clock time must stay below conservative thresholds even on slow CI."""

    DIM = 64        # smaller than 384 to keep test fast while exercising the path
    N = 1000        # number of vectors to insert
    TOP_K = 10
    TIME_LIMIT = 5.0  # seconds — very conservative; optimised code should be <1 s

    def setUp(self):
        self.rng = random.Random(42)
        self.vectors = [_make_random_vector(self.DIM, self.rng) for _ in range(self.N)]
        self.queries = [_make_random_vector(self.DIM, self.rng) for _ in range(10)]

    def test_bulk_insert_and_search_within_time_limit(self):
        idx = HNSWIndex(M=16, efConstruction=100, maxLayers=8)

        t0 = time.perf_counter()
        for v in self.vectors:
            idx.add_vector(v)
        for q in self.queries:
            idx.search(q, top_k=self.TOP_K, ef=50)
        elapsed = time.perf_counter() - t0

        self.assertLess(
            elapsed,
            self.TIME_LIMIT,
            f"insert+search took {elapsed:.2f}s — exceeds {self.TIME_LIMIT}s budget",
        )
        self.assertEqual(len(idx), self.N)

    def test_delete_bulk_within_time_limit(self):
        """Deleting half the index must also be fast (O(degree) path)."""
        idx = HNSWIndex(M=16, efConstruction=100, maxLayers=8)
        ids = []
        for v in self.vectors:
            ids.append(idx.add_vector(v))

        to_delete = ids[: self.N // 2]

        t0 = time.perf_counter()
        for vid in to_delete:
            idx.delete_vector(vid)
        elapsed = time.perf_counter() - t0

        self.assertLess(
            elapsed,
            self.TIME_LIMIT,
            f"bulk delete took {elapsed:.2f}s — exceeds {self.TIME_LIMIT}s budget",
        )
        self.assertEqual(len(idx), self.N - len(to_delete))


# ---------------------------------------------------------------------------
# Correctness: recall vs brute-force
# ---------------------------------------------------------------------------

class TestHNSWRecall(unittest.TestCase):
    """HNSW approximate search should recover >=70% of true top-K neighbours."""

    DIM = 64
    N = 500
    TOP_K = 10
    MIN_RECALL = 0.85

    def setUp(self):
        self.rng = random.Random(7)
        self.pairs: list[tuple[int, Vector]] = []
        idx = HNSWIndex(M=16, efConstruction=200, maxLayers=8)
        for i in range(self.N):
            v = _make_random_vector(self.DIM, self.rng)
            idx.add_vector(v, id=i)
            self.pairs.append((i, v))
        self.idx = idx
        self.queries = [_make_random_vector(self.DIM, self.rng) for _ in range(20)]

    def test_recall_at_k(self):
        recalls = []
        for q in self.queries:
            true_ids = set(_brute_force_top_k(q, self.pairs, self.TOP_K))
            approx = self.idx.search(q, top_k=self.TOP_K, ef=self.TOP_K * 5)
            approx_ids = {vid for vid, _ in approx}
            hit = len(true_ids & approx_ids)
            recalls.append(hit / self.TOP_K)

        mean_recall = sum(recalls) / len(recalls)
        self.assertGreaterEqual(
            mean_recall,
            self.MIN_RECALL,
            f"Mean Recall@{self.TOP_K} = {mean_recall:.2f} < {self.MIN_RECALL}",
        )


# ---------------------------------------------------------------------------
# Numpy matrix bookkeeping correctness
# ---------------------------------------------------------------------------

class TestHNSWMatrixBookkeeping(unittest.TestCase):
    """Verify that the numpy matrix stays in sync with the node registry."""

    DIM = 8

    def _make_v(self, vals):
        v = Vector(self.DIM)
        for i, x in enumerate(vals):
            v[i] = x
        return v

    def test_row_freed_after_delete(self):
        idx = HNSWIndex()
        v = self._make_v([1.0] + [0.0] * (self.DIM - 1))
        vid = idx.add_vector(v)
        self.assertIn(vid, idx._id_to_row)
        idx.delete_vector(vid)
        self.assertNotIn(vid, idx._id_to_row)
        self.assertEqual(len(idx._free_rows), 1)

    def test_row_reused_after_delete(self):
        idx = HNSWIndex()
        v1 = self._make_v([1.0] + [0.0] * (self.DIM - 1))
        v2 = self._make_v([0.0, 1.0] + [0.0] * (self.DIM - 2))
        id1 = idx.add_vector(v1)
        row1 = idx._id_to_row[id1]
        idx.delete_vector(id1)
        id2 = idx.add_vector(v2)
        row2 = idx._id_to_row[id2]
        # The freed row should have been reused.
        self.assertEqual(row1, row2)

    def test_reverse_adj_cleared_after_delete(self):
        idx = HNSWIndex(M=4, efConstruction=10)
        vecs = []
        for i in range(10):
            v = Vector(self.DIM)
            v[i % self.DIM] = 1.0
            vecs.append(idx.add_vector(v))
        target = vecs[0]
        idx.delete_vector(target)
        # No surviving node should have target_id in its connections.
        for nid, node in idx.id_to_node.items():
            for layer_conns in node.connections.values():
                self.assertNotIn(target, [cid for cid, _ in layer_conns])
        # reverse_adj entry for target should be gone.
        self.assertNotIn(target, idx._reverse_adj)

    def test_dimension_mismatch_raises(self):
        idx = HNSWIndex()
        v1 = Vector(4)
        v1[0] = 1.0
        idx.add_vector(v1)
        v2 = Vector(8)
        v2[0] = 1.0
        with self.assertRaises(ValueError):
            idx.add_vector(v2)


if __name__ == "__main__":
    unittest.main()
