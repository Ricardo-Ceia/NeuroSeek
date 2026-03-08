"""Comprehensive tests for HNSWLibIndex (hnswlib-backed HNSW index).

All tests are skipped automatically if hnswlib is not installed.
"""

import unittest

import pytest

hnswlib = pytest.importorskip("hnswlib")

from neuroseek.core.hnswlib_index import HNSWLibIndex
from neuroseek.core.vector import Vector


def make_vector(data):
    v = Vector(len(data))
    v.data = list(data)
    return v


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestHNSWLibIndexConstructor(unittest.TestCase):
    def test_default_params(self):
        idx = HNSWLibIndex()
        self.assertEqual(idx.M, 16)
        self.assertEqual(idx.efConstruction, 200)
        self.assertEqual(idx.maxLayers, 16)

    def test_custom_params(self):
        idx = HNSWLibIndex(M=8, efConstruction=100, maxLayers=8)
        self.assertEqual(idx.M, 8)
        self.assertEqual(idx.efConstruction, 100)
        self.assertEqual(idx.maxLayers, 8)

    def test_initial_len_is_zero(self):
        self.assertEqual(len(HNSWLibIndex()), 0)

    def test_initial_dim_is_zero(self):
        idx = HNSWLibIndex()
        self.assertEqual(idx._dim, 0)

    def test_initial_next_id_is_zero(self):
        idx = HNSWLibIndex()
        self.assertEqual(idx._next_id, 0)

    def test_initial_num_vectors_is_zero(self):
        idx = HNSWLibIndex()
        self.assertEqual(idx.num_vectors, 0)

    def test_initial_index_is_none(self):
        idx = HNSWLibIndex()
        self.assertIsNone(idx._index)


# ---------------------------------------------------------------------------
# add_vector — basic insertion
# ---------------------------------------------------------------------------

class TestHNSWLibIndexAddVector(unittest.TestCase):
    def test_add_single_vector_returns_given_id(self):
        idx = HNSWLibIndex()
        v = make_vector([1.0, 2.0, 3.0])
        returned = idx.add_vector(v, id=1)
        self.assertEqual(returned, 1)

    def test_add_single_vector_increases_len(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0, 3.0]), id=1)
        self.assertEqual(len(idx), 1)

    def test_add_multiple_vectors_len(self):
        idx = HNSWLibIndex()
        for i in range(10):
            idx.add_vector(make_vector([float(i), float(i + 1)]), id=i)
        self.assertEqual(len(idx), 10)

    def test_auto_id_starts_at_zero(self):
        idx = HNSWLibIndex()
        returned = idx.add_vector(make_vector([1.0, 0.0]))
        self.assertEqual(returned, 0)

    def test_auto_ids_are_sequential(self):
        idx = HNSWLibIndex()
        ids = [idx.add_vector(make_vector([float(i), 1.0])) for i in range(5)]
        self.assertEqual(ids, [0, 1, 2, 3, 4])

    def test_auto_id_does_not_collide_with_manual_id(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=0)
        auto_id = idx.add_vector(make_vector([0.0, 1.0]))
        self.assertNotEqual(auto_id, 0)

    def test_auto_id_no_collision_after_deletion(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=0)
        idx.delete_vector(0)
        auto_id = idx.add_vector(make_vector([0.0, 1.0]))
        # Must not reuse the soft-deleted ID 0
        self.assertNotEqual(auto_id, 0)

    def test_invalid_vector_type_raises_type_error(self):
        idx = HNSWLibIndex()
        with self.assertRaises(TypeError):
            idx.add_vector([1, 2, 3], id=1)

    def test_invalid_id_type_raises_type_error(self):
        idx = HNSWLibIndex()
        with self.assertRaises(TypeError):
            idx.add_vector(make_vector([1.0, 2.0]), id="abc")

    def test_duplicate_id_raises_value_error(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0]), id=1)
        with self.assertRaises(ValueError):
            idx.add_vector(make_vector([3.0, 4.0]), id=1)

    def test_dimension_mismatch_raises_value_error(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0, 3.0]), id=1)
        with self.assertRaises(ValueError):
            idx.add_vector(make_vector([1.0, 2.0]), id=2)

    def test_zero_norm_vector_raises_value_error(self):
        idx = HNSWLibIndex()
        with self.assertRaises(ValueError):
            idx.add_vector(make_vector([0.0, 0.0, 0.0]), id=1)

    def test_dim_is_set_after_first_insertion(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0, 3.0]), id=1)
        self.assertEqual(idx._dim, 3)

    def test_index_is_initialised_after_first_insertion(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        self.assertIsNotNone(idx._index)

    def test_explicit_id_advances_next_id(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=100)
        self.assertGreater(idx._next_id, 100)

    def test_capacity_doubles_when_exceeded(self):
        """Insert more vectors than the initial capacity to trigger resize."""
        from neuroseek.core.hnswlib_index import _INITIAL_CAPACITY
        idx = HNSWLibIndex()
        for i in range(_INITIAL_CAPACITY + 1):
            # Use unit vectors spread around the unit circle to stay non-zero
            import math
            angle = 2 * math.pi * i / (_INITIAL_CAPACITY + 1)
            idx.add_vector(make_vector([math.cos(angle), math.sin(angle)]))
        self.assertGreater(idx._capacity, _INITIAL_CAPACITY)
        self.assertEqual(len(idx), _INITIAL_CAPACITY + 1)


# ---------------------------------------------------------------------------
# add_vectors — batch insertion
# ---------------------------------------------------------------------------

class TestHNSWLibIndexAddVectors(unittest.TestCase):
    def test_batch_with_explicit_ids(self):
        idx = HNSWLibIndex()
        vecs = [make_vector([1.0, 2.0]), make_vector([3.0, 4.0]), make_vector([5.0, 6.0])]
        ids = idx.add_vectors(vecs, [10, 20, 30])
        self.assertEqual(ids, [10, 20, 30])
        self.assertEqual(len(idx), 3)

    def test_batch_auto_ids(self):
        idx = HNSWLibIndex()
        vecs = [make_vector([1.0, 2.0]), make_vector([3.0, 4.0])]
        ids = idx.add_vectors(vecs)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(len(idx), 2)

    def test_batch_empty_list(self):
        idx = HNSWLibIndex()
        ids = idx.add_vectors([])
        self.assertEqual(ids, [])
        self.assertEqual(len(idx), 0)

    def test_batch_invalid_vectors_type_raises(self):
        idx = HNSWLibIndex()
        with self.assertRaises(TypeError):
            idx.add_vectors("not a list")

    def test_batch_invalid_ids_type_raises(self):
        idx = HNSWLibIndex()
        with self.assertRaises(TypeError):
            idx.add_vectors([make_vector([1.0, 2.0])], "not a list")

    def test_batch_length_mismatch_raises(self):
        idx = HNSWLibIndex()
        with self.assertRaises(ValueError):
            idx.add_vectors([make_vector([1.0, 2.0])], [1, 2])

    def test_batch_invalid_id_element_type_raises(self):
        idx = HNSWLibIndex()
        with self.assertRaises(TypeError):
            idx.add_vectors([make_vector([1.0, 2.0])], ["abc"])


# ---------------------------------------------------------------------------
# get_vector
# ---------------------------------------------------------------------------

class TestHNSWLibIndexGetVector(unittest.TestCase):
    def test_get_returns_original_vector(self):
        idx = HNSWLibIndex()
        v = make_vector([1.0, 2.0, 3.0])
        idx.add_vector(v, id=5)
        retrieved = idx.get_vector(5)
        self.assertIsInstance(retrieved, Vector)
        self.assertEqual(list(retrieved.data), [1.0, 2.0, 3.0])

    def test_get_after_multiple_inserts(self):
        idx = HNSWLibIndex()
        for i in range(5):
            idx.add_vector(make_vector([float(i), float(i + 1)]), id=i)
        retrieved = idx.get_vector(3)
        self.assertEqual(list(retrieved.data), [3.0, 4.0])

    def test_get_nonexistent_raises_value_error(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0]), id=1)
        with self.assertRaises(ValueError):
            idx.get_vector(999)

    def test_get_invalid_type_raises_type_error(self):
        idx = HNSWLibIndex()
        with self.assertRaises(TypeError):
            idx.get_vector("abc")


# ---------------------------------------------------------------------------
# delete_vector
# ---------------------------------------------------------------------------

class TestHNSWLibIndexDeleteVector(unittest.TestCase):
    def test_delete_returns_correct_vector(self):
        idx = HNSWLibIndex()
        v = make_vector([7.0, 8.0, 9.0])
        idx.add_vector(v, id=3)
        deleted = idx.delete_vector(3)
        self.assertIsInstance(deleted, Vector)
        self.assertEqual(list(deleted.data), [7.0, 8.0, 9.0])

    def test_delete_decrements_len(self):
        idx = HNSWLibIndex()
        for i in range(5):
            idx.add_vector(make_vector([float(i + 1), 1.0]), id=i)
        idx.delete_vector(2)
        self.assertEqual(len(idx), 4)

    def test_deleted_id_not_retrievable(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0]), id=1)
        idx.delete_vector(1)
        with self.assertRaises(ValueError):
            idx.get_vector(1)

    def test_deleted_id_tracked_in_deleted_ids(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0]), id=42)
        idx.delete_vector(42)
        self.assertIn(42, idx._deleted_ids)

    def test_deleted_id_removed_from_id_to_vector(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 2.0]), id=7)
        idx.delete_vector(7)
        self.assertNotIn(7, idx._id_to_vector)

    def test_delete_nonexistent_raises_value_error(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        with self.assertRaises(ValueError):
            idx.delete_vector(999)

    def test_delete_invalid_type_raises_type_error(self):
        idx = HNSWLibIndex()
        with self.assertRaises(TypeError):
            idx.delete_vector("abc")

    def test_delete_from_empty_raises_value_error(self):
        idx = HNSWLibIndex()
        with self.assertRaises(ValueError):
            idx.delete_vector(1)

    def test_delete_all_then_add_works(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        idx.add_vector(make_vector([0.0, 1.0]), id=2)
        idx.delete_vector(1)
        idx.delete_vector(2)
        self.assertEqual(len(idx), 0)
        new_id = idx.add_vector(make_vector([1.0, 1.0]), id=10)
        self.assertEqual(new_id, 10)
        self.assertEqual(len(idx), 1)

    def test_num_vectors_never_goes_negative(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        idx.delete_vector(1)
        self.assertGreaterEqual(idx.num_vectors, 0)

    def test_next_id_never_decremented_on_delete(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]))  # _next_id becomes 1
        before = idx._next_id
        idx.delete_vector(0)
        self.assertGreaterEqual(idx._next_id, before)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestHNSWLibIndexSearch(unittest.TestCase):
    def _build_index(self, n=20, dim=4):
        idx = HNSWLibIndex(M=16, efConstruction=200)
        for i in range(1, n + 1):
            data = [float(i * 10 + j) for j in range(dim)]
            idx.add_vector(make_vector(data), id=i)
        return idx

    def test_search_returns_list(self):
        idx = self._build_index()
        results = idx.search(make_vector([10.0, 11.0, 12.0, 13.0]), top_k=3)
        self.assertIsInstance(results, list)

    def test_search_returns_tuples_of_id_and_float(self):
        idx = self._build_index()
        results = idx.search(make_vector([10.0, 11.0, 12.0, 13.0]), top_k=3)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], int)
            self.assertIsInstance(item[1], float)

    def test_search_scores_in_range(self):
        idx = self._build_index()
        results = idx.search(make_vector([10.0, 11.0, 12.0, 13.0]), top_k=5)
        for _, score in results:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_search_results_sorted_descending(self):
        idx = self._build_index()
        results = idx.search(make_vector([50.0, 51.0, 52.0, 53.0]), top_k=5)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_returns_at_most_top_k(self):
        idx = self._build_index(n=20)
        results = idx.search(make_vector([10.0, 11.0, 12.0, 13.0]), top_k=5)
        self.assertLessEqual(len(results), 5)

    def test_search_top_k_larger_than_index_returns_all(self):
        idx = HNSWLibIndex()
        for i in range(1, 4):
            idx.add_vector(make_vector([float(i), 1.0]), id=i)
        results = idx.search(make_vector([1.0, 1.0]), top_k=100)
        self.assertEqual(len(results), 3)

    def test_search_empty_index_returns_empty(self):
        idx = HNSWLibIndex()
        results = idx.search(make_vector([1.0, 0.0]), top_k=3)
        self.assertEqual(results, [])

    def test_search_exact_match_is_first(self):
        """Searching for a stored direction should return it as top result."""
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0, 0.0]), id=1)
        idx.add_vector(make_vector([0.0, 1.0, 0.0]), id=2)
        idx.add_vector(make_vector([0.0, 0.0, 1.0]), id=3)
        results = idx.search(make_vector([1.0, 0.0, 0.0]), top_k=1)
        self.assertEqual(results[0][0], 1)

    def test_search_accuracy_vs_brute_force(self):
        """hnswlib nearest neighbour must match the ground truth for a clear case."""
        idx = HNSWLibIndex(M=16, efConstruction=200)
        for i in range(1, 31):
            idx.add_vector(make_vector([float(i * 10), 1.0, 0.0]), id=i)
        # Query closest to id=5 ([50.0, 1.0, 0.0])
        results = idx.search(make_vector([51.0, 1.0, 0.0]), top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 5,
                         f"Expected nearest id=5, got id={results[0][0]}")

    def test_search_deleted_vector_not_returned(self):
        """A soft-deleted vector must not appear in search results."""
        idx = HNSWLibIndex(M=16, efConstruction=200)
        # Insert two orthogonal vectors; search will retrieve both without deletion
        idx.add_vector(make_vector([1.0, 0.0, 0.0]), id=1)  # target
        idx.add_vector(make_vector([0.0, 1.0, 0.0]), id=2)
        idx.add_vector(make_vector([0.0, 0.0, 1.0]), id=3)
        # Delete the best match for the query
        idx.delete_vector(1)
        results = idx.search(make_vector([1.0, 0.0, 0.0]), top_k=3)
        result_ids = [r[0] for r in results]
        self.assertNotIn(1, result_ids, "Deleted vector should not appear in search results")

    def test_search_invalid_query_type_raises(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        with self.assertRaises(TypeError):
            idx.search([1, 0], top_k=1)

    def test_search_invalid_top_k_type_raises(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        with self.assertRaises(TypeError):
            idx.search(make_vector([1.0, 0.0]), top_k="3")

    def test_search_top_k_zero_raises(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        with self.assertRaises(ValueError):
            idx.search(make_vector([1.0, 0.0]), top_k=0)

    def test_search_top_k_negative_raises(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        with self.assertRaises(ValueError):
            idx.search(make_vector([1.0, 0.0]), top_k=-1)

    def test_search_zero_norm_query_raises(self):
        idx = HNSWLibIndex()
        idx.add_vector(make_vector([1.0, 0.0]), id=1)
        with self.assertRaises(ValueError):
            idx.search(make_vector([0.0, 0.0]), top_k=1)

    def test_multiple_searches_consistent(self):
        """Two identical searches must return the same result."""
        idx = self._build_index(n=20, dim=4)
        query = make_vector([50.0, 51.0, 52.0, 53.0])
        r1 = idx.search(query, top_k=3)
        r2 = idx.search(query, top_k=3)
        self.assertEqual(r1, r2)

    def test_ef_param_accepted(self):
        """Passing ef should not raise and should still return results."""
        idx = self._build_index(n=10)
        results = idx.search(make_vector([10.0, 11.0, 12.0, 13.0]), top_k=3, ef=50)
        self.assertGreater(len(results), 0)


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestHNSWLibIndexLen(unittest.TestCase):
    def test_len_empty(self):
        self.assertEqual(len(HNSWLibIndex()), 0)

    def test_len_after_inserts(self):
        idx = HNSWLibIndex()
        for i in range(7):
            idx.add_vector(make_vector([float(i + 1), 1.0]), id=i)
        self.assertEqual(len(idx), 7)

    def test_len_after_delete(self):
        idx = HNSWLibIndex()
        for i in range(5):
            idx.add_vector(make_vector([float(i + 1), 1.0]), id=i)
        idx.delete_vector(2)
        self.assertEqual(len(idx), 4)

    def test_len_matches_num_vectors(self):
        idx = HNSWLibIndex()
        for i in range(6):
            idx.add_vector(make_vector([float(i + 1), 1.0]), id=i)
        idx.delete_vector(3)
        self.assertEqual(len(idx), idx.num_vectors)


# ---------------------------------------------------------------------------
# Auto-ID collision safety
# ---------------------------------------------------------------------------

class TestHNSWLibIndexAutoId(unittest.TestCase):
    def test_auto_ids_unique(self):
        idx = HNSWLibIndex()
        ids = [idx.add_vector(make_vector([float(i + 1), 1.0])) for i in range(5)]
        self.assertEqual(len(set(ids)), 5)

    def test_auto_id_skips_occupied_manual_ids(self):
        idx = HNSWLibIndex()
        for i in range(3):
            idx.add_vector(make_vector([float(i + 1), 1.0]), id=i)
        auto_id = idx.add_vector(make_vector([4.0, 1.0]))
        self.assertNotIn(auto_id, [0, 1, 2])

    def test_auto_id_no_collision_mixed_manual_and_auto_after_delete(self):
        idx = HNSWLibIndex()
        a0 = idx.add_vector(make_vector([1.0, 0.0]))          # auto → 0
        idx.add_vector(make_vector([0.0, 1.0]), id=1)          # manual 1
        idx.add_vector(make_vector([1.0, 1.0]))                # auto → 2
        idx.delete_vector(a0)                                   # soft-delete 0
        a_new = idx.add_vector(make_vector([0.5, 0.5]))
        self.assertNotEqual(a_new, a0)
        self.assertIn(a_new, idx._id_to_vector)
        all_ids = list(idx._id_to_vector.keys())
        self.assertEqual(len(all_ids), len(set(all_ids)))


# ---------------------------------------------------------------------------
# Large-scale smoke test
# ---------------------------------------------------------------------------

class TestHNSWLibIndexLargeScale(unittest.TestCase):
    def test_insert_and_search_1000_vectors(self):
        """Insert 1000 384-d random-ish vectors and do a nearest-neighbour query."""
        import math
        dim = 32  # keep it light for a unit test
        n = 1000
        idx = HNSWLibIndex(M=16, efConstruction=100)
        for i in range(n):
            # Deterministic pseudo-random unit vector
            data = [math.sin(i * 0.1 + j) for j in range(dim)]
            idx.add_vector(make_vector(data), id=i)

        self.assertEqual(len(idx), n)

        # Query: should return top_k results, all with valid ids
        query = make_vector([math.sin(j) for j in range(dim)])
        results = idx.search(query, top_k=10)
        self.assertEqual(len(results), 10)
        for r_id, score in results:
            self.assertGreaterEqual(r_id, 0)
            self.assertLess(r_id, n)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
