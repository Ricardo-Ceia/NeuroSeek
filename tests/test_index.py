import unittest
from neuroseek import Vector, Index


class TestIndex(unittest.TestCase):
    def test_index_constructor(self):
        idx = Index()
        self.assertEqual(idx.vectors, [])
        self.assertEqual(idx.id_to_index, {})
        self.assertEqual(idx._next_id, 0)

    def test_add_vector_basic(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        self.assertEqual(len(idx.vectors), 1)
        self.assertEqual(idx.vectors[0][0], 1)
        self.assertEqual(list(idx.vectors[0][1].data), [1, 2, 3])

    def test_add_vector_multiple(self):
        idx = Index()
        for i in range(5):
            v = Vector(2)
            v.data = [i, i+1]
            idx.add_vector(v, i)
        self.assertEqual(len(idx.vectors), 5)
        self.assertEqual(len(idx.id_to_index), 5)

    def test_add_vector_auto_generates_id(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v1)
        idx.add_vector(v2)
        self.assertEqual(idx.vectors[0][0], 0)
        self.assertEqual(idx.vectors[1][0], 1)

    def test_add_vector_populates_id_to_index(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 42)
        self.assertEqual(idx.id_to_index[42], 0)

    def test_add_vector_duplicate_id_raises(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v1, 1)
        with self.assertRaises(ValueError):
            idx.add_vector(v2, 1)

    def test_add_vector_non_vector_raises(self):
        idx = Index()
        with self.assertRaises(TypeError):
            idx.add_vector([1, 2, 3], 1)

    def test_add_vector_invalid_id_type_raises(self):
        idx = Index()
        v = Vector(3)
        with self.assertRaises(TypeError):
            idx.add_vector(v, "abc")

    def test_add_vector_float_id_raises(self):
        idx = Index()
        v = Vector(3)
        with self.assertRaises(TypeError):
            idx.add_vector(v, 1.5)

    def test_add_vector_empty_vector(self):
        idx = Index()
        v = Vector(0)
        idx.add_vector(v, 1)
        self.assertEqual(len(idx.vectors), 1)

    def test_add_vector_with_existing_auto_id(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v1, 0)
        idx.add_vector(v2)
        self.assertEqual(idx.vectors[1][0], 1)

    def test_add_vector_returns_provided_id(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        returned_id = idx.add_vector(v, 42)
        self.assertEqual(returned_id, 42)

    def test_add_vector_returns_auto_generated_id(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        id1 = idx.add_vector(v1)
        id2 = idx.add_vector(v2)
        self.assertEqual(id1, 0)
        self.assertEqual(id2, 1)

    def test_search_basic(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        idx.add_vector(v1, 1)
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        idx.add_vector(v2, 2)
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx.search(query, 2)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[0][1], 1.0)

    def test_search_returns_of_tuples(self):
        idx = Index()
        v = Vector(2)
        v.data = [1, 0]
        idx.add_vector(v, 1)
        query = Vector(2)
        query.data = [1, 0]
        results = idx.search(query, 1)
        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0], tuple)

    def test_search_sorted_by_similarity(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        idx.add_vector(v1, 1)
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        idx.add_vector(v2, 2)
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx.search(query, 2)
        self.assertGreaterEqual(results[0][1], results[1][1])

    def test_search_top_k_less_than_total(self):
        idx = Index()
        for i in range(10):
            v = Vector(2)
            v.data = [i + 1, i + 2]
            idx.add_vector(v, i)
        query = Vector(2)
        query.data = [1, 2]
        results = idx.search(query, 3)
        self.assertEqual(len(results), 3)

    def test_search_top_k_greater_than_total(self):
        idx = Index()
        for i in range(3):
            v = Vector(2)
            v.data = [i + 1, i + 2]
            idx.add_vector(v, i)
        query = Vector(2)
        query.data = [1, 2]
        results = idx.search(query, 10)
        self.assertEqual(len(results), 3)

    def test_search_empty_index(self):
        idx = Index()
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx.search(query, 5)
        self.assertEqual(results, [])

    def test_search_non_vector_query_raises(self):
        idx = Index()
        with self.assertRaises(TypeError):
            idx.search([1, 2, 3], 5)

    def test_search_invalid_top_k_type_raises(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        query = Vector(3)
        query.data = [1, 0, 0]
        with self.assertRaises(TypeError):
            idx.search(query, "5")

    def test_search_negative_top_k_raises(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        query = Vector(3)
        query.data = [1, 0, 0]
        with self.assertRaises(ValueError):
            idx.search(query, -1)

    def test_search_dimension_mismatch_raises(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        query = Vector(2)
        query.data = [1, 0]
        with self.assertRaises(ValueError):
            idx.search(query, 5)

    def test_search_empty_query_vector_raises(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        query = Vector(0)
        with self.assertRaises(ValueError):
            idx.search(query, 5)

    def test_search_identical_vectors(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        query = Vector(3)
        query.data = [1, 2, 3]
        results = idx.search(query, 1)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[0][1], 1.0)

    def test_search_orthogonal_vectors(self):
        idx = Index()
        v1 = Vector(2)
        v1.data = [1, 0]
        idx.add_vector(v1, 1)
        v2 = Vector(2)
        v2.data = [0, 1]
        idx.add_vector(v2, 2)
        query = Vector(2)
        query.data = [1, 0]
        results = idx.search(query, 2)
        self.assertEqual(results[1][1], 0.0)

    def test_search_top_k_zero(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx.search(query, 0)
        self.assertEqual(results, [])

    def test_delete_vector_basic(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        deleted = idx.delete_vector(1)
        self.assertEqual(deleted[0], 1)
        self.assertEqual(len(idx.vectors), 0)

    def test_delete_vector_updates_id_to_index(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        idx.add_vector(v1, 1)
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v2, 2)
        idx.delete_vector(1)
        self.assertEqual(idx.id_to_index[2], 0)

    def test_delete_vector_middle_element(self):
        idx = Index()
        for i in range(5):
            v = Vector(2)
            v.data = [i, i+1]
            idx.add_vector(v, i)
        idx.delete_vector(2)
        self.assertEqual(len(idx.vectors), 4)
        self.assertEqual(idx.id_to_index[3], 2)

    def test_delete_vector_nonexistent_raises(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        with self.assertRaises(ValueError):
            idx.delete_vector(999)

    def test_delete_vector_no_id_raises(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        with self.assertRaises(ValueError):
            idx.delete_vector(None)

    def test_delete_vector_invalid_type_raises(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        with self.assertRaises(TypeError):
            idx.delete_vector("abc")

    def test_delete_vector_from_empty_index_raises(self):
        idx = Index()
        with self.assertRaises(ValueError):
            idx.delete_vector(1)

    def test_delete_vector_returns_vector(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        deleted = idx.delete_vector(1)
        self.assertIsInstance(deleted[1], Vector)

    def test_delete_last_vector(self):
        idx = Index()
        for i in range(3):
            v = Vector(2)
            v.data = [i, i+1]
            idx.add_vector(v, i)
        idx.delete_vector(2)
        self.assertEqual(len(idx.vectors), 2)
        self.assertNotIn(2, idx.id_to_index)


if __name__ == "__main__":
    unittest.main()
