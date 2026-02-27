import unittest
import random
from neuroseek import Vector
from neuroseek.hnsw_index import HNSWIndex


class TestHNSWIndex(unittest.TestCase):
    def test_constructor_default(self):
        idx = HNSWIndex()
        self.assertEqual(idx.M, 16)
        self.assertEqual(idx.efConstruction, 200)
        self.assertEqual(idx.maxLayers, 16)
        self.assertEqual(len(idx.layers), 0)
        self.assertEqual(len(idx.id_to_node), 0)
        self.assertIsNone(idx.entry_point)
        self.assertEqual(len(idx), 0)

    def test_constructor_custom_params(self):
        idx = HNSWIndex(M=8, efConstruction=100, maxLayers=8)
        self.assertEqual(idx.M, 8)
        self.assertEqual(idx.efConstruction, 100)
        self.assertEqual(idx.maxLayers, 8)

    def test_add_vector_basic(self):
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        returned_id = idx.add_vector(v, id=1)
        self.assertEqual(returned_id, 1)
        self.assertEqual(len(idx.id_to_node), 1)
        self.assertEqual(len(idx), 1)

    def test_add_vector_multiple(self):
        idx = HNSWIndex()
        for i in range(5):
            v = Vector(3)
            v.data = [i, i+1, i+2]
            idx.add_vector(v, id=i)
        self.assertEqual(len(idx.id_to_node), 5)
        self.assertEqual(len(idx), 5)

    def test_add_vector_auto_id(self):
        idx = HNSWIndex()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        id1 = idx.add_vector(v1)
        id2 = idx.add_vector(v2)
        self.assertEqual(id1, 0)
        self.assertEqual(id2, 1)

    def test_add_vector_invalid_vector_type(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.add_vector([1, 2, 3], id=1)

    def test_add_vector_invalid_id_type(self):
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        with self.assertRaises(TypeError):
            idx.add_vector(v, id="abc")

    def test_add_vector_duplicate_id(self):
        idx = HNSWIndex()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v1, id=1)
        with self.assertRaises(ValueError):
            idx.add_vector(v2, id=1)

    def test_get_vector_basic(self):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        retrieved = idx.get_vector(1)
        self.assertIsInstance(retrieved, Vector)
        self.assertEqual(list(retrieved.data), [1, 2, 3])

    def test_get_vector_multiple(self):
        random.seed(42)
        idx = HNSWIndex()
        for i in range(5):
            v = Vector(2)
            v.data = [i, i+1]
            idx.add_vector(v, id=i)
        retrieved = idx.get_vector(3)
        self.assertEqual(list(retrieved.data), [3, 4])

    def test_get_vector_nonexistent_raises(self):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        with self.assertRaises(ValueError):
            idx.get_vector(999)

    def test_get_vector_invalid_type_raises(self):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        with self.assertRaises(TypeError):
            idx.get_vector("abc")

    def test_delete_vector_basic(self):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        deleted = idx.delete_vector(1)
        self.assertIsInstance(deleted, Vector)
        self.assertEqual(len(idx), 0)

    def test_delete_vector_multiple(self):
        random.seed(42)
        idx = HNSWIndex()
        for i in range(5):
            v = Vector(2)
            v.data = [i, i+1]
            idx.add_vector(v, id=i)
        idx.delete_vector(2)
        self.assertEqual(len(idx), 4)

    def test_delete_vector_nonexistent_raises(self):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        with self.assertRaises(ValueError):
            idx.delete_vector(999)

    def test_delete_vector_invalid_type_raises(self):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        with self.assertRaises(TypeError):
            idx.delete_vector("abc")

    def test_delete_vector_from_empty_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(ValueError):
            idx.delete_vector(1)

    def test_search_basic(self):
        random.seed(42)
        idx = HNSWIndex()
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        idx.add_vector(v1, id=1)
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        idx.add_vector(v2, id=2)
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx.search(query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 1)

    def test_search_multiple_results(self):
        random.seed(42)
        idx = HNSWIndex()
        for i in range(10):
            v = Vector(2)
            v.data = [i + 1, 0]
            idx.add_vector(v, id=i)
        query = Vector(2)
        query.data = [1, 0]
        results = idx.search(query, top_k=3)
        self.assertGreaterEqual(len(results), 1)

    def test_search_orthogonal_vectors(self):
        random.seed(42)
        idx = HNSWIndex()
        v1 = Vector(2)
        v1.data = [1, 0]
        idx.add_vector(v1, id=1)
        v2 = Vector(2)
        v2.data = [0, 1]
        idx.add_vector(v2, id=2)
        query = Vector(2)
        query.data = [1, 0]
        results = idx.search(query, top_k=2)
        self.assertGreaterEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
