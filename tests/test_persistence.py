import unittest
import os
from neuroseek import Vector, Index
from neuroseek.persistence import save_index, load_index


class TestPersistence(unittest.TestCase):
    def test_save_and_load_basic(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        self.assertEqual(len(idx2.vectors), 1)
        self.assertEqual(idx2.vectors[0][0], 1)
        self.assertEqual(list(idx2.vectors[0][1].data), [1, 2, 3])
        os.remove('test_save.pkl')

    def test_save_and_load_multiple_vectors(self):
        idx = Index()
        for i in range(5):
            v = Vector(3)
            v.data = [i, i+1, i+2]
            idx.add_vector(v, i)
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        self.assertEqual(len(idx2.vectors), 5)
        self.assertEqual(len(idx2.id_to_index), 5)
        os.remove('test_save.pkl')

    def test_save_and_load_preserves_id_to_index(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 42)
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        self.assertEqual(idx2.id_to_index[42], 0)
        os.remove('test_save.pkl')

    def test_save_and_load_preserves_next_id(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 5)
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        self.assertEqual(idx2._next_id, 0)
        os.remove('test_save.pkl')

    def test_save_and_load_with_auto_id(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        idx.add_vector(v1)
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v2)
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        self.assertEqual(idx2.vectors[0][0], 0)
        self.assertEqual(idx2.vectors[1][0], 1)
        os.remove('test_save.pkl')

    def test_save_and_load_empty_index(self):
        idx = Index()
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        self.assertEqual(len(idx2.vectors), 0)
        self.assertEqual(idx2.id_to_index, {})
        os.remove('test_save.pkl')

    def test_save_and_load_with_floats(self):
        idx = Index()
        v = Vector(2)
        v.data = [1.5, 2.5]
        idx.add_vector(v, 1)
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        self.assertAlmostEqual(idx2.vectors[0][1].data[0], 1.5)
        self.assertAlmostEqual(idx2.vectors[0][1].data[1], 2.5)
        os.remove('test_save.pkl')

    def test_save_file_not_found_raises(self):
        idx = Index()
        with self.assertRaises(FileNotFoundError):
            load_index(idx, 'nonexistent.pkl')

    def test_save_and_load_search_results_preserved(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        idx.add_vector(v1, 1)
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        idx.add_vector(v2, 2)
        save_index(idx, 'test_save.pkl')
        
        idx2 = Index()
        load_index(idx2, 'test_save.pkl')
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx2.search(query, 2)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[0][1], 1.0)
        os.remove('test_save.pkl')


if __name__ == "__main__":
    unittest.main()
