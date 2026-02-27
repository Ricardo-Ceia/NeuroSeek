import unittest
import os
import random
from neuroseek import Vector
from neuroseek.hnsw_index import HNSWIndex
from neuroseek.hnsw_persistence import save_hnsw_index, load_hnsw_index


class TestHNSWPersistence(unittest.TestCase):
    def test_save_and_load_basic(self):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        save_hnsw_index(idx, 'test_hnsw.pkl')
        
        idx2 = load_hnsw_index('test_hnsw.pkl', HNSWIndex)
        self.assertEqual(len(idx2), 1)
        os.remove('test_hnsw.pkl')

    def test_save_and_load_multiple_vectors(self):
        random.seed(42)
        idx = HNSWIndex()
        for i in range(5):
            v = Vector(3)
            v.data = [i, i+1, i+2]
            idx.add_vector(v, id=i)
        save_hnsw_index(idx, 'test_hnsw.pkl')
        
        idx2 = load_hnsw_index('test_hnsw.pkl', HNSWIndex)
        self.assertEqual(len(idx2), 5)
        os.remove('test_hnsw.pkl')

    def test_save_and_load_preserves_params(self):
        random.seed(42)
        idx = HNSWIndex(M=8, efConstruction=100, maxLayers=8)
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        save_hnsw_index(idx, 'test_hnsw.pkl')
        
        idx2 = load_hnsw_index('test_hnsw.pkl', HNSWIndex)
        self.assertEqual(idx2.M, 8)
        self.assertEqual(idx2.efConstruction, 100)
        self.assertEqual(idx2.maxLayers, 8)
        os.remove('test_hnsw.pkl')

    def test_save_and_load_search_results_preserved(self):
        random.seed(42)
        idx = HNSWIndex()
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        idx.add_vector(v1, id=1)
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        idx.add_vector(v2, id=2)
        save_hnsw_index(idx, 'test_hnsw.pkl')
        
        idx2 = load_hnsw_index('test_hnsw.pkl', HNSWIndex)
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx2.search(query, top_k=2)
        self.assertEqual(results[0][0], 1)
        os.remove('test_hnsw.pkl')

    def test_save_and_load_empty_index(self):
        idx = HNSWIndex()
        save_hnsw_index(idx, 'test_hnsw.pkl')
        
        idx2 = load_hnsw_index('test_hnsw.pkl', HNSWIndex)
        self.assertEqual(len(idx2), 0)
        os.remove('test_hnsw.pkl')

    def test_save_file_not_found_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(FileNotFoundError):
            load_hnsw_index('nonexistent.pkl', HNSWIndex)


if __name__ == "__main__":
    unittest.main()
