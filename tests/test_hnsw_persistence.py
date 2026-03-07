import random
from pathlib import Path
from neuroseek import Vector, HNSWIndex
from neuroseek.persistence.hnsw_persistence import save_hnsw_index, load_hnsw_index


class TestHNSWPersistence:
    def test_save_and_load_basic(self, tmp_path):
        random.seed(42)
        idx = HNSWIndex()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        path = tmp_path / "index.pkl"
        save_hnsw_index(idx, str(path))

        idx2 = load_hnsw_index(str(path), HNSWIndex)
        assert len(idx2) == 1

    def test_save_and_load_multiple_vectors(self, tmp_path):
        random.seed(42)
        idx = HNSWIndex()
        for i in range(5):
            v = Vector(3)
            v.data = [i, i + 1, i + 2]
            idx.add_vector(v, id=i)
        path = tmp_path / "index.pkl"
        save_hnsw_index(idx, str(path))

        idx2 = load_hnsw_index(str(path), HNSWIndex)
        assert len(idx2) == 5

    def test_save_and_load_preserves_params(self, tmp_path):
        random.seed(42)
        idx = HNSWIndex(M=8, efConstruction=100, maxLayers=8)
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, id=1)
        path = tmp_path / "index.pkl"
        save_hnsw_index(idx, str(path))

        idx2 = load_hnsw_index(str(path), HNSWIndex)
        assert idx2.M == 8
        assert idx2.efConstruction == 100
        assert idx2.maxLayers == 8

    def test_save_and_load_search_results_preserved(self, tmp_path):
        random.seed(42)
        idx = HNSWIndex()
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        idx.add_vector(v1, id=1)
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        idx.add_vector(v2, id=2)
        path = tmp_path / "index.pkl"
        save_hnsw_index(idx, str(path))

        idx2 = load_hnsw_index(str(path), HNSWIndex)
        query = Vector(3)
        query.data = [1, 0, 0]
        results = idx2.search(query, top_k=2)
        assert results[0][0] == 1

    def test_save_and_load_empty_index(self, tmp_path):
        idx = HNSWIndex()
        path = tmp_path / "index.pkl"
        save_hnsw_index(idx, str(path))

        idx2 = load_hnsw_index(str(path), HNSWIndex)
        assert len(idx2) == 0

    def test_save_file_not_found_raises(self, tmp_path):
        idx = HNSWIndex()
        import pytest
        with pytest.raises(FileNotFoundError):
            load_hnsw_index(str(tmp_path / "nonexistent.pkl"), HNSWIndex)
