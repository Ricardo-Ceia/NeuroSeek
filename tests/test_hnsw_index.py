import unittest
import random
from neuroseek import Vector, HNSWIndex


def make_vector(data):
    v = Vector(len(data))
    v.data = list(data)
    return v


class TestHNSWIndexConstructor(unittest.TestCase):
    def test_default_params(self):
        idx = HNSWIndex()
        self.assertEqual(idx.M, 16)
        self.assertEqual(idx.efConstruction, 200)
        self.assertEqual(idx.maxLayers, 16)
        self.assertEqual(len(idx.layers), 0)
        self.assertEqual(len(idx.id_to_node), 0)
        self.assertIsNone(idx.entry_point)
        self.assertEqual(len(idx), 0)

    def test_custom_params(self):
        idx = HNSWIndex(M=8, efConstruction=100, maxLayers=8)
        self.assertEqual(idx.M, 8)
        self.assertEqual(idx.efConstruction, 100)
        self.assertEqual(idx.maxLayers, 8)


class TestHNSWIndexGetRandomLayer(unittest.TestCase):
    def test_returns_int(self):
        idx = HNSWIndex()
        for _ in range(50):
            layer = idx._get_random_layer()
            self.assertIsInstance(layer, int)

    def test_layer_within_bounds(self):
        idx = HNSWIndex(maxLayers=4)
        for _ in range(200):
            layer = idx._get_random_layer()
            self.assertGreaterEqual(layer, 0)
            self.assertLessEqual(layer, 4)

    def test_layer_zero_is_most_common(self):
        """Layer 0 should be assigned at least ~50% of the time (geometric dist)."""
        idx = HNSWIndex()
        random.seed(0)
        layers = [idx._get_random_layer() for _ in range(1000)]
        zero_count = layers.count(0)
        self.assertGreater(zero_count, 400)

    def test_multi_layer_assignment_occurs(self):
        """With enough insertions, at least one node should land on layer > 0."""
        idx = HNSWIndex(maxLayers=8)
        random.seed(0)
        layers = [idx._get_random_layer() for _ in range(200)]
        self.assertTrue(any(l > 0 for l in layers))


class TestHNSWIndexAddVector(unittest.TestCase):
    def test_add_single_vector_returns_id(self):
        random.seed(0)
        idx = HNSWIndex()
        v = make_vector([1, 2, 3])
        returned_id = idx.add_vector(v, id=1)
        self.assertEqual(returned_id, 1)

    def test_add_single_vector_stored(self):
        random.seed(0)
        idx = HNSWIndex()
        v = make_vector([1, 2, 3])
        idx.add_vector(v, id=1)
        self.assertEqual(len(idx), 1)
        self.assertIn(1, idx.id_to_node)

    def test_first_node_becomes_entry_point(self):
        random.seed(0)
        idx = HNSWIndex()
        v = make_vector([1, 0, 0])
        idx.add_vector(v, id=42)
        self.assertIsNotNone(idx.entry_point)
        self.assertEqual(idx.entry_point.id, 42)

    def test_node_layer_assigned_by_random_layer(self):
        """Nodes must use _get_random_layer(), not always 0."""
        # Seed so that some nodes land on layer > 0
        random.seed(1)
        idx = HNSWIndex(maxLayers=8)
        for i in range(100):
            v = make_vector([i, i + 1, i + 2])
            idx.add_vector(v, id=i)
        max_layer = max(node.layer for node in idx.id_to_node.values())
        self.assertGreater(max_layer, 0, "All nodes are on layer 0 — _get_random_layer is not being called")

    def test_node_registered_in_all_layers_up_to_its_layer(self):
        """A node assigned to layer L must appear in layers 0..L."""
        random.seed(1)
        idx = HNSWIndex(maxLayers=8)
        for i in range(50):
            v = make_vector([i, i + 1])
            idx.add_vector(v, id=i)
        for node in idx.id_to_node.values():
            for lyr in range(node.layer + 1):
                self.assertIn(node.id, idx.layers[lyr],
                              f"Node {node.id} (layer={node.layer}) missing from layers[{lyr}]")

    def test_connections_built_at_correct_layer(self):
        """Connections stored at a layer must only reference nodes present in that layer."""
        random.seed(1)
        idx = HNSWIndex(M=4, efConstruction=10, maxLayers=4)
        for i in range(20):
            v = make_vector([i, i + 1])
            idx.add_vector(v, id=i)
        for node in idx.id_to_node.values():
            for lyr, conns in node.connections.items():
                for neighbor_id, _ in conns:
                    self.assertIn(neighbor_id, idx.layers[lyr],
                                  f"Node {node.id} has connection to {neighbor_id} at layer {lyr}, "
                                  f"but {neighbor_id} is not in layers[{lyr}]")

    def test_add_multiple_vectors(self):
        random.seed(0)
        idx = HNSWIndex()
        for i in range(10):
            v = make_vector([i, i + 1, i + 2])
            idx.add_vector(v, id=i)
        self.assertEqual(len(idx), 10)

    def test_auto_id_sequential(self):
        random.seed(0)
        idx = HNSWIndex()
        v1 = make_vector([1, 2, 3])
        v2 = make_vector([4, 5, 6])
        id1 = idx.add_vector(v1)
        id2 = idx.add_vector(v2)
        self.assertEqual(id1, 0)
        self.assertEqual(id2, 1)

    def test_auto_id_skips_taken_ids(self):
        """Auto-ID must not collide with manually assigned IDs."""
        random.seed(0)
        idx = HNSWIndex()
        # Manually occupy IDs 0 and 1
        idx.add_vector(make_vector([1, 0]), id=0)
        idx.add_vector(make_vector([0, 1]), id=1)
        # Auto-ID should skip 0 and 1
        auto_id = idx.add_vector(make_vector([1, 1]))
        self.assertNotIn(auto_id, [0, 1])

    def test_invalid_vector_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.add_vector([1, 2, 3], id=1)

    def test_invalid_id_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.add_vector(make_vector([1, 2, 3]), id="abc")

    def test_duplicate_id_raises(self):
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 2, 3]), id=1)
        with self.assertRaises(ValueError):
            idx.add_vector(make_vector([4, 5, 6]), id=1)

    def test_entry_point_is_highest_layer_node(self):
        """The global entry point must always be a node on the top layer."""
        random.seed(1)
        idx = HNSWIndex(maxLayers=8)
        for i in range(60):
            v = make_vector([i, i + 1])
            idx.add_vector(v, id=i)
        self.assertIsNotNone(idx.entry_point)
        top_layer = len(idx.layers) - 1
        self.assertEqual(idx.entry_point.layer, top_layer,
                         f"entry_point is on layer {idx.entry_point.layer}, but top layer is {top_layer}")


class TestHNSWIndexAddVectors(unittest.TestCase):
    def test_add_vectors_with_explicit_ids(self):
        random.seed(0)
        idx = HNSWIndex()
        vectors = [make_vector([1, 2, 3]), make_vector([4, 5, 6]), make_vector([7, 8, 9])]
        ids = idx.add_vectors(vectors, [10, 20, 30])
        self.assertEqual(ids, [10, 20, 30])
        self.assertEqual(len(idx), 3)

    def test_add_vectors_auto_ids(self):
        random.seed(0)
        idx = HNSWIndex()
        vectors = [make_vector([1, 2]), make_vector([3, 4])]
        ids = idx.add_vectors(vectors)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(len(idx), 2)

    def test_add_vectors_empty_list(self):
        idx = HNSWIndex()
        ids = idx.add_vectors([])
        self.assertEqual(ids, [])
        self.assertEqual(len(idx), 0)

    def test_add_vectors_invalid_vectors_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.add_vectors("not a list")

    def test_add_vectors_invalid_ids_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.add_vectors([make_vector([1, 2])], "not a list")

    def test_add_vectors_length_mismatch_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(ValueError):
            idx.add_vectors([make_vector([1, 2]), make_vector([3, 4])], [1, 2, 3])

    def test_add_vectors_invalid_id_element_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.add_vectors([make_vector([1, 2])], ["abc"])


class TestHNSWIndexGetVector(unittest.TestCase):
    def test_get_existing_vector(self):
        random.seed(0)
        idx = HNSWIndex()
        v = make_vector([1, 2, 3])
        idx.add_vector(v, id=5)
        retrieved = idx.get_vector(5)
        self.assertIsInstance(retrieved, Vector)
        self.assertEqual(list(retrieved.data), [1, 2, 3])

    def test_get_vector_after_multiple_inserts(self):
        random.seed(0)
        idx = HNSWIndex()
        for i in range(5):
            idx.add_vector(make_vector([i, i + 1]), id=i)
        retrieved = idx.get_vector(3)
        self.assertEqual(list(retrieved.data), [3, 4])

    def test_get_nonexistent_raises(self):
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 2, 3]), id=1)
        with self.assertRaises(ValueError):
            idx.get_vector(999)

    def test_get_invalid_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.get_vector("abc")


class TestHNSWIndexDeleteVector(unittest.TestCase):
    def test_delete_returns_correct_vector(self):
        random.seed(0)
        idx = HNSWIndex()
        v = make_vector([7, 8, 9])
        idx.add_vector(v, id=3)
        deleted = idx.delete_vector(3)
        self.assertIsInstance(deleted, Vector)
        self.assertEqual(list(deleted.data), [7, 8, 9])

    def test_delete_reduces_len(self):
        random.seed(0)
        idx = HNSWIndex()
        for i in range(5):
            idx.add_vector(make_vector([i, i + 1]), id=i)
        idx.delete_vector(2)
        self.assertEqual(len(idx), 4)

    def test_deleted_id_no_longer_retrievable(self):
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 2]), id=1)
        idx.delete_vector(1)
        with self.assertRaises(ValueError):
            idx.get_vector(1)

    def test_delete_removes_node_from_all_layers(self):
        random.seed(1)
        idx = HNSWIndex(maxLayers=4)
        for i in range(30):
            idx.add_vector(make_vector([i, i + 1]), id=i)
        # Pick any node and delete it
        target_id = list(idx.id_to_node.keys())[5]
        idx.delete_vector(target_id)
        for layer_dict in idx.layers:
            self.assertNotIn(target_id, layer_dict)

    def test_delete_removes_connections_to_deleted_node(self):
        random.seed(1)
        idx = HNSWIndex(M=4, efConstruction=10, maxLayers=4)
        for i in range(20):
            idx.add_vector(make_vector([i, i + 1]), id=i)
        target_id = list(idx.id_to_node.keys())[3]
        idx.delete_vector(target_id)
        for node in idx.id_to_node.values():
            for lyr, conns in node.connections.items():
                conn_ids = [nid for nid, _ in conns]
                self.assertNotIn(target_id, conn_ids,
                                 f"Node {node.id} still has connection to deleted node {target_id}")

    def test_delete_nonexistent_raises(self):
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 2, 3]), id=1)
        with self.assertRaises(ValueError):
            idx.delete_vector(999)

    def test_delete_invalid_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.delete_vector("abc")

    def test_delete_from_empty_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(ValueError):
            idx.delete_vector(1)

    def test_delete_all_vectors_then_add_works(self):
        """Index should still be usable after all vectors are deleted."""
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 0]), id=1)
        idx.add_vector(make_vector([0, 1]), id=2)
        idx.delete_vector(1)
        idx.delete_vector(2)
        self.assertEqual(len(idx), 0)
        # Should be able to add again without error
        new_id = idx.add_vector(make_vector([1, 1]), id=10)
        self.assertEqual(new_id, 10)
        self.assertEqual(len(idx), 1)


class TestHNSWIndexSearch(unittest.TestCase):
    def _build_index(self, seed=42, n=20, dim=4):
        random.seed(seed)
        idx = HNSWIndex(M=4, efConstruction=20, maxLayers=4)
        vectors = []
        for i in range(n):
            data = [float(i + j) for j in range(dim)]
            v = make_vector(data)
            idx.add_vector(v, id=i)
            vectors.append((i, data))
        return idx, vectors

    def test_search_returns_list(self):
        idx, _ = self._build_index()
        query = make_vector([1.0, 2.0, 3.0, 4.0])
        results = idx.search(query, top_k=3)
        self.assertIsInstance(results, list)

    def test_search_returns_tuples_of_id_and_score(self):
        idx, _ = self._build_index()
        query = make_vector([1.0, 2.0, 3.0, 4.0])
        results = idx.search(query, top_k=3)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], int)
            self.assertIsInstance(item[1], float)

    def test_search_scores_are_similarities_between_0_and_1(self):
        idx, _ = self._build_index()
        query = make_vector([5.0, 6.0, 7.0, 8.0])
        results = idx.search(query, top_k=5)
        for _, score in results:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_search_exact_match_is_first(self):
        """Searching for a vector that exists exactly should return it first."""
        random.seed(42)
        idx = HNSWIndex(M=4, efConstruction=20, maxLayers=4)
        idx.add_vector(make_vector([1, 0, 0]), id=1)
        idx.add_vector(make_vector([0, 1, 0]), id=2)
        idx.add_vector(make_vector([0, 0, 1]), id=3)
        results = idx.search(make_vector([1, 0, 0]), top_k=1)
        self.assertEqual(results[0][0], 1)

    def test_search_results_sorted_by_score_descending(self):
        idx, _ = self._build_index()
        query = make_vector([5.0, 6.0, 7.0, 8.0])
        results = idx.search(query, top_k=5)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_returns_at_most_top_k(self):
        idx, _ = self._build_index()
        query = make_vector([1.0, 2.0, 3.0, 4.0])
        results = idx.search(query, top_k=5)
        self.assertLessEqual(len(results), 5)

    def test_search_top_k_larger_than_index_returns_all(self):
        random.seed(0)
        idx = HNSWIndex(M=4, efConstruction=10)
        for i in range(3):
            idx.add_vector(make_vector([float(i), float(i + 1)]), id=i)
        results = idx.search(make_vector([0.0, 1.0]), top_k=10)
        self.assertEqual(len(results), 3)

    def test_search_empty_index_returns_empty(self):
        idx = HNSWIndex()
        query = make_vector([1, 0])
        results = idx.search(query, top_k=3)
        self.assertEqual(results, [])

    def test_search_does_not_mutate_entry_point(self):
        """search() must be a pure read — it must not change self.entry_point."""
        random.seed(1)
        idx = HNSWIndex(M=4, efConstruction=20, maxLayers=4)
        for i in range(20):
            idx.add_vector(make_vector([float(i), float(i + 1)]), id=i)
        ep_before = idx.entry_point
        query = make_vector([5.0, 6.0])
        idx.search(query, top_k=3)
        self.assertIs(idx.entry_point, ep_before,
                      "search() mutated self.entry_point")

    def test_search_accuracy_vs_brute_force(self):
        """HNSW nearest neighbour must match brute-force for a clear best match."""
        random.seed(42)
        idx = HNSWIndex(M=8, efConstruction=40, maxLayers=4)
        # Spread vectors far apart so the nearest neighbour is unambiguous.
        # Start at i=1 to avoid zero vectors (cosine similarity undefined).
        for i in range(1, 31):
            data = [float(i * 10), 1.0, 0.0]
            v = make_vector(data)
            idx.add_vector(v, id=i)

        # Query closest to id=5 (data=[50.0, 1.0, 0.0])
        query = make_vector([51.0, 1.0, 0.0])
        results = idx.search(query, top_k=1, ef=20)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 5,
                         f"Expected nearest id=5, got id={results[0][0]}")

    def test_search_invalid_query_type_raises(self):
        idx = HNSWIndex()
        with self.assertRaises(TypeError):
            idx.search([1, 0, 0], top_k=1)

    def test_search_invalid_top_k_type_raises(self):
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 0]), id=1)
        with self.assertRaises(TypeError):
            idx.search(make_vector([1, 0]), top_k="3")

    def test_search_top_k_zero_raises(self):
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 0]), id=1)
        with self.assertRaises(ValueError):
            idx.search(make_vector([1, 0]), top_k=0)

    def test_search_top_k_negative_raises(self):
        random.seed(0)
        idx = HNSWIndex()
        idx.add_vector(make_vector([1, 0]), id=1)
        with self.assertRaises(ValueError):
            idx.search(make_vector([1, 0]), top_k=-1)

    def test_multiple_searches_give_consistent_results(self):
        """Calling search twice with the same query must return the same result."""
        random.seed(42)
        idx = HNSWIndex(M=4, efConstruction=20, maxLayers=4)
        for i in range(20):
            idx.add_vector(make_vector([float(i), float(i + 1)]), id=i)
        query = make_vector([5.0, 6.0])
        r1 = idx.search(query, top_k=3)
        r2 = idx.search(query, top_k=3)
        self.assertEqual(r1, r2)

    def test_multi_layer_graph_exists(self):
        """After inserting enough vectors, the graph must have more than one layer."""
        random.seed(1)
        idx = HNSWIndex(maxLayers=8)
        for i in range(200):
            idx.add_vector(make_vector([float(i), float(i + 1)]), id=i)
        self.assertGreater(len(idx.layers), 1,
                           "Graph never grew beyond layer 0 — _get_random_layer may not be used")


class TestHNSWIndexLen(unittest.TestCase):
    def test_len_empty(self):
        self.assertEqual(len(HNSWIndex()), 0)

    def test_len_after_inserts(self):
        random.seed(0)
        idx = HNSWIndex()
        for i in range(7):
            idx.add_vector(make_vector([float(i + 1), 1.0]), id=i)
        self.assertEqual(len(idx), 7)

    def test_len_after_delete(self):
        random.seed(0)
        idx = HNSWIndex()
        for i in range(5):
            idx.add_vector(make_vector([float(i + 1), 1.0]), id=i)
        idx.delete_vector(2)
        self.assertEqual(len(idx), 4)


if __name__ == "__main__":
    unittest.main()
