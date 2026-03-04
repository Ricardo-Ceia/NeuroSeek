"""
Comprehensive tests for neuroseek.NamespaceManager.

No mocking — real embeddings, real search.
A shared manager is built once at module level for read-only tests.
"""

import unittest
from neuroseek.namespace_manager import NamespaceManager


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestNamespaceManagerInit(unittest.TestCase):

    def test_default_model_name(self):
        mgr = NamespaceManager()
        self.assertEqual(mgr.model_name, "multi-qa-MiniLM-L6-cos-v1")

    def test_default_M(self):
        self.assertEqual(NamespaceManager().M, 16)

    def test_default_efConstruction(self):
        self.assertEqual(NamespaceManager().efConstruction, 200)

    def test_custom_params_stored(self):
        mgr = NamespaceManager(M=8, efConstruction=50)
        self.assertEqual(mgr.M, 8)
        self.assertEqual(mgr.efConstruction, 50)

    def test_initial_len_is_zero(self):
        self.assertEqual(len(NamespaceManager()), 0)

    def test_initial_namespaces_empty(self):
        self.assertEqual(NamespaceManager().list_namespaces(), [])

    def test_model_name_not_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            NamespaceManager(model_name=42)  # type: ignore

    def test_model_name_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            NamespaceManager(model_name="")

    def test_M_not_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            NamespaceManager(M=1.5)  # type: ignore

    def test_M_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            NamespaceManager(M=0)

    def test_efConstruction_not_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            NamespaceManager(efConstruction=10.0)  # type: ignore

    def test_efConstruction_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            NamespaceManager(efConstruction=0)


# ---------------------------------------------------------------------------
# create_namespace / delete_namespace / list_namespaces
# ---------------------------------------------------------------------------

class TestNamespaceManagement(unittest.TestCase):

    def setUp(self):
        self.mgr = NamespaceManager()

    def test_create_namespace_appears_in_list(self):
        self.mgr.create_namespace("ns1")
        self.assertIn("ns1", self.mgr.list_namespaces())

    def test_list_namespaces_sorted(self):
        self.mgr.create_namespace("zebra")
        self.mgr.create_namespace("apple")
        self.assertEqual(self.mgr.list_namespaces(), ["apple", "zebra"])

    def test_create_duplicate_raises_value_error(self):
        self.mgr.create_namespace("ns1")
        with self.assertRaises(ValueError):
            self.mgr.create_namespace("ns1")

    def test_create_namespace_non_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.mgr.create_namespace(42)  # type: ignore

    def test_create_namespace_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mgr.create_namespace("")

    def test_create_namespace_whitespace_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mgr.create_namespace("   ")

    def test_delete_namespace_removes_from_list(self):
        self.mgr.create_namespace("ns1")
        self.mgr.delete_namespace("ns1")
        self.assertNotIn("ns1", self.mgr.list_namespaces())

    def test_delete_missing_namespace_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.mgr.delete_namespace("does_not_exist")

    def test_delete_namespace_removes_docs_from_total_len(self):
        self.mgr.add("hello", "ns1")
        self.mgr.delete_namespace("ns1")
        self.assertEqual(len(self.mgr), 0)

    def test_multiple_namespaces_listed(self):
        for name in ["a", "b", "c"]:
            self.mgr.create_namespace(name)
        self.assertEqual(self.mgr.list_namespaces(), ["a", "b", "c"])


# ---------------------------------------------------------------------------
# add() / add_batch()
# ---------------------------------------------------------------------------

class TestNamespaceManagerAdd(unittest.TestCase):

    def setUp(self):
        self.mgr = NamespaceManager()

    def test_add_returns_int(self):
        self.assertIsInstance(self.mgr.add("hello", "ns"), int)

    def test_add_auto_creates_namespace(self):
        self.mgr.add("hello", "new_ns")
        self.assertIn("new_ns", self.mgr.list_namespaces())

    def test_add_first_id_is_zero(self):
        self.assertEqual(self.mgr.add("hello", "ns"), 0)

    def test_add_explicit_id(self):
        self.assertEqual(self.mgr.add("hello", "ns", id=42), 42)

    def test_namespace_len_increases(self):
        self.mgr.add("hello", "ns")
        self.assertEqual(self.mgr.namespace_len("ns"), 1)

    def test_total_len_increases(self):
        self.mgr.add("hello", "ns1")
        self.mgr.add("world", "ns2")
        self.assertEqual(len(self.mgr), 2)

    def test_ids_independent_across_namespaces(self):
        id1 = self.mgr.add("hello", "ns1")
        id2 = self.mgr.add("world", "ns2")
        # Both get id=0 — namespaces are isolated
        self.assertEqual(id1, 0)
        self.assertEqual(id2, 0)

    def test_add_invalid_text_raises(self):
        with self.assertRaises(TypeError):
            self.mgr.add(123, "ns")  # type: ignore

    def test_add_empty_text_raises(self):
        with self.assertRaises(ValueError):
            self.mgr.add("", "ns")

    def test_add_invalid_namespace_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.mgr.add("hello", 42)  # type: ignore

    def test_add_batch_returns_list(self):
        self.assertIsInstance(self.mgr.add_batch(["a", "b"], "ns"), list)

    def test_add_batch_auto_creates_namespace(self):
        self.mgr.add_batch(["a", "b"], "fresh")
        self.assertIn("fresh", self.mgr.list_namespaces())

    def test_add_batch_correct_count(self):
        ids = self.mgr.add_batch(["a", "b", "c"], "ns")
        self.assertEqual(len(ids), 3)

    def test_add_batch_explicit_ids(self):
        ids = self.mgr.add_batch(["a", "b"], "ns", ids=[10, 20])
        self.assertEqual(ids, [10, 20])


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestNamespaceManagerDelete(unittest.TestCase):

    def setUp(self):
        self.mgr = NamespaceManager()

    def test_delete_reduces_namespace_len(self):
        self.mgr.add("hello", "ns", id=0)
        self.mgr.delete(0, "ns")
        self.assertEqual(self.mgr.namespace_len("ns"), 0)

    def test_delete_reduces_total_len(self):
        self.mgr.add("hello", "ns", id=0)
        self.mgr.delete(0, "ns")
        self.assertEqual(len(self.mgr), 0)

    def test_delete_missing_id_raises(self):
        self.mgr.create_namespace("ns")
        with self.assertRaises(Exception):
            self.mgr.delete(99, "ns")

    def test_delete_missing_namespace_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.mgr.delete(0, "ghost")

    def test_delete_only_affects_target_namespace(self):
        self.mgr.add("hello", "ns1", id=0)
        self.mgr.add("hello", "ns2", id=0)
        self.mgr.delete(0, "ns1")
        self.assertEqual(self.mgr.namespace_len("ns2"), 1)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestNamespaceManagerSearch(unittest.TestCase):

    def setUp(self):
        self.mgr = NamespaceManager()

    def test_search_returns_list(self):
        self.mgr.add("hello world", "ns")
        self.assertIsInstance(self.mgr.search("hello", "ns"), list)

    def test_search_empty_namespace_returns_empty(self):
        self.mgr.create_namespace("empty")
        self.assertEqual(self.mgr.search("anything", "empty"), [])

    def test_search_missing_namespace_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.mgr.search("query", "ghost")

    def test_search_result_has_id_text_score(self):
        self.mgr.add("machine learning", "ns")
        result = self.mgr.search("machine learning", "ns", top_k=1)[0]
        self.assertIn("id", result)
        self.assertIn("text", result)
        self.assertIn("score", result)

    def test_search_returns_correct_text(self):
        self.mgr.add("Paris is the capital of France", "cities")
        results = self.mgr.search("capital of France", "cities", top_k=1)
        self.assertEqual(results[0]["text"], "Paris is the capital of France")

    def test_namespaces_are_isolated(self):
        self.mgr.add("dogs and puppies", "animals")
        self.mgr.add("quantum physics", "science")
        animal_results = self.mgr.search("dogs", "animals", top_k=1)
        science_results = self.mgr.search("dogs", "science", top_k=1)
        # animals namespace has the relevant doc; science does not
        self.assertEqual(animal_results[0]["text"], "dogs and puppies")
        self.assertNotEqual(science_results[0]["text"], "dogs and puppies")

    def test_search_top_k_respected(self):
        for i in range(10):
            self.mgr.add(f"document {i}", "ns")
        results = self.mgr.search("document", "ns", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_search_scores_descending(self):
        texts = [
            "the cat sat on the mat",
            "dogs are great pets",
            "quantum mechanics",
        ]
        self.mgr.add_batch(texts, "ns")
        results = self.mgr.search("feline animals", "ns", top_k=3)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_invalid_query_raises(self):
        self.mgr.create_namespace("ns")
        with self.assertRaises(TypeError):
            self.mgr.search(123, "ns")  # type: ignore

    def test_search_empty_query_raises(self):
        self.mgr.create_namespace("ns")
        with self.assertRaises(ValueError):
            self.mgr.search("", "ns")


# ---------------------------------------------------------------------------
# namespace_len / __len__
# ---------------------------------------------------------------------------

class TestNamespaceManagerLen(unittest.TestCase):

    def setUp(self):
        self.mgr = NamespaceManager()

    def test_namespace_len_zero_on_empty(self):
        self.mgr.create_namespace("ns")
        self.assertEqual(self.mgr.namespace_len("ns"), 0)

    def test_namespace_len_after_adds(self):
        self.mgr.add_batch(["a", "b", "c"], "ns")
        self.assertEqual(self.mgr.namespace_len("ns"), 3)

    def test_namespace_len_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.mgr.namespace_len("ghost")

    def test_total_len_sums_all_namespaces(self):
        self.mgr.add_batch(["a", "b"], "ns1")
        self.mgr.add_batch(["x", "y", "z"], "ns2")
        self.assertEqual(len(self.mgr), 5)

    def test_total_len_zero_when_all_deleted(self):
        self.mgr.add("hello", "ns", id=0)
        self.mgr.delete(0, "ns")
        self.assertEqual(len(self.mgr), 0)


if __name__ == "__main__":
    unittest.main()
