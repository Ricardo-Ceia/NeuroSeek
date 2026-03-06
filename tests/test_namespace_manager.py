"""
Comprehensive tests for neuroseek.NamespaceManager.

No mocking — real embeddings, real search.
The model is loaded once per session via the conftest fixture and injected
into every test class via NamespaceManager._from_embedder().
"""

import unittest
import pytest

from neuroseek.embedder import Embedder
from neuroseek.namespace_manager import NamespaceManager


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestNamespaceManagerInit(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def test_default_model_name(self):
        mgr = NamespaceManager._from_embedder(self._embedder)
        self.assertEqual(mgr.model_name, "multi-qa-MiniLM-L6-cos-v1")

    def test_default_M(self):
        self.assertEqual(NamespaceManager._from_embedder(self._embedder).M, 16)

    def test_default_efConstruction(self):
        self.assertEqual(NamespaceManager._from_embedder(self._embedder).efConstruction, 200)

    def test_custom_params_stored(self):
        mgr = NamespaceManager._from_embedder(self._embedder, M=8, efConstruction=50)
        self.assertEqual(mgr.M, 8)
        self.assertEqual(mgr.efConstruction, 50)

    def test_initial_len_is_zero(self):
        self.assertEqual(len(NamespaceManager._from_embedder(self._embedder)), 0)

    def test_initial_namespaces_empty(self):
        self.assertEqual(NamespaceManager._from_embedder(self._embedder).list_namespaces(), [])

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

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.mgr = NamespaceManager._from_embedder(embedder)

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

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.mgr = NamespaceManager._from_embedder(embedder)

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

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.mgr = NamespaceManager._from_embedder(embedder)

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

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.mgr = NamespaceManager._from_embedder(embedder)

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
        self.assertIn("metadata", result)

    def test_search_returns_correct_text(self):
        self.mgr.add("Paris is the capital of France", "cities")
        results = self.mgr.search("capital of France", "cities", top_k=1)
        self.assertEqual(results[0]["text"], "Paris is the capital of France")

    def test_namespaces_are_isolated(self):
        self.mgr.add("dogs and puppies", "animals")
        self.mgr.add("quantum physics", "science")
        animal_results = self.mgr.search("dogs", "animals", top_k=1)
        science_results = self.mgr.search("dogs", "science", top_k=1)
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

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.mgr = NamespaceManager._from_embedder(embedder)

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


# ---------------------------------------------------------------------------
# Metadata — add() / add_batch()
# ---------------------------------------------------------------------------

class TestNamespaceManagerAddMetadata(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.mgr = NamespaceManager._from_embedder(embedder)

    def test_add_with_metadata_stores_it(self):
        self.mgr.add("hello", "ns", id=0, metadata={"lang": "en"})
        result = self.mgr.search("hello", "ns", top_k=1)[0]
        self.assertEqual(result["metadata"], {"lang": "en"})

    def test_add_without_metadata_returns_empty_dict(self):
        self.mgr.add("hello", "ns", id=0)
        result = self.mgr.search("hello", "ns", top_k=1)[0]
        self.assertEqual(result["metadata"], {})

    def test_add_metadata_none_returns_empty_dict(self):
        self.mgr.add("hello", "ns", id=0, metadata=None)
        result = self.mgr.search("hello", "ns", top_k=1)[0]
        self.assertEqual(result["metadata"], {})

    def test_add_metadata_invalid_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.mgr.add("hello", "ns", metadata="bad")  # type: ignore

    def test_add_metadata_non_str_key_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.mgr.add("hello", "ns", metadata={1: "v"})  # type: ignore

    def test_add_metadata_invalid_value_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.mgr.add("hello", "ns", metadata={"k": [1, 2]})  # type: ignore

    def test_add_batch_with_metadata_list(self):
        ids = self.mgr.add_batch(
            ["cat", "dog"],
            "ns",
            ids=[0, 1],
            metadata_list=[{"animal": "cat"}, {"animal": "dog"}],
        )
        self.assertEqual(ids, [0, 1])
        results = {r["id"]: r["metadata"] for r in self.mgr.search("animal", "ns", top_k=2)}
        self.assertEqual(results[0], {"animal": "cat"})
        self.assertEqual(results[1], {"animal": "dog"})

    def test_add_batch_metadata_list_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self.mgr.add_batch(["a", "b", "c"], "ns", metadata_list=[{"k": "v"}])

    def test_add_batch_without_metadata_list_returns_empty_dicts(self):
        self.mgr.add_batch(["a", "b"], "ns", ids=[0, 1])
        results = {r["id"]: r["metadata"] for r in self.mgr.search("a", "ns", top_k=2)}
        self.assertEqual(results[0], {})
        self.assertEqual(results[1], {})

    def test_add_metadata_multiple_fields(self):
        self.mgr.add("hello", "ns", id=0, metadata={"lang": "en", "version": 2, "active": True})
        result = self.mgr.search("hello", "ns", top_k=1)[0]
        self.assertEqual(result["metadata"]["lang"], "en")
        self.assertEqual(result["metadata"]["version"], 2)
        self.assertEqual(result["metadata"]["active"], True)


# ---------------------------------------------------------------------------
# Metadata — search() with filter
# ---------------------------------------------------------------------------

class TestNamespaceManagerSearchFilter(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.mgr = NamespaceManager._from_embedder(embedder)
        self.mgr.add_batch(
            [
                "the cat sat on the mat",
                "dogs are wonderful pets",
                "python is a programming language",
                "cats love to sleep",
                "dogs fetch sticks",
            ],
            "ns",
            ids=[0, 1, 2, 3, 4],
            metadata_list=[
                {"type": "animal", "subject": "cat"},
                {"type": "animal", "subject": "dog"},
                {"type": "tech", "subject": "python"},
                {"type": "animal", "subject": "cat"},
                {"type": "animal", "subject": "dog"},
            ],
        )

    def test_filter_single_field_reduces_results(self):
        results = self.mgr.search("animals", "ns", top_k=5, filter={"type": "tech"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 2)

    def test_filter_returns_only_matching_docs(self):
        results = self.mgr.search("pets", "ns", top_k=5, filter={"subject": "cat"})
        ids = {r["id"] for r in results}
        self.assertEqual(ids, {0, 3})

    def test_filter_multi_field(self):
        results = self.mgr.search("animals", "ns", top_k=5,
                                  filter={"type": "animal", "subject": "dog"})
        ids = {r["id"] for r in results}
        self.assertEqual(ids, {1, 4})

    def test_filter_no_match_returns_empty(self):
        results = self.mgr.search("anything", "ns", top_k=5, filter={"type": "nonexistent"})
        self.assertEqual(results, [])

    def test_filter_none_returns_all(self):
        results = self.mgr.search("animals", "ns", top_k=5, filter=None)
        self.assertEqual(len(results), 5)

    def test_filter_invalid_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.mgr.search("query", "ns", filter="bad")  # type: ignore

    def test_filter_non_str_key_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.mgr.search("query", "ns", filter={1: "v"})  # type: ignore

    def test_filter_result_includes_metadata_key(self):
        results = self.mgr.search("cat", "ns", top_k=1, filter={"subject": "cat"})
        self.assertIn("metadata", results[0])

    def test_filter_respects_namespace_isolation(self):
        self.mgr.add("cats meow", "other", id=0, metadata={"subject": "cat"})
        results_ns = self.mgr.search("cat", "ns", top_k=5, filter={"subject": "cat"})
        ids_ns = {r["id"] for r in results_ns}
        self.assertEqual(ids_ns, {0, 3})

    def test_filter_scores_still_descending(self):
        results = self.mgr.search("cats", "ns", top_k=5, filter={"type": "animal"})
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestNamespaceManagerListSources(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._mgr = NamespaceManager._from_embedder(embedder)

    def test_empty_namespace_returns_empty_set(self):
        self._mgr.create_namespace("ns")
        self.assertEqual(self._mgr.list_sources("ns"), set())

    def test_single_doc_returns_its_filename(self):
        self._mgr.add("hello", namespace="ns", metadata={"filename": "a.txt"})
        self.assertEqual(self._mgr.list_sources("ns"), {"a.txt"})

    def test_multiple_chunks_same_file_returns_one_entry(self):
        self._mgr.add("chunk 1", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.add("chunk 2", namespace="ns", metadata={"filename": "a.txt"})
        self.assertEqual(self._mgr.list_sources("ns"), {"a.txt"})

    def test_multiple_files_returns_all(self):
        self._mgr.add("text a", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.add("text b", namespace="ns", metadata={"filename": "b.md"})
        self.assertEqual(self._mgr.list_sources("ns"), {"a.txt", "b.md"})

    def test_sources_are_namespace_isolated(self):
        self._mgr.add("text", namespace="ns1", metadata={"filename": "a.txt"})
        self._mgr.add("text", namespace="ns2", metadata={"filename": "b.txt"})
        self.assertEqual(self._mgr.list_sources("ns1"), {"a.txt"})
        self.assertEqual(self._mgr.list_sources("ns2"), {"b.txt"})

    def test_nonexistent_namespace_raises_key_error(self):
        with self.assertRaises(KeyError):
            self._mgr.list_sources("does_not_exist")

    def test_custom_key(self):
        self._mgr.add("text", namespace="ns", metadata={"path": "/docs/readme.md"})
        self.assertEqual(self._mgr.list_sources("ns", key="path"), {"/docs/readme.md"})

    def test_returns_set_type(self):
        self._mgr.add("text", namespace="ns", metadata={"filename": "a.txt"})
        self.assertIsInstance(self._mgr.list_sources("ns"), set)

    def test_docs_without_filename_are_ignored(self):
        self._mgr.add("no filename", namespace="ns", metadata={"other": "x"})
        self._mgr.add("with filename", namespace="ns", metadata={"filename": "f.txt"})
        self.assertEqual(self._mgr.list_sources("ns"), {"f.txt"})


# ---------------------------------------------------------------------------
# delete_source()
# ---------------------------------------------------------------------------

class TestNamespaceManagerDeleteSource(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._mgr = NamespaceManager._from_embedder(embedder)

    def test_delete_source_returns_correct_count(self):
        self._mgr.add("chunk 1", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.add("chunk 2", namespace="ns", metadata={"filename": "a.txt"})
        count = self._mgr.delete_source("a.txt", "ns")
        self.assertEqual(count, 2)

    def test_delete_source_reduces_namespace_len(self):
        self._mgr.add("chunk 1", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.add("chunk 2", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.delete_source("a.txt", "ns")
        self.assertEqual(self._mgr.namespace_len("ns"), 0)

    def test_delete_source_reduces_total_len(self):
        self._mgr.add("chunk 1", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.delete_source("a.txt", "ns")
        self.assertEqual(len(self._mgr), 0)

    def test_delete_source_removed_from_search(self):
        self._mgr.add("machine learning models", namespace="ns", metadata={"filename": "ml.txt"})
        self._mgr.add("deep neural networks", namespace="ns", metadata={"filename": "nn.txt"})
        self._mgr.delete_source("ml.txt", "ns")
        results = self._mgr.search("machine learning", "ns", top_k=5)
        ids = [r["id"] for r in results]
        self.assertNotIn(0, ids)

    def test_delete_source_does_not_affect_other_files(self):
        self._mgr.add("content from a", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.add("content from b", namespace="ns", metadata={"filename": "b.txt"})
        self._mgr.delete_source("a.txt", "ns")
        self.assertEqual(self._mgr.namespace_len("ns"), 1)

    def test_delete_source_namespace_isolated(self):
        self._mgr.add("content", namespace="ns1", metadata={"filename": "a.txt"})
        self._mgr.add("content", namespace="ns2", metadata={"filename": "a.txt"})
        self._mgr.delete_source("a.txt", "ns1")
        self.assertEqual(self._mgr.namespace_len("ns1"), 0)
        self.assertEqual(self._mgr.namespace_len("ns2"), 1)

    def test_delete_source_unknown_filename_returns_zero(self):
        self._mgr.create_namespace("ns")
        count = self._mgr.delete_source("ghost.txt", "ns")
        self.assertEqual(count, 0)

    def test_delete_source_missing_namespace_raises_key_error(self):
        with self.assertRaises(KeyError):
            self._mgr.delete_source("a.txt", "does_not_exist")

    def test_delete_source_non_str_filename_raises_type_error(self):
        self._mgr.create_namespace("ns")
        with self.assertRaises(TypeError):
            self._mgr.delete_source(42, "ns")  # type: ignore

    def test_delete_source_empty_filename_raises_value_error(self):
        self._mgr.create_namespace("ns")
        with self.assertRaises(ValueError):
            self._mgr.delete_source("", "ns")

    def test_delete_source_multiple_chunks_all_removed(self):
        for i in range(4):
            self._mgr.add(f"chunk {i}", namespace="ns", metadata={"filename": "big.txt"})
        self._mgr.delete_source("big.txt", "ns")
        self.assertEqual(self._mgr.namespace_len("ns"), 0)

    def test_delete_source_idempotent(self):
        self._mgr.add("content", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.delete_source("a.txt", "ns")
        count = self._mgr.delete_source("a.txt", "ns")
        self.assertEqual(count, 0)

    def test_delete_source_updates_list_sources(self):
        self._mgr.add("content", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.delete_source("a.txt", "ns")
        self.assertNotIn("a.txt", self._mgr.list_sources("ns"))


# ---------------------------------------------------------------------------
# delete_by_query()
# ---------------------------------------------------------------------------

class TestNamespaceManagerDeleteByQuery(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._mgr = NamespaceManager._from_embedder(embedder)

    def test_delete_by_query_returns_list(self):
        self._mgr.add("dogs are great pets", namespace="ns")
        result = self._mgr.delete_by_query("dogs", "ns")
        self.assertIsInstance(result, list)

    def test_delete_by_query_returns_result_dicts(self):
        self._mgr.add("cats are curious animals", namespace="ns")
        results = self._mgr.delete_by_query("cats", "ns", top_k=1)
        for r in results:
            self.assertIn("id", r)
            self.assertIn("text", r)
            self.assertIn("score", r)
            self.assertIn("metadata", r)

    def test_delete_by_query_removes_from_namespace(self):
        self._mgr.add("dogs and puppies are great", namespace="ns", id=0)
        self._mgr.add("quantum physics", namespace="ns", id=1)
        self._mgr.delete_by_query("dogs and puppies", "ns", top_k=1)
        self.assertEqual(self._mgr.namespace_len("ns"), 1)

    def test_delete_by_query_is_namespace_isolated(self):
        self._mgr.add("dogs and puppies are great", namespace="ns1", id=0)
        self._mgr.add("dogs and puppies are great", namespace="ns2", id=0)
        self._mgr.delete_by_query("dogs and puppies", "ns1", top_k=1)
        self.assertEqual(self._mgr.namespace_len("ns1"), 0)
        self.assertEqual(self._mgr.namespace_len("ns2"), 1)

    def test_delete_by_query_top_k_limits_deletions(self):
        for i in range(6):
            self._mgr.add(f"document about dogs number {i}", namespace="ns", id=i)
        self._mgr.delete_by_query("dogs", "ns", top_k=3)
        self.assertEqual(self._mgr.namespace_len("ns"), 3)

    def test_delete_by_query_empty_engine_returns_empty_list(self):
        self._mgr.create_namespace("ns")
        result = self._mgr.delete_by_query("anything", "ns")
        self.assertEqual(result, [])

    def test_delete_by_query_missing_namespace_raises_key_error(self):
        with self.assertRaises(KeyError):
            self._mgr.delete_by_query("query", "does_not_exist")

    def test_delete_by_query_invalid_query_raises_type_error(self):
        self._mgr.create_namespace("ns")
        with self.assertRaises(TypeError):
            self._mgr.delete_by_query(123, "ns")  # type: ignore

    def test_delete_by_query_empty_query_raises_value_error(self):
        self._mgr.create_namespace("ns")
        with self.assertRaises(ValueError):
            self._mgr.delete_by_query("", "ns")

    def test_delete_by_query_with_filter(self):
        self._mgr.add("dogs bark loudly", namespace="ns", id=0, metadata={"type": "animal"})
        self._mgr.add("dogs in cartoons", namespace="ns", id=1, metadata={"type": "fiction"})
        deleted = self._mgr.delete_by_query("dogs", "ns", top_k=5, filter={"type": "animal"})
        deleted_ids = {r["id"] for r in deleted}
        self.assertIn(0, deleted_ids)
        self.assertNotIn(1, deleted_ids)

    def test_delete_by_query_reduces_total_len(self):
        self._mgr.add("dogs and puppies are great", namespace="ns", id=0)
        self._mgr.delete_by_query("dogs", "ns", top_k=1)
        self.assertEqual(len(self._mgr), 0)


# ---------------------------------------------------------------------------
# update_source()
# ---------------------------------------------------------------------------

class TestNamespaceManagerUpdateSource(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._mgr = NamespaceManager._from_embedder(embedder)

    def test_update_source_returns_new_chunk_count(self):
        self._mgr.add("old content", namespace="ns", metadata={"filename": "a.txt"})
        count = self._mgr.update_source("a.txt", "ns", ["new chunk one", "new chunk two"])
        self.assertEqual(count, 2)

    def test_update_source_removes_old_chunks(self):
        self._mgr.add("old content about dogs", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.update_source("a.txt", "ns", ["new content about cats"])
        results = self._mgr.search("dogs", "ns", top_k=5)
        texts = [r["text"] for r in results]
        self.assertNotIn("old content about dogs", texts)

    def test_update_source_indexes_new_chunks(self):
        self._mgr.add("old content", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.update_source("a.txt", "ns", ["brand new machine learning content"])
        results = self._mgr.search("machine learning", "ns", top_k=1)
        self.assertEqual(results[0]["text"], "brand new machine learning content")

    def test_update_source_is_namespace_isolated(self):
        self._mgr.add("old content about dogs", namespace="ns1", metadata={"filename": "a.txt"})
        self._mgr.add("old content about dogs", namespace="ns2", metadata={"filename": "a.txt"})
        self._mgr.update_source("a.txt", "ns1", ["new content about cats"])
        # ns2 should be unchanged
        results = self._mgr.search("dogs", "ns2", top_k=1)
        self.assertEqual(results[0]["text"], "old content about dogs")

    def test_update_source_correct_length_after_update(self):
        self._mgr.add("old one", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.add("old two", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.update_source("a.txt", "ns", ["single new chunk"])
        self.assertEqual(self._mgr.namespace_len("ns"), 1)

    def test_update_source_missing_namespace_raises_key_error(self):
        with self.assertRaises(KeyError):
            self._mgr.update_source("a.txt", "ghost", ["chunk"])

    def test_update_source_non_str_filename_raises_type_error(self):
        self._mgr.create_namespace("ns")
        with self.assertRaises(TypeError):
            self._mgr.update_source(42, "ns", ["chunk"])  # type: ignore

    def test_update_source_empty_filename_raises_value_error(self):
        self._mgr.create_namespace("ns")
        with self.assertRaises(ValueError):
            self._mgr.update_source("", "ns", ["chunk"])

    def test_update_source_empty_chunks_raises_value_error(self):
        self._mgr.create_namespace("ns")
        with self.assertRaises(ValueError):
            self._mgr.update_source("a.txt", "ns", [])

    def test_update_source_preserves_filename_in_metadata(self):
        self._mgr.add("old", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.update_source("a.txt", "ns", ["updated content here"])
        results = self._mgr.search("updated content", "ns", top_k=1)
        self.assertEqual(results[0]["metadata"]["filename"], "a.txt")

    def test_update_source_updates_list_sources(self):
        self._mgr.add("old", namespace="ns", metadata={"filename": "a.txt"})
        self._mgr.update_source("a.txt", "ns", ["new content"])
        self.assertIn("a.txt", self._mgr.list_sources("ns"))


if __name__ == "__main__":
    unittest.main()
