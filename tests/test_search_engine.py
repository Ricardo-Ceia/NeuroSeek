"""
Comprehensive tests for neuroseek.SearchEngine.

No mocking — real embeddings via sentence-transformers.
The model is loaded once per session (via the conftest fixture) and shared
across all test classes through SearchEngine._from_embedder().
"""

import unittest
import pytest
from neuroseek.embedder import Embedder
from neuroseek.search_engine import SearchEngine


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestSearchEngineInit(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def test_default_model_name(self):
        e = SearchEngine._from_embedder(self._embedder)
        self.assertEqual(e.model_name, "multi-qa-MiniLM-L6-cos-v1")

    def test_custom_model_name_stored(self):
        e = SearchEngine("all-MiniLM-L6-v2")
        self.assertEqual(e.model_name, "all-MiniLM-L6-v2")

    def test_default_M(self):
        e = SearchEngine._from_embedder(self._embedder)
        self.assertEqual(e.M, 16)

    def test_default_efConstruction(self):
        e = SearchEngine._from_embedder(self._embedder)
        self.assertEqual(e.efConstruction, 200)

    def test_custom_M(self):
        e = SearchEngine._from_embedder(self._embedder, M=8)
        self.assertEqual(e.M, 8)

    def test_custom_efConstruction(self):
        e = SearchEngine._from_embedder(self._embedder, efConstruction=50)
        self.assertEqual(e.efConstruction, 50)

    def test_initial_length_is_zero(self):
        self.assertEqual(len(SearchEngine._from_embedder(self._embedder)), 0)

    def test_model_name_not_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            SearchEngine(model_name=42)  # type: ignore

    def test_model_name_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            SearchEngine(model_name="")

    def test_model_name_whitespace_raises_value_error(self):
        with self.assertRaises(ValueError):
            SearchEngine(model_name="   ")

    def test_M_not_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            SearchEngine(M=1.5)  # type: ignore

    def test_M_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            SearchEngine(M=0)

    def test_M_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            SearchEngine(M=-1)

    def test_efConstruction_not_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            SearchEngine(efConstruction=10.0)  # type: ignore

    def test_efConstruction_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            SearchEngine(efConstruction=0)

    def test_efConstruction_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            SearchEngine(efConstruction=-5)

    def test_independent_instances_do_not_share_state(self):
        a = SearchEngine._from_embedder(self._embedder)
        b = SearchEngine._from_embedder(self._embedder)
        a.add("hello world")
        self.assertEqual(len(b), 0)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestSearchEngineAdd(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.engine = SearchEngine._from_embedder(embedder)

    def test_add_returns_int(self):
        self.assertIsInstance(self.engine.add("hello"), int)

    def test_first_auto_id_is_zero(self):
        self.assertEqual(self.engine.add("hello"), 0)

    def test_second_auto_id_is_one(self):
        self.engine.add("first")
        self.assertEqual(self.engine.add("second"), 1)

    def test_explicit_id_is_honoured(self):
        self.assertEqual(self.engine.add("hello", id=42), 42)

    def test_length_increases_after_add(self):
        self.engine.add("hello")
        self.assertEqual(len(self.engine), 1)

    def test_length_correct_after_multiple_adds(self):
        for i in range(5):
            self.engine.add(f"document number {i}")
        self.assertEqual(len(self.engine), 5)

    def test_duplicate_id_raises(self):
        self.engine.add("first", id=0)
        with self.assertRaises(Exception):
            self.engine.add("second", id=0)

    def test_non_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.add(123)  # type: ignore

    def test_empty_str_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.add("")

    def test_whitespace_str_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.add("   ")

    def test_invalid_add_does_not_change_length(self):
        try:
            self.engine.add(None)  # type: ignore
        except TypeError:
            pass
        self.assertEqual(len(self.engine), 0)


# ---------------------------------------------------------------------------
# add_batch()
# ---------------------------------------------------------------------------

class TestSearchEngineAddBatch(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.engine = SearchEngine._from_embedder(embedder)

    def test_returns_list(self):
        self.assertIsInstance(self.engine.add_batch(["a", "b"]), list)

    def test_returns_correct_count(self):
        self.assertEqual(len(self.engine.add_batch(["a", "b", "c"])), 3)

    def test_each_element_is_int(self):
        for doc_id in self.engine.add_batch(["a", "b"]):
            self.assertIsInstance(doc_id, int)

    def test_auto_ids_sequential(self):
        ids = self.engine.add_batch(["a", "b", "c"])
        self.assertEqual(ids, [0, 1, 2])

    def test_length_correct_after_batch(self):
        self.engine.add_batch(["a", "b", "c"])
        self.assertEqual(len(self.engine), 3)

    def test_explicit_ids_honoured(self):
        ids = self.engine.add_batch(["a", "b"], ids=[10, 20])
        self.assertEqual(ids, [10, 20])

    def test_tuple_input_accepted(self):
        ids = self.engine.add_batch(("a", "b"))
        self.assertEqual(len(ids), 2)

    def test_non_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.add_batch("hello")  # type: ignore

    def test_empty_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.add_batch([])

    def test_non_str_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.add_batch(["valid", 42])  # type: ignore

    def test_empty_str_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.add_batch(["valid", ""])

    def test_ids_length_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.add_batch(["a", "b"], ids=[0])

    def test_ids_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.add_batch(["a", "b"], ids="bad")  # type: ignore


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestSearchEngineDelete(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.engine = SearchEngine._from_embedder(embedder)

    def test_delete_reduces_length(self):
        self.engine.add("hello", id=0)
        self.engine.delete(0)
        self.assertEqual(len(self.engine), 0)

    def test_deleted_doc_not_in_search_results(self):
        self.engine.add("machine learning algorithms", id=0)
        self.engine.add("deep neural networks", id=1)
        self.engine.delete(0)
        results = self.engine.search("machine learning", top_k=5)
        ids = [r["id"] for r in results]
        self.assertNotIn(0, ids)

    def test_delete_missing_raises(self):
        with self.assertRaises(Exception):
            self.engine.delete(99)

    def test_delete_non_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.delete("0")  # type: ignore

    def test_delete_does_not_affect_other_docs(self):
        self.engine.add("first doc", id=0)
        self.engine.add("second doc", id=1)
        self.engine.delete(0)
        results = self.engine.search("second doc", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestSearchEngineSearch(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.engine = SearchEngine._from_embedder(embedder)

    def test_empty_engine_returns_empty_list(self):
        self.assertEqual(self.engine.search("anything"), [])

    def test_returns_list(self):
        self.engine.add("hello world")
        self.assertIsInstance(self.engine.search("hello"), list)

    def test_each_result_is_dict(self):
        self.engine.add("hello world")
        for r in self.engine.search("hello"):
            self.assertIsInstance(r, dict)

    def test_result_has_id_key(self):
        self.engine.add("hello world")
        self.assertIn("id", self.engine.search("hello")[0])

    def test_result_has_text_key(self):
        self.engine.add("hello world")
        self.assertIn("text", self.engine.search("hello")[0])

    def test_result_has_score_key(self):
        self.engine.add("hello world")
        self.assertIn("score", self.engine.search("hello")[0])

    def test_result_has_metadata_key(self):
        self.engine.add("hello world")
        self.assertIn("metadata", self.engine.search("hello")[0])

    def test_result_metadata_is_dict(self):
        self.engine.add("hello world")
        self.assertIsInstance(self.engine.search("hello")[0]["metadata"], dict)

    def test_result_id_is_int(self):
        self.engine.add("hello world")
        self.assertIsInstance(self.engine.search("hello")[0]["id"], int)

    def test_result_text_is_str(self):
        self.engine.add("hello world")
        self.assertIsInstance(self.engine.search("hello")[0]["text"], str)

    def test_result_score_is_float(self):
        self.engine.add("hello world")
        self.assertIsInstance(self.engine.search("hello")[0]["score"], float)

    def test_result_text_matches_original(self):
        self.engine.add("machine learning rocks")
        results = self.engine.search("machine learning", top_k=1)
        self.assertEqual(results[0]["text"], "machine learning rocks")

    def test_score_is_between_zero_and_one(self):
        self.engine.add("natural language processing")
        for r in self.engine.search("language model"):
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_results_sorted_by_score_descending(self):
        self.engine.add("the cat sat on the mat")
        self.engine.add("dogs and puppies are cute")
        self.engine.add("quantum physics and thermodynamics")
        results = self.engine.search("feline animals", top_k=3)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limits_results(self):
        for i in range(10):
            self.engine.add(f"document about topic {i}")
        results = self.engine.search("document", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_top_k_default_is_five(self):
        for i in range(10):
            self.engine.add(f"some text number {i}")
        results = self.engine.search("some text")
        self.assertLessEqual(len(results), 5)

    def test_semantically_closest_doc_ranks_first(self):
        self.engine.add("I love dogs and puppies", id=0)
        self.engine.add("thermodynamics and entropy", id=1)
        self.engine.add("the stock market crashed today", id=2)
        results = self.engine.search("pets and animals", top_k=3)
        self.assertEqual(results[0]["id"], 0)

    def test_search_non_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.search(123)  # type: ignore

    def test_search_empty_str_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.search("")

    def test_search_whitespace_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.search("   ")

    def test_search_top_k_not_int_raises_type_error(self):
        self.engine.add("hello")
        with self.assertRaises(TypeError):
            self.engine.search("hello", top_k=1.0)  # type: ignore

    def test_search_top_k_zero_raises_value_error(self):
        self.engine.add("hello")
        with self.assertRaises(ValueError):
            self.engine.search("hello", top_k=0)

    def test_search_top_k_negative_raises_value_error(self):
        self.engine.add("hello")
        with self.assertRaises(ValueError):
            self.engine.search("hello", top_k=-1)

    def test_search_fewer_docs_than_top_k_returns_all(self):
        self.engine.add("only doc")
        results = self.engine.search("doc", top_k=10)
        self.assertEqual(len(results), 1)

    def test_exact_match_gets_high_score(self):
        text = "artificial intelligence and machine learning"
        self.engine.add(text)
        results = self.engine.search(text, top_k=1)
        self.assertGreater(results[0]["score"], 0.99)

    def test_add_then_search_round_trip(self):
        texts = [
            "Paris is the capital of France",
            "Berlin is the capital of Germany",
            "Madrid is the capital of Spain",
        ]
        self.engine.add_batch(texts)
        results = self.engine.search("capital city of France", top_k=1)
        self.assertEqual(results[0]["text"], texts[0])


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestSearchEngineLen(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.engine = SearchEngine._from_embedder(embedder)

    def test_len_zero_initially(self):
        self.assertEqual(len(self.engine), 0)

    def test_len_after_add(self):
        self.engine.add("hello")
        self.assertEqual(len(self.engine), 1)

    def test_len_after_delete(self):
        self.engine.add("hello", id=0)
        self.engine.delete(0)
        self.assertEqual(len(self.engine), 0)

    def test_len_after_batch(self):
        self.engine.add_batch(["a", "b", "c"])
        self.assertEqual(len(self.engine), 3)


# ---------------------------------------------------------------------------
# metadata — add() / add_batch()
# ---------------------------------------------------------------------------

class TestSearchEngineMetadata(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.engine = SearchEngine._from_embedder(embedder)

    def test_add_with_metadata_stores_it(self):
        self.engine.add("hello world", metadata={"source": "web"})
        results = self.engine.search("hello", top_k=1)
        self.assertEqual(results[0]["metadata"], {"source": "web"})

    def test_add_without_metadata_returns_empty_dict(self):
        self.engine.add("hello world")
        results = self.engine.search("hello", top_k=1)
        self.assertEqual(results[0]["metadata"], {})

    def test_add_metadata_none_returns_empty_dict(self):
        self.engine.add("hello world", metadata=None)
        results = self.engine.search("hello", top_k=1)
        self.assertEqual(results[0]["metadata"], {})

    def test_add_invalid_metadata_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.add("hello", metadata="bad")  # type: ignore

    def test_add_metadata_invalid_value_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.add("hello", metadata={"k": [1, 2]})  # type: ignore

    def test_add_batch_with_metadata_list(self):
        self.engine.add_batch(
            ["doc a", "doc b"],
            metadata_list=[{"tag": "x"}, {"tag": "y"}],
        )
        results = self.engine.search("doc", top_k=2)
        tags = {r["metadata"].get("tag") for r in results}
        self.assertEqual(tags, {"x", "y"})

    def test_add_batch_metadata_list_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.engine.add_batch(["a", "b"], metadata_list=[{"k": "v"}])

    def test_add_batch_invalid_metadata_element_raises(self):
        with self.assertRaises(TypeError):
            self.engine.add_batch(["a", "b"], metadata_list=[{"k": "v"}, {"bad": []}])  # type: ignore


# ---------------------------------------------------------------------------
# search() with filter
# ---------------------------------------------------------------------------

class TestSearchEngineFilter(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self.engine = SearchEngine._from_embedder(embedder)
        self.engine.add("Paris is the capital of France",
                        id=0, metadata={"country": "France", "type": "city"})
        self.engine.add("Berlin is the capital of Germany",
                        id=1, metadata={"country": "Germany", "type": "city"})
        self.engine.add("The Eiffel Tower is in Paris",
                        id=2, metadata={"country": "France", "type": "landmark"})
        self.engine.add("dogs and puppies are great pets",
                        id=3, metadata={"category": "animals"})

    def test_filter_returns_only_matching_docs(self):
        results = self.engine.search("Paris", top_k=5, filter={"country": "France"})
        for r in results:
            self.assertEqual(r["metadata"]["country"], "France")

    def test_filter_excludes_non_matching_docs(self):
        results = self.engine.search("capital city", top_k=5, filter={"country": "Germany"})
        ids = [r["id"] for r in results]
        self.assertNotIn(0, ids)
        self.assertNotIn(2, ids)
        self.assertNotIn(3, ids)

    def test_filter_multiple_keys(self):
        results = self.engine.search("Paris", top_k=5,
                                     filter={"country": "France", "type": "city"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 0)

    def test_none_filter_returns_all_candidates(self):
        results = self.engine.search("Paris", top_k=5, filter=None)
        self.assertGreater(len(results), 1)

    def test_empty_filter_returns_all_candidates(self):
        results = self.engine.search("Paris", top_k=5, filter={})
        self.assertGreater(len(results), 1)

    def test_filter_no_matches_returns_empty(self):
        results = self.engine.search("Paris", top_k=5,
                                     filter={"country": "Japan"})
        self.assertEqual(results, [])

    def test_filter_respects_top_k(self):
        results = self.engine.search("France", top_k=1,
                                     filter={"country": "France"})
        self.assertLessEqual(len(results), 1)

    def test_filter_invalid_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.search("hello", filter="bad")  # type: ignore

    def test_filter_invalid_value_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.search("hello", filter={"k": [1, 2]})  # type: ignore

    def test_filtered_results_sorted_by_score_descending(self):
        results = self.engine.search("France", top_k=5,
                                     filter={"country": "France"})
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestSearchEngineListSources(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._engine = SearchEngine._from_embedder(embedder)

    def test_empty_engine_returns_empty_set(self):
        self.assertEqual(self._engine.list_sources(), set())

    def test_single_doc_with_filename(self):
        self._engine.add("hello world", metadata={"filename": "a.txt"})
        self.assertEqual(self._engine.list_sources(), {"a.txt"})

    def test_multiple_chunks_same_file_returns_one_entry(self):
        self._engine.add("chunk 1", metadata={"filename": "a.txt"})
        self._engine.add("chunk 2", metadata={"filename": "a.txt"})
        self.assertEqual(self._engine.list_sources(), {"a.txt"})

    def test_multiple_files_returns_all(self):
        self._engine.add("text a", metadata={"filename": "a.txt"})
        self._engine.add("text b", metadata={"filename": "b.md"})
        self.assertEqual(self._engine.list_sources(), {"a.txt", "b.md"})

    def test_doc_without_filename_ignored(self):
        self._engine.add("no filename", metadata={"other": "x"})
        self._engine.add("with filename", metadata={"filename": "a.txt"})
        self.assertEqual(self._engine.list_sources(), {"a.txt"})

    def test_custom_key(self):
        self._engine.add("text", metadata={"path": "/some/file.md"})
        self.assertEqual(self._engine.list_sources(key="path"), {"/some/file.md"})

    def test_returns_set_type(self):
        self._engine.add("text", metadata={"filename": "a.txt"})
        self.assertIsInstance(self._engine.list_sources(), set)

    def test_invalid_key_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._engine.list_sources(key=99)  # type: ignore

    def test_empty_key_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._engine.list_sources(key="")


# ---------------------------------------------------------------------------
# delete_by_source()
# ---------------------------------------------------------------------------

class TestSearchEngineDeleteBySource(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject(self, embedder: Embedder) -> None:
        self._engine = SearchEngine._from_embedder(embedder)

    def test_delete_by_source_returns_correct_count(self):
        self._engine.add("chunk one", id=0, metadata={"filename": "a.txt"})
        self._engine.add("chunk two", id=1, metadata={"filename": "a.txt"})
        count = self._engine.delete_by_source("a.txt")
        self.assertEqual(count, 2)

    def test_delete_by_source_reduces_length(self):
        self._engine.add("chunk one", id=0, metadata={"filename": "a.txt"})
        self._engine.add("chunk two", id=1, metadata={"filename": "a.txt"})
        self._engine.delete_by_source("a.txt")
        self.assertEqual(len(self._engine), 0)

    def test_delete_by_source_removed_from_search_results(self):
        self._engine.add("machine learning basics", id=0, metadata={"filename": "ml.txt"})
        self._engine.add("deep neural networks", id=1, metadata={"filename": "nn.txt"})
        self._engine.delete_by_source("ml.txt")
        results = self._engine.search("machine learning", top_k=5)
        ids = [r["id"] for r in results]
        self.assertNotIn(0, ids)

    def test_delete_by_source_does_not_affect_other_files(self):
        self._engine.add("content from file a", id=0, metadata={"filename": "a.txt"})
        self._engine.add("content from file b", id=1, metadata={"filename": "b.txt"})
        self._engine.delete_by_source("a.txt")
        self.assertEqual(len(self._engine), 1)
        results = self._engine.search("content from file b", top_k=1)
        self.assertEqual(results[0]["id"], 1)

    def test_delete_by_source_unknown_filename_returns_zero(self):
        self._engine.add("some content", id=0, metadata={"filename": "real.txt"})
        count = self._engine.delete_by_source("ghost.txt")
        self.assertEqual(count, 0)

    def test_delete_by_source_unknown_filename_does_not_change_length(self):
        self._engine.add("some content", id=0, metadata={"filename": "real.txt"})
        self._engine.delete_by_source("ghost.txt")
        self.assertEqual(len(self._engine), 1)

    def test_delete_by_source_multiple_chunks_all_removed(self):
        for i in range(5):
            self._engine.add(f"chunk {i}", id=i, metadata={"filename": "big.txt"})
        self._engine.delete_by_source("big.txt")
        self.assertEqual(len(self._engine), 0)

    def test_delete_by_source_only_affects_matching_chunks(self):
        self._engine.add("alpha", id=0, metadata={"filename": "a.txt"})
        self._engine.add("beta", id=1, metadata={"filename": "b.txt"})
        self._engine.add("gamma", id=2, metadata={"filename": "a.txt"})
        self._engine.delete_by_source("a.txt")
        self.assertEqual(len(self._engine), 1)

    def test_delete_by_source_non_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._engine.delete_by_source(123)  # type: ignore

    def test_delete_by_source_empty_str_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._engine.delete_by_source("")

    def test_delete_by_source_whitespace_str_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._engine.delete_by_source("   ")

    def test_delete_by_source_idempotent(self):
        self._engine.add("content", id=0, metadata={"filename": "a.txt"})
        self._engine.delete_by_source("a.txt")
        # Second call on already-deleted file should return 0, not raise
        count = self._engine.delete_by_source("a.txt")
        self.assertEqual(count, 0)

    def test_delete_by_source_docs_without_filename_not_affected(self):
        self._engine.add("no filename here", id=0, metadata={"other": "x"})
        self._engine.delete_by_source("a.txt")
        self.assertEqual(len(self._engine), 1)


if __name__ == "__main__":
    unittest.main()
