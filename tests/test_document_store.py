"""
Comprehensive tests for neuroseek.DocumentStore.

No mocking — all tests exercise real behavior.
"""

import unittest
from neuroseek.document_store import DocumentStore


class TestDocumentStoreInit(unittest.TestCase):

    def test_initial_length_is_zero(self):
        self.assertEqual(len(DocumentStore()), 0)

    def test_independent_instances(self):
        a = DocumentStore()
        b = DocumentStore()
        a.add("hello")
        self.assertEqual(len(b), 0)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestDocumentStoreAdd(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    # --- Return value ---

    def test_add_returns_int(self):
        self.assertIsInstance(self.store.add("hello"), int)

    def test_first_auto_id_is_zero(self):
        self.assertEqual(self.store.add("hello"), 0)

    def test_second_auto_id_is_one(self):
        self.store.add("first")
        self.assertEqual(self.store.add("second"), 1)

    def test_auto_ids_are_monotonically_increasing(self):
        ids = [self.store.add(f"doc {i}") for i in range(10)]
        self.assertEqual(ids, list(range(10)))

    # --- Explicit id ---

    def test_explicit_id_is_honoured(self):
        self.assertEqual(self.store.add("explicit", id=42), 42)

    def test_explicit_id_document_retrievable(self):
        self.store.add("explicit", id=42)
        self.assertEqual(self.store.get(42), "explicit")

    def test_explicit_id_does_not_break_auto_counter(self):
        self.store.add("a", id=5)
        next_id = self.store.add("b")       # auto
        self.assertGreater(next_id, 5)

    def test_upsert_overwrites_existing(self):
        self.store.add("original", id=0)
        self.store.add("updated", id=0)
        self.assertEqual(self.store.get(0), "updated")

    def test_upsert_does_not_increase_length(self):
        self.store.add("original", id=0)
        self.store.add("updated", id=0)
        self.assertEqual(len(self.store), 1)

    def test_explicit_id_zero_accepted(self):
        self.assertEqual(self.store.add("zero", id=0), 0)

    # --- Length ---

    def test_length_increases_after_add(self):
        self.store.add("hello")
        self.assertEqual(len(self.store), 1)

    def test_length_correct_after_multiple_adds(self):
        for i in range(5):
            self.store.add(f"doc {i}")
        self.assertEqual(len(self.store), 5)

    # --- Validation errors ---

    def test_non_str_text_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add(123)  # type: ignore

    def test_none_text_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add(None)  # type: ignore

    def test_empty_text_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add("")

    def test_whitespace_text_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add("   ")

    def test_tab_newline_text_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add("\t\n")

    def test_non_int_id_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add("hello", id="abc")  # type: ignore

    def test_float_id_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add("hello", id=1.0)  # type: ignore

    def test_negative_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add("hello", id=-1)

    def test_invalid_input_does_not_change_length(self):
        try:
            self.store.add(None)  # type: ignore
        except TypeError:
            pass
        self.assertEqual(len(self.store), 0)


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestDocumentStoreGet(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    def test_get_returns_correct_text(self):
        self.store.add("hello world", id=0)
        self.assertEqual(self.store.get(0), "hello world")

    def test_get_returns_str(self):
        self.store.add("text", id=0)
        self.assertIsInstance(self.store.get(0), str)

    def test_get_after_upsert_returns_new_text(self):
        self.store.add("old", id=0)
        self.store.add("new", id=0)
        self.assertEqual(self.store.get(0), "new")

    def test_get_missing_id_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.store.get(99)

    def test_get_non_int_id_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.get("0")  # type: ignore

    def test_get_negative_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.get(-1)

    def test_get_multiple_documents_independent(self):
        self.store.add("first", id=0)
        self.store.add("second", id=1)
        self.assertEqual(self.store.get(0), "first")
        self.assertEqual(self.store.get(1), "second")


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDocumentStoreDelete(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    def test_delete_reduces_length(self):
        self.store.add("hello", id=0)
        self.store.delete(0)
        self.assertEqual(len(self.store), 0)

    def test_delete_makes_get_raise_key_error(self):
        self.store.add("hello", id=0)
        self.store.delete(0)
        with self.assertRaises(KeyError):
            self.store.get(0)

    def test_delete_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.store.delete(99)

    def test_delete_non_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.delete("0")  # type: ignore

    def test_delete_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.delete(-1)

    def test_delete_does_not_affect_other_docs(self):
        self.store.add("a", id=0)
        self.store.add("b", id=1)
        self.store.delete(0)
        self.assertEqual(self.store.get(1), "b")

    def test_auto_id_not_reused_after_delete(self):
        self.store.add("a")   # id=0
        self.store.delete(0)
        new_id = self.store.add("b")
        self.assertNotEqual(new_id, 0)

    def test_double_delete_raises_key_error(self):
        self.store.add("hello", id=0)
        self.store.delete(0)
        with self.assertRaises(KeyError):
            self.store.delete(0)


# ---------------------------------------------------------------------------
# add_batch()
# ---------------------------------------------------------------------------

class TestDocumentStoreAddBatch(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    # --- Return value ---

    def test_returns_list(self):
        self.assertIsInstance(self.store.add_batch(["a", "b"]), list)

    def test_returns_correct_count(self):
        self.assertEqual(len(self.store.add_batch(["a", "b", "c"])), 3)

    def test_each_element_is_int(self):
        for doc_id in self.store.add_batch(["a", "b"]):
            self.assertIsInstance(doc_id, int)

    def test_auto_ids_sequential(self):
        ids = self.store.add_batch(["a", "b", "c"])
        self.assertEqual(ids, [0, 1, 2])

    def test_length_correct_after_batch(self):
        self.store.add_batch(["a", "b", "c"])
        self.assertEqual(len(self.store), 3)

    def test_documents_retrievable_after_batch(self):
        ids = self.store.add_batch(["hello", "world"])
        self.assertEqual(self.store.get(ids[0]), "hello")
        self.assertEqual(self.store.get(ids[1]), "world")

    # --- Explicit ids ---

    def test_explicit_ids_honoured(self):
        ids = self.store.add_batch(["a", "b"], ids=[10, 20])
        self.assertEqual(ids, [10, 20])

    def test_explicit_ids_docs_retrievable(self):
        self.store.add_batch(["alpha", "beta"], ids=[10, 20])
        self.assertEqual(self.store.get(10), "alpha")
        self.assertEqual(self.store.get(20), "beta")

    def test_tuple_input_accepted(self):
        ids = self.store.add_batch(("a", "b"))
        self.assertEqual(len(ids), 2)

    # --- Single item ---

    def test_single_item_batch(self):
        ids = self.store.add_batch(["only"])
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.store.get(ids[0]), "only")

    # --- Atomicity: invalid input leaves store unchanged ---

    def test_invalid_text_element_leaves_store_unchanged(self):
        self.store.add("pre-existing", id=99)
        try:
            self.store.add_batch(["valid", 42])  # type: ignore
        except TypeError:
            pass
        self.assertEqual(len(self.store), 1)

    def test_empty_string_element_leaves_store_unchanged(self):
        try:
            self.store.add_batch(["ok", ""])
        except ValueError:
            pass
        self.assertEqual(len(self.store), 0)

    # --- Input type validation ---

    def test_string_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add_batch("hello")  # type: ignore

    def test_dict_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add_batch({"a": 1})  # type: ignore

    def test_empty_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add_batch([])

    def test_non_str_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add_batch(["valid", 42])  # type: ignore

    def test_none_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add_batch(["valid", None])  # type: ignore

    def test_empty_string_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add_batch(["valid", ""])

    def test_whitespace_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add_batch(["valid", "   "])

    def test_error_reports_offending_index(self):
        try:
            self.store.add_batch(["ok", "ok", 99])  # type: ignore
        except TypeError as exc:
            self.assertIn("2", str(exc))
        else:
            self.fail("Expected TypeError")

    def test_ids_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add_batch(["a", "b"], ids="bad")  # type: ignore

    def test_ids_length_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add_batch(["a", "b"], ids=[0])

    def test_negative_explicit_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.add_batch(["a", "b"], ids=[0, -1])

    # --- Interaction with add() ---

    def test_batch_after_single_add_continues_counter(self):
        self.store.add("first")          # id=0
        ids = self.store.add_batch(["second", "third"])
        self.assertEqual(ids, [1, 2])


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------

class TestDocumentStoreContains(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    def test_contains_true_after_add(self):
        self.store.add("hello", id=5)
        self.assertIn(5, self.store)

    def test_contains_false_for_missing(self):
        self.assertNotIn(99, self.store)

    def test_contains_false_after_delete(self):
        self.store.add("hello", id=5)
        self.store.delete(5)
        self.assertNotIn(5, self.store)


# ---------------------------------------------------------------------------
# metadata — add() / get_metadata() / matches_filter()
# ---------------------------------------------------------------------------

class TestDocumentStoreMetadata(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    # --- add with metadata ---

    def test_add_with_metadata_returns_id(self):
        self.assertIsInstance(self.store.add("hello", metadata={"tag": "a"}), int)

    def test_metadata_retrievable(self):
        self.store.add("hello", id=0, metadata={"source": "web", "year": 2024})
        self.assertEqual(self.store.get_metadata(0), {"source": "web", "year": 2024})

    def test_no_metadata_returns_empty_dict(self):
        self.store.add("hello", id=0)
        self.assertEqual(self.store.get_metadata(0), {})

    def test_none_metadata_treated_as_empty(self):
        self.store.add("hello", id=0, metadata=None)
        self.assertEqual(self.store.get_metadata(0), {})

    def test_metadata_is_a_copy(self):
        meta = {"key": "value"}
        self.store.add("hello", id=0, metadata=meta)
        meta["key"] = "mutated"
        self.assertEqual(self.store.get_metadata(0)["key"], "value")

    def test_get_metadata_missing_id_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.store.get_metadata(99)

    def test_get_metadata_non_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.get_metadata("0")  # type: ignore

    def test_metadata_str_value_accepted(self):
        self.store.add("hello", id=0, metadata={"k": "v"})
        self.assertEqual(self.store.get_metadata(0)["k"], "v")

    def test_metadata_int_value_accepted(self):
        self.store.add("hello", id=0, metadata={"n": 42})
        self.assertEqual(self.store.get_metadata(0)["n"], 42)

    def test_metadata_float_value_accepted(self):
        self.store.add("hello", id=0, metadata={"score": 0.9})
        self.assertAlmostEqual(self.store.get_metadata(0)["score"], 0.9)

    def test_metadata_bool_value_accepted(self):
        self.store.add("hello", id=0, metadata={"active": True})
        self.assertEqual(self.store.get_metadata(0)["active"], True)

    def test_metadata_non_dict_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add("hello", metadata="bad")  # type: ignore

    def test_metadata_non_str_key_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add("hello", metadata={1: "v"})  # type: ignore

    def test_metadata_invalid_value_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.add("hello", metadata={"k": [1, 2]})  # type: ignore

    def test_upsert_replaces_metadata(self):
        self.store.add("hello", id=0, metadata={"old": "yes"})
        self.store.add("hello", id=0, metadata={"new": "yes"})
        self.assertNotIn("old", self.store.get_metadata(0))
        self.assertIn("new", self.store.get_metadata(0))

    # --- matches_filter ---

    def test_none_filter_matches_everything(self):
        self.store.add("hello", id=0, metadata={"tag": "a"})
        self.assertTrue(self.store.matches_filter(0, None))

    def test_empty_filter_matches_everything(self):
        self.store.add("hello", id=0, metadata={"tag": "a"})
        self.assertTrue(self.store.matches_filter(0, {}))

    def test_matching_filter_returns_true(self):
        self.store.add("hello", id=0, metadata={"tag": "news", "year": 2024})
        self.assertTrue(self.store.matches_filter(0, {"tag": "news"}))

    def test_non_matching_filter_returns_false(self):
        self.store.add("hello", id=0, metadata={"tag": "news"})
        self.assertFalse(self.store.matches_filter(0, {"tag": "sports"}))

    def test_partial_filter_all_match(self):
        self.store.add("hello", id=0, metadata={"tag": "news", "year": 2024})
        self.assertTrue(self.store.matches_filter(0, {"tag": "news", "year": 2024}))

    def test_partial_filter_one_mismatch(self):
        self.store.add("hello", id=0, metadata={"tag": "news", "year": 2024})
        self.assertFalse(self.store.matches_filter(0, {"tag": "news", "year": 2023}))

    def test_filter_key_absent_returns_false(self):
        self.store.add("hello", id=0, metadata={"tag": "news"})
        self.assertFalse(self.store.matches_filter(0, {"missing_key": "value"}))

    def test_filter_on_doc_with_no_metadata_returns_false(self):
        self.store.add("hello", id=0)
        self.assertFalse(self.store.matches_filter(0, {"tag": "news"}))

    # --- add_batch with metadata_list ---

    def test_add_batch_with_metadata_list(self):
        ids = self.store.add_batch(
            ["doc a", "doc b"],
            metadata_list=[{"tag": "x"}, {"tag": "y"}],
        )
        self.assertEqual(self.store.get_metadata(ids[0])["tag"], "x")
        self.assertEqual(self.store.get_metadata(ids[1])["tag"], "y")

    def test_add_batch_metadata_list_none_entries(self):
        ids = self.store.add_batch(["a", "b"], metadata_list=[{"k": "v"}, None])
        self.assertEqual(self.store.get_metadata(ids[0]), {"k": "v"})
        self.assertEqual(self.store.get_metadata(ids[1]), {})

    def test_add_batch_metadata_list_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.store.add_batch(["a", "b"], metadata_list=[{"k": "v"}])

    def test_add_batch_invalid_metadata_leaves_store_unchanged(self):
        try:
            self.store.add_batch(["a", "b"], metadata_list=[{"k": "v"}, {"bad": [1]}])  # type: ignore
        except TypeError:
            pass
        self.assertEqual(len(self.store), 0)


class TestDocumentStoreListSources(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    def test_empty_store_returns_empty_set(self):
        self.assertEqual(self.store.list_sources(), set())

    def test_single_doc_with_filename_returns_that_filename(self):
        self.store.add("text", metadata={"filename": "a.txt"})
        self.assertEqual(self.store.list_sources(), {"a.txt"})

    def test_multiple_docs_same_file_returns_one_entry(self):
        self.store.add("chunk 1", metadata={"filename": "a.txt"})
        self.store.add("chunk 2", metadata={"filename": "a.txt"})
        self.store.add("chunk 3", metadata={"filename": "a.txt"})
        self.assertEqual(self.store.list_sources(), {"a.txt"})

    def test_multiple_docs_different_files_returns_all(self):
        self.store.add("text a", metadata={"filename": "a.txt"})
        self.store.add("text b", metadata={"filename": "b.txt"})
        self.store.add("text c", metadata={"filename": "c.txt"})
        self.assertEqual(self.store.list_sources(), {"a.txt", "b.txt", "c.txt"})

    def test_docs_without_key_are_ignored(self):
        self.store.add("no filename", metadata={"other": "value"})
        self.store.add("with filename", metadata={"filename": "a.txt"})
        self.assertEqual(self.store.list_sources(), {"a.txt"})

    def test_docs_with_no_metadata_are_ignored(self):
        self.store.add("no meta at all")
        self.store.add("with meta", metadata={"filename": "b.txt"})
        self.assertEqual(self.store.list_sources(), {"b.txt"})

    def test_custom_key(self):
        self.store.add("text", metadata={"path": "/some/path.md"})
        self.assertEqual(self.store.list_sources(key="path"), {"/some/path.md"})

    def test_custom_key_missing_from_all_docs_returns_empty_set(self):
        self.store.add("text", metadata={"filename": "a.txt"})
        self.assertEqual(self.store.list_sources(key="nonexistent"), set())

    def test_returns_a_set_type(self):
        self.store.add("text", metadata={"filename": "a.txt"})
        self.assertIsInstance(self.store.list_sources(), set)

    def test_key_not_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.store.list_sources(key=123)  # type: ignore

    def test_key_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.list_sources(key="")

    def test_key_whitespace_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.store.list_sources(key="   ")

    def test_delete_removes_source_when_last_doc_gone(self):
        doc_id = self.store.add("only chunk", metadata={"filename": "a.txt"})
        self.assertIn("a.txt", self.store.list_sources())
        self.store.delete(doc_id)
        self.assertNotIn("a.txt", self.store.list_sources())

    def test_partial_delete_keeps_source_if_other_docs_remain(self):
        id1 = self.store.add("chunk 1", metadata={"filename": "a.txt"})
        self.store.add("chunk 2", metadata={"filename": "a.txt"})
        self.store.delete(id1)
        self.assertIn("a.txt", self.store.list_sources())

    def test_mixed_files_returns_correct_set(self):
        self.store.add("a1", metadata={"filename": "a.txt"})
        self.store.add("a2", metadata={"filename": "a.txt"})
        self.store.add("b1", metadata={"filename": "b.md"})
        result = self.store.list_sources()
        self.assertEqual(result, {"a.txt", "b.md"})


if __name__ == "__main__":
    unittest.main()
