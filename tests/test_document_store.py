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


if __name__ == "__main__":
    unittest.main()
