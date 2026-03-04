import unittest
from neuroseek.vector import Vector
from neuroseek.embedder import Embedder, DEFAULT_MODEL


# The real model is downloaded once and reused across all tests in this module.
# Instantiation is done at module level so every test class shares the same
# loaded model — this keeps the suite fast without any mocking.
_embedder = Embedder()


class TestEmbedderInit(unittest.TestCase):

    def test_default_model_name(self):
        self.assertEqual(_embedder.model_name, DEFAULT_MODEL)

    def test_custom_model_name_stored(self):
        e = Embedder("all-MiniLM-L6-v2")
        self.assertEqual(e.model_name, "all-MiniLM-L6-v2")

    def test_dimension_is_positive_int(self):
        self.assertIsInstance(_embedder.dimension, int)
        self.assertGreater(_embedder.dimension, 0)

    def test_default_model_dimension_is_384(self):
        # multi-qa-MiniLM-L6-cos-v1 produces 384-dim embeddings
        self.assertEqual(_embedder.dimension, 384)

    def test_model_stored_internally(self):
        self.assertIsNotNone(_embedder._model)

    def test_model_name_not_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            Embedder(42)  # type: ignore

    def test_model_name_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            Embedder(None)  # type: ignore

    def test_model_name_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            Embedder("")

    def test_model_name_whitespace_raises_value_error(self):
        with self.assertRaises(ValueError):
            Embedder("   ")

    def test_invalid_model_name_raises(self):
        with self.assertRaises(Exception):
            Embedder("this-model-does-not-exist-xyz-123")


class TestEmbedderEncode(unittest.TestCase):

    def test_returns_vector(self):
        self.assertIsInstance(_embedder.encode("hello world"), Vector)

    def test_vector_has_correct_dimension(self):
        self.assertEqual(len(_embedder.encode("hello world")), 384)

    def test_vector_dimension_matches_embedder_dimension(self):
        result = _embedder.encode("test sentence")
        self.assertEqual(len(result), _embedder.dimension)

    def test_vector_data_is_list(self):
        self.assertIsInstance(_embedder.encode("hello").data, list)

    def test_vector_data_elements_are_floats(self):
        for val in _embedder.encode("hello").data:
            self.assertIsInstance(val, float)

    def test_different_texts_produce_different_vectors(self):
        v1 = _embedder.encode("the cat sat on the mat")
        v2 = _embedder.encode("quantum mechanics and black holes")
        self.assertNotEqual(v1, v2)

    def test_same_text_produces_identical_vectors(self):
        text = "deterministic encoding"
        self.assertEqual(_embedder.encode(text), _embedder.encode(text))

    def test_semantically_similar_texts_are_closer_than_unrelated(self):
        # "dog" and "puppy" should be closer than "dog" and "calculus"
        v_dog = _embedder.encode("dog")
        v_puppy = _embedder.encode("puppy")
        v_calculus = _embedder.encode("calculus")
        sim_related = v_dog.cosine_similarity(v_puppy)
        sim_unrelated = v_dog.cosine_similarity(v_calculus)
        self.assertGreater(sim_related, sim_unrelated)

    def test_cosine_similarity_of_identical_text_is_near_one(self):
        v = _embedder.encode("identical text")
        sim = v.cosine_similarity(v)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_non_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            _embedder.encode(123)  # type: ignore

    def test_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            _embedder.encode(None)  # type: ignore

    def test_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            _embedder.encode(["hello"])  # type: ignore

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            _embedder.encode("")

    def test_whitespace_only_raises_value_error(self):
        with self.assertRaises(ValueError):
            _embedder.encode("   ")

    def test_tab_newline_only_raises_value_error(self):
        with self.assertRaises(ValueError):
            _embedder.encode("\t\n")

    def test_does_not_mutate_input_text(self):
        original = "hello world"
        text = original
        _embedder.encode(text)
        self.assertEqual(text, original)


class TestEmbedderEncodeBatch(unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(_embedder.encode_batch(["hello", "world"]), list)

    def test_returns_correct_count(self):
        self.assertEqual(len(_embedder.encode_batch(["a", "b", "c"])), 3)

    def test_each_element_is_vector(self):
        for v in _embedder.encode_batch(["hello", "world"]):
            self.assertIsInstance(v, Vector)

    def test_each_vector_has_correct_dimension(self):
        for v in _embedder.encode_batch(["hello", "world"]):
            self.assertEqual(len(v), 384)

    def test_single_item_list(self):
        result = _embedder.encode_batch(["only one"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Vector)

    def test_tuple_input_accepted(self):
        result = _embedder.encode_batch(("hello", "world"))
        self.assertEqual(len(result), 2)

    def test_order_matches_individual_encode(self):
        texts = ["alpha", "beta", "gamma"]
        batch = _embedder.encode_batch(texts)
        singles = [_embedder.encode(t) for t in texts]
        for b, s in zip(batch, singles):
            # Batch inference and single inference may differ by tiny floating
            # point amounts; cosine similarity > 0.9999 confirms they represent
            # the same vector.
            self.assertGreater(b.cosine_similarity(s), 0.9999)

    def test_vector_data_elements_are_floats(self):
        for v in _embedder.encode_batch(["hello", "world"]):
            for val in v.data:
                self.assertIsInstance(val, float)

    def test_not_list_or_tuple_raises_type_error(self):
        with self.assertRaises(TypeError):
            _embedder.encode_batch("hello")  # type: ignore

    def test_dict_raises_type_error(self):
        with self.assertRaises(TypeError):
            _embedder.encode_batch({"a": 1})  # type: ignore

    def test_empty_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            _embedder.encode_batch([])

    def test_non_str_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            _embedder.encode_batch(["valid", 42])  # type: ignore

    def test_none_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            _embedder.encode_batch(["valid", None])  # type: ignore

    def test_empty_string_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            _embedder.encode_batch(["valid", ""])

    def test_whitespace_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            _embedder.encode_batch(["valid", "   "])

    def test_error_reports_offending_index(self):
        try:
            _embedder.encode_batch(["ok", "also ok", 99])  # type: ignore
        except TypeError as exc:
            self.assertIn("2", str(exc))
        else:
            self.fail("Expected TypeError was not raised")

    def test_error_on_first_element_reports_index_zero(self):
        try:
            _embedder.encode_batch([None, "ok"])  # type: ignore
        except TypeError as exc:
            self.assertIn("0", str(exc))
        else:
            self.fail("Expected TypeError was not raised")


if __name__ == "__main__":
    unittest.main()
