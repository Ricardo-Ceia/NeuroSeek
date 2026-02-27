import unittest
from neuroseek import Vector


class TestVector(unittest.TestCase):
    def test_constructor_creates_zero_filled_vector(self):
        v = Vector(3)
        self.assertEqual(v.size, 3)
        self.assertEqual(list(v.data), [0, 0, 0])

    def test_constructor_with_different_sizes(self):
        for size in [1, 5, 100]:
            v = Vector(size)
            self.assertEqual(v.size, size)
            self.assertEqual(len(v.data), size)

    def test_constructor_empty_vector(self):
        v = Vector(0)
        self.assertEqual(v.size, 0)
        self.assertEqual(list(v.data), [])

    def test_len(self):
        v = Vector(5)
        self.assertEqual(len(v), 5)

    def test_len_empty(self):
        v = Vector(0)
        self.assertEqual(len(v), 0)

    def test_getitem_valid_index(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        self.assertEqual(v[2], 3)

    def test_getitem_negative_index(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        self.assertEqual(v[-1], 3)
        self.assertEqual(v[-2], 2)
        self.assertEqual(v[-3], 1)

    def test_getitem_out_of_bounds_raises(self):
        v = Vector(3)
        with self.assertRaises(IndexError):
            _ = v[3]
        with self.assertRaises(IndexError):
            _ = v[10]

    def test_getitem_negative_out_of_bounds_raises(self):
        v = Vector(3)
        with self.assertRaises(IndexError):
            _ = v[-4]
        with self.assertRaises(IndexError):
            _ = v[-10]

    def test_getitem_invalid_type_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            _ = v["a"]
        with self.assertRaises(TypeError):
            _ = v[1.5]

    def test_repr(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        r = repr(v)
        self.assertIn("1", r)
        self.assertIn("2", r)
        self.assertIn("3", r)

    def test_repr_empty(self):
        v = Vector(0)
        r = repr(v)
        self.assertEqual(r, "[]")

    def test_add_two_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        result = v1 + v2
        self.assertEqual(list(result.data), [5, 7, 9])
        self.assertIsInstance(result, Vector)

    def test_add_returns_vector_type(self):
        v1 = Vector(3)
        v2 = Vector(3)
        result = v1 + v2
        self.assertIsInstance(result, Vector)

    def test_add_with_zeros(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [0, 0, 0]
        result = v1 + v2
        self.assertEqual(list(result.data), [1, 2, 3])

    def test_add_with_negatives(self):
        v1 = Vector(3)
        v1.data = [-1, 2, -3]
        v2 = Vector(3)
        v2.data = [4, -5, 6]
        result = v1 + v2
        self.assertEqual(list(result.data), [3, -3, 3])

    def test_add_with_floats(self):
        v1 = Vector(2)
        v1.data = [1.5, 2.5]
        v2 = Vector(2)
        v2.data = [0.5, 1.5]
        result = v1 + v2
        self.assertAlmostEqual(result[0], 2.0)
        self.assertAlmostEqual(result[1], 4.0)

    def test_add_empty_vectors(self):
        v1 = Vector(0)
        v2 = Vector(0)
        result = v1 + v2
        self.assertEqual(list(result.data), [])

    def test_add_mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            _ = v1 + v2

    def test_add_non_vector_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            _ = v + [1, 2, 3]

    def test_sub_two_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        result = v1 - v2
        self.assertEqual(list(result.data), [-3, -3, -3])
        self.assertIsInstance(result, Vector)

    def test_sub_returns_vector_type(self):
        v1 = Vector(3)
        v2 = Vector(3)
        result = v1 - v2
        self.assertIsInstance(result, Vector)

    def test_sub_with_zeros(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [0, 0, 0]
        result = v1 - v2
        self.assertEqual(list(result.data), [1, 2, 3])

    def test_sub_with_negatives(self):
        v1 = Vector(3)
        v1.data = [-1, 2, -3]
        v2 = Vector(3)
        v2.data = [4, -5, 6]
        result = v1 - v2
        self.assertEqual(list(result.data), [-5, 7, -9])

    def test_sub_same_vector_equals_zero(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = v - v
        self.assertEqual(list(result.data), [0, 0, 0])

    def test_sub_empty_vectors(self):
        v1 = Vector(0)
        v2 = Vector(0)
        result = v1 - v2
        self.assertEqual(list(result.data), [])

    def test_sub_mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            _ = v1 - v2

    def test_sub_non_vector_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            _ = v - [1, 2, 3]

    def test_mul_two_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        result = v1 * v2
        self.assertEqual(list(result.data), [4, 10, 18])
        self.assertIsInstance(result, Vector)

    def test_mul_returns_vector_type(self):
        v1 = Vector(3)
        v2 = Vector(3)
        result = v1 * v2
        self.assertIsInstance(result, Vector)

    def test_mul_with_zeros(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [0, 0, 0]
        result = v1 * v2
        self.assertEqual(list(result.data), [0, 0, 0])

    def test_mul_with_negatives(self):
        v1 = Vector(3)
        v1.data = [-1, 2, -3]
        v2 = Vector(3)
        v2.data = [4, -5, 6]
        result = v1 * v2
        self.assertEqual(list(result.data), [-4, -10, -18])

    def test_mul_empty_vectors(self):
        v1 = Vector(0)
        v2 = Vector(0)
        result = v1 * v2
        self.assertEqual(list(result.data), [])

    def test_mul_mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            _ = v1 * v2

    def test_mul_non_vector_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            _ = v * [1, 2, 3]

    def test_mul_scalar_on_right(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = v * 3
        self.assertEqual(list(result.data), [3, 6, 9])
        self.assertIsInstance(result, Vector)

    def test_mul_scalar_with_zero(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = v * 0
        self.assertEqual(list(result.data), [0, 0, 0])

    def test_mul_scalar_with_negative(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = v * -2
        self.assertEqual(list(result.data), [-2, -4, -6])

    def test_mul_scalar_with_float(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = v * 0.5
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertAlmostEqual(result[2], 1.5)

    def test_mul_scalar_empty_vector(self):
        v = Vector(0)
        result = v * 5
        self.assertEqual(list(result.data), [])

    def test_mul_scalar_single_element(self):
        v = Vector(1)
        v.data = [5]
        result = v * 3
        self.assertEqual(list(result.data), [15])

    def test_mul_scalar_and_vector_both_work(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [2, 2, 2]
        self.assertEqual(list((v1 * 2).data), [2, 4, 6])
        self.assertEqual(list((v1 * v2).data), [2, 4, 6])

    def test_dot_basic(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        self.assertEqual(v1.dot(v2), 32)

    def test_dot_returns_scalar(self):
        v1 = Vector(3)
        v2 = Vector(3)
        result = v1.dot(v2)
        self.assertIsInstance(result, (int, float))
        self.assertNotIsInstance(result, Vector)

    def test_dot_with_zeros(self):
        v1 = Vector(3)
        v1.data = [0, 0, 0]
        v2 = Vector(3)
        v2.data = [1, 2, 3]
        self.assertEqual(v1.dot(v2), 0)

    def test_dot_with_negatives(self):
        v1 = Vector(3)
        v1.data = [-1, 2, -3]
        v2 = Vector(3)
        v2.data = [4, -5, 6]
        self.assertEqual(v1.dot(v2), -32)

    def test_dot_with_floats(self):
        v1 = Vector(2)
        v1.data = [1.5, 2.5]
        v2 = Vector(2)
        v2.data = [2.0, 3.0]
        self.assertAlmostEqual(v1.dot(v2), 10.5)

    def test_dot_empty_vectors(self):
        v1 = Vector(0)
        v2 = Vector(0)
        self.assertEqual(v1.dot(v2), 0)

    def test_dot_mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            v1.dot(v2)

    def test_dot_non_vector_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            v.dot([1, 2, 3])

    def test_dot_single_element(self):
        v1 = Vector(1)
        v1.data = [5]
        v2 = Vector(1)
        v2.data = [3]
        self.assertEqual(v1.dot(v2), 15)

    def test_dot_orthogonal_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        self.assertEqual(v1.dot(v2), 0)

    def test_dot_same_vector_squared(self):
        v = Vector(3)
        v.data = [3, 4, 0]
        result = v.dot(v)
        self.assertEqual(result, 25)

    def test_matmul_basic(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        result = v1 @ v2
        self.assertEqual(result, 32)

    def test_matmul_returns_same_as_dot(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        self.assertEqual(v1 @ v2, v1.dot(v2))

    def test_matmul_mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            _ = v1 @ v2
    def test_norm_basic(self):
        v = Vector(2)
        v.data = [3, 4]
        self.assertEqual(v.norm(), 5.0)

    def test_norm_returns_scalar(self):
        v = Vector(3)
        result = v.norm()
        self.assertIsInstance(result, (int, float))
        self.assertNotIsInstance(result, Vector)

    def test_norm_zero_vector(self):
        v = Vector(3)
        v.data = [0, 0, 0]
        self.assertEqual(v.norm(), 0)

    def test_norm_with_negatives(self):
        v = Vector(3)
        v.data = [-3, -4, 0]
        self.assertEqual(v.norm(), 5.0)

    def test_norm_with_floats(self):
        v = Vector(2)
        v.data = [1.5, 2.0]
        self.assertAlmostEqual(v.norm(), 2.5)

    def test_norm_empty_vector(self):
        v = Vector(0)
        self.assertEqual(v.norm(), 0)

    def test_norm_single_element(self):
        v = Vector(1)
        v.data = [5]
        self.assertEqual(v.norm(), 5.0)

    def test_norm_single_element_zero(self):
        v = Vector(1)
        v.data = [0]
        self.assertEqual(v.norm(), 0.0)

    def test_norm_larger_vector(self):
        v = Vector(4)
        v.data = [1, 2, 2, 4]
        self.assertAlmostEqual(v.norm(), 5.0)

    def test_norm_uses_dot_product(self):
        v = Vector(3)
        v.data = [3, 4, 0]
        self.assertEqual(v.norm() ** 2, v.dot(v))

    def test_eq_equal_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [1, 2, 3]
        self.assertTrue(v1 == v2)

    def test_eq_different_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [1, 2, 4]
        self.assertFalse(v1 == v2)

    def test_eq_different_sizes(self):
        v1 = Vector(3)
        v2 = Vector(5)
        self.assertFalse(v1 == v2)

    def test_eq_empty_vectors(self):
        v1 = Vector(0)
        v2 = Vector(0)
        self.assertTrue(v1 == v2)

    def test_eq_with_floats(self):
        v1 = Vector(2)
        v1.data = [1.5, 2.5]
        v2 = Vector(2)
        v2.data = [1.5, 2.5]
        self.assertTrue(v1 == v2)

    def test_eq_with_negatives(self):
        v1 = Vector(3)
        v1.data = [-1, -2, -3]
        v2 = Vector(3)
        v2.data = [-1, -2, -3]
        self.assertTrue(v1 == v2)

    def test_eq_same_vector(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        self.assertTrue(v == v)

    def test_eq_non_vector_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            _ = v == [1, 2, 3]

    def test_nequal_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        self.assertTrue(v1 != v2)

    def test_ne_different_sizes(self):
        v1 = Vector(3)
        v2 = Vector(5)
        self.assertTrue(v1 != v2)

    def test_neg_basic(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = -v
        self.assertEqual(list(result.data), [-1, -2, -3])
        self.assertIsInstance(result, Vector)

    def test_neg_returns_vector_type(self):
        v = Vector(3)
        result = -v
        self.assertIsInstance(result, Vector)

    def test_neg_already_negative(self):
        v = Vector(3)
        v.data = [-1, -2, -3]
        result = -v
        self.assertEqual(list(result.data), [1, 2, 3])

    def test_neg_with_zeros(self):
        v = Vector(3)
        v.data = [0, 0, 0]
        result = -v
        self.assertEqual(list(result.data), [0, 0, 0])

    def test_neg_with_floats(self):
        v = Vector(2)
        v.data = [1.5, -2.5]
        result = -v
        self.assertAlmostEqual(result[0], -1.5)
        self.assertAlmostEqual(result[1], 2.5)

    def test_neg_empty_vector(self):
        v = Vector(0)
        result = -v
        self.assertEqual(list(result.data), [])

    def test_neg_single_element(self):
        v = Vector(1)
        v.data = [5]
        result = -v
        self.assertEqual(list(result.data), [-5])

    def test_neg_double_negation(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = -(-v)
        self.assertEqual(list(result.data), [1, 2, 3])

    def test_setitem_basic(self):
        v = Vector(3)
        v[0] = 5
        self.assertEqual(v[0], 5)

    def test_setitem_multiple(self):
        v = Vector(3)
        v[0] = 1
        v[1] = 2
        v[2] = 3
        self.assertEqual(list(v.data), [1, 2, 3])

    def test_setitem_negative_index(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        v[-1] = 5
        self.assertEqual(v[2], 5)

    def test_setitem_out_of_bounds_raises(self):
        v = Vector(3)
        with self.assertRaises(IndexError):
            v[3] = 5
        with self.assertRaises(IndexError):
            v[10] = 5

    def test_setitem_negative_out_of_bounds_raises(self):
        v = Vector(3)
        with self.assertRaises(IndexError):
            v[-4] = 5

    def test_setitem_invalid_type_index_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            v["a"] = 5
        with self.assertRaises(TypeError):
            v[1.5] = 5

    def test_setitem_with_float(self):
        v = Vector(2)
        v[0] = 1.5
        self.assertAlmostEqual(v[0], 1.5)

    def test_setitem_with_negative(self):
        v = Vector(2)
        v[0] = -5
        self.assertEqual(v[0], -5)

    def test_setitem_empty_vector_raises(self):
        v = Vector(0)
        with self.assertRaises(IndexError):
            v[0] = 5

    def test_iter_basic(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = list(v)
        self.assertEqual(result, [1, 2, 3])

    def test_iter_with_for_loop(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        collected = []
        for x in v:
            collected.append(x)
        self.assertEqual(collected, [1, 2, 3])

    def test_iter_empty_vector(self):
        v = Vector(0)
        result = list(v)
        self.assertEqual(result, [])

    def test_iter_single_element(self):
        v = Vector(1)
        v.data = [5]
        result = list(v)
        self.assertEqual(result, [5])

    def test_iter_with_floats(self):
        v = Vector(2)
        v.data = [1.5, 2.5]
        result = list(v)
        self.assertAlmostEqual(result[0], 1.5)
        self.assertAlmostEqual(result[1], 2.5)

    def test_iter_with_negatives(self):
        v = Vector(3)
        v.data = [-1, 0, 1]
        result = list(v)
        self.assertEqual(result, [-1, 0, 1])

    def test_iter_multiple_iterations(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        it1 = iter(v)
        it2 = iter(v)
        self.assertEqual(next(it1), 1)
        self.assertEqual(next(it2), 1)
        self.assertEqual(next(it1), 2)
        self.assertEqual(next(it2), 2)

    def test_contains_basic(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        self.assertTrue(2 in v)
        self.assertTrue(1 in v)
        self.assertTrue(3 in v)

    def test_contains_not_present(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        self.assertFalse(5 in v)
        self.assertFalse(0 in v)

    def test_contains_with_zeros(self):
        v = Vector(3)
        v.data = [0, 0, 0]
        self.assertTrue(0 in v)
        self.assertFalse(1 in v)

    def test_contains_with_floats(self):
        v = Vector(3)
        v.data = [1.5, 2.5, 3.5]
        self.assertTrue(2.5 in v)
        self.assertFalse(1.0 in v)

    def test_contains_with_negatives(self):
        v = Vector(3)
        v.data = [-1, 0, 1]
        self.assertTrue(-1 in v)
        self.assertTrue(0 in v)
        self.assertFalse(-2 in v)

    def test_contains_empty_vector(self):
        v = Vector(0)
        self.assertFalse(1 in v)

    def test_contains_single_element(self):
        v = Vector(1)
        v.data = [5]
        self.assertTrue(5 in v)
        self.assertFalse(3 in v)

    def test_contains_invalid_type_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            _ = "a" in v
        with self.assertRaises(TypeError):
            _ = [1] in v

    def test_cosine_similarity_identical(self):
        v1 = Vector(2)
        v1.data = [1, 0]
        v2 = Vector(2)
        v2.data = [1, 0]
        self.assertEqual(v1.cosine_similarity(v2), 1.0)

    def test_cosine_similarity_returns_scalar(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        result = v1.cosine_similarity(v2)
        self.assertIsInstance(result, (int, float))
        self.assertNotIsInstance(result, Vector)

    def test_cosine_similarity_orthogonal(self):
        v1 = Vector(2)
        v1.data = [1, 0]
        v2 = Vector(2)
        v2.data = [0, 1]
        self.assertEqual(v1.cosine_similarity(v2), 0.0)

    def test_cosine_similarity_opposite(self):
        v1 = Vector(2)
        v1.data = [1, 0]
        v2 = Vector(2)
        v2.data = [-1, 0]
        self.assertEqual(v1.cosine_similarity(v2), -1.0)

    def test_cosine_similarity_with_floats(self):
        v1 = Vector(2)
        v1.data = [1.5, 2.5]
        v2 = Vector(2)
        v2.data = [3.0, 4.0]
        result = v1.cosine_similarity(v2)
        self.assertAlmostEqual(result, 0.9947, places=3)

    def test_cosine_similarity_different_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            v1.cosine_similarity(v2)

    def test_cosine_similarity_zero_vector_raises(self):
        v1 = Vector(3)
        v1.data = [0, 0, 0]
        v2 = Vector(3)
        v2.data = [1, 2, 3]
        with self.assertRaises(ValueError):
            v1.cosine_similarity(v2)

    def test_cosine_similarity_both_zero_vectors_raises(self):
        v1 = Vector(3)
        v1.data = [0, 0, 0]
        v2 = Vector(3)
        v2.data = [0, 0, 0]
        with self.assertRaises(ValueError):
            v1.cosine_similarity(v2)

    def test_cosine_similarity_non_vector_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            v.cosine_similarity([1, 2, 3])

    def test_cosine_similarity_45_degrees(self):
        v1 = Vector(2)
        v1.data = [1, 1]
        v2 = Vector(2)
        v2.data = [1, 0]
        result = v1.cosine_similarity(v2)
        self.assertAlmostEqual(result, 0.7071, places=3)

    def test_cosine_similarity_with_negatives(self):
        v1 = Vector(2)
        v1.data = [1, -1]
        v2 = Vector(2)
        v2.data = [-1, 1]
        self.assertAlmostEqual(v1.cosine_similarity(v2), -1.0)

    def test_cosine_similarity_single_element(self):
        v1 = Vector(1)
        v1.data = [3]
        v2 = Vector(1)
        v2.data = [4]
        self.assertEqual(v1.cosine_similarity(v2), 1.0)

    def test_cosine_similarity_matches_formula(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        dot = v1.dot(v2)
        norms = v1.norm() * v2.norm()
        expected = dot / norms
        self.assertEqual(v1.cosine_similarity(v2), expected)


if __name__ == "__main__":
    unittest.main()
