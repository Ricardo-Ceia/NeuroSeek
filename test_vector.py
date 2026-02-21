import unittest
from main import Vector


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
        with self.assertRaises(AttributeError):
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
        with self.assertRaises(AttributeError):
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
        with self.assertRaises(AttributeError):
            _ = v * [1, 2, 3]

    def test_dot_basic(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        self.assertEqual(v1.__dot__(v2), 32)

    def test_dot_returns_scalar(self):
        v1 = Vector(3)
        v2 = Vector(3)
        result = v1.__dot__(v2)
        self.assertIsInstance(result, (int, float))
        self.assertNotIsInstance(result, Vector)

    def test_dot_with_zeros(self):
        v1 = Vector(3)
        v1.data = [0, 0, 0]
        v2 = Vector(3)
        v2.data = [1, 2, 3]
        self.assertEqual(v1.__dot__(v2), 0)

    def test_dot_with_negatives(self):
        v1 = Vector(3)
        v1.data = [-1, 2, -3]
        v2 = Vector(3)
        v2.data = [4, -5, 6]
        self.assertEqual(v1.__dot__(v2), -32)

    def test_dot_with_floats(self):
        v1 = Vector(2)
        v1.data = [1.5, 2.5]
        v2 = Vector(2)
        v2.data = [2.0, 3.0]
        self.assertAlmostEqual(v1.__dot__(v2), 10.5)

    def test_dot_empty_vectors(self):
        v1 = Vector(0)
        v2 = Vector(0)
        self.assertEqual(v1.__dot__(v2), 0)

    def test_dot_mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            v1.__dot__(v2)

    def test_dot_non_vector_raises(self):
        v = Vector(3)
        with self.assertRaises(AttributeError):
            v.__dot__([1, 2, 3])

    def test_dot_single_element(self):
        v1 = Vector(1)
        v1.data = [5]
        v2 = Vector(1)
        v2.data = [3]
        self.assertEqual(v1.__dot__(v2), 15)

    def test_dot_orthogonal_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 0, 0]
        v2 = Vector(3)
        v2.data = [0, 1, 0]
        self.assertEqual(v1.__dot__(v2), 0)

    def test_dot_same_vector_squared(self):
        v = Vector(3)
        v.data = [3, 4, 0]
        result = v.__dot__(v)
        self.assertEqual(result, 25)

    def test_rmul_scalar_on_left(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = 3 * v
        self.assertEqual(list(result.data), [3, 6, 9])
        self.assertIsInstance(result, Vector)

    def test_rmul_returns_vector_type(self):
        v = Vector(3)
        result = 2 * v
        self.assertIsInstance(result, Vector)

    def test_rmul_with_zero(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = 0 * v
        self.assertEqual(list(result.data), [0, 0, 0])

    def test_rmul_with_negative(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = -2 * v
        self.assertEqual(list(result.data), [-2, -4, -6])

    def test_rmul_with_float(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        result = 0.5 * v
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertAlmostEqual(result[2], 1.5)

    def test_rmul_empty_vector(self):
        v = Vector(0)
        result = 5 * v
        self.assertEqual(list(result.data), [])

    def test_rmul_non_numeric_raises(self):
        v = Vector(3)
        with self.assertRaises(TypeError):
            _ = "string" * v

    def test_norm_basic(self):
        v = Vector(2)
        v.data = [3, 4]
        self.assertEqual(v.__norm__(), 5.0)

    def test_norm_returns_scalar(self):
        v = Vector(3)
        result = v.__norm__()
        self.assertIsInstance(result, (int, float))
        self.assertNotIsInstance(result, Vector)

    def test_norm_zero_vector(self):
        v = Vector(3)
        v.data = [0, 0, 0]
        self.assertEqual(v.__norm__(), 0)

    def test_norm_with_negatives(self):
        v = Vector(3)
        v.data = [-3, -4, 0]
        self.assertEqual(v.__norm__(), 5.0)

    def test_norm_with_floats(self):
        v = Vector(2)
        v.data = [1.5, 2.0]
        self.assertAlmostEqual(v.__norm__(), 2.5)

    def test_norm_empty_vector(self):
        v = Vector(0)
        self.assertEqual(v.__norm__(), 0)

    def test_norm_single_element(self):
        v = Vector(1)
        v.data = [5]
        self.assertEqual(v.__norm__(), 5.0)

    def test_norm_single_element_zero(self):
        v = Vector(1)
        v.data = [0]
        self.assertEqual(v.__norm__(), 0.0)

    def test_norm_larger_vector(self):
        v = Vector(4)
        v.data = [1, 2, 2, 4]
        self.assertAlmostEqual(v.__norm__(), 5.0)

    def test_norm_uses_dot_product(self):
        v = Vector(3)
        v.data = [3, 4, 0]
        self.assertEqual(v.__norm__() ** 2, v.__dot__(v))

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
        with self.assertRaises(AttributeError):
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


if __name__ == "__main__":
    unittest.main()
