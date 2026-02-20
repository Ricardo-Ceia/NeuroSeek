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

    def test_len(self):
        v = Vector(5)
        self.assertEqual(len(v), 5)

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

    def test_getitem_out_of_bounds_raises(self):
        v = Vector(3)
        with self.assertRaises(IndexError):
            _ = v[3]
        with self.assertRaises(IndexError):
            _ = v[10]

    def test_repr(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        r = repr(v)
        self.assertIn("1", r)
        self.assertIn("2", r)
        self.assertIn("3", r)

    def test_add_two_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        result = v1 + v2
        self.assertEqual(list(result.data), [5, 7, 9])

    def test__mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            _ = v1 + v2

    def test_sub_two_vectors(self):
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        result = v1 - v2
        self.assertEqual(list(result.data), [-3, -3, -3])

    def test_sub_mismatched_sizes_raises(self):
        v1 = Vector(3)
        v2 = Vector(5)
        with self.assertRaises(ValueError):
            _ = v1 - v2



if __name__ == "__main__":
    unittest.main()
