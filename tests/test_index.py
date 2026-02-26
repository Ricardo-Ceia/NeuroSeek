import unittest
from neuroseek import Vector, Index


class TestIndex(unittest.TestCase):
    def test_index_constructor(self):
        idx = Index()
        self.assertEqual(idx.vectors, [])
        self.assertEqual(idx.id_to_index, {})
        self.assertEqual(idx._next_id, 0)

    def test_add_vector_basic(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 1)
        self.assertEqual(len(idx.vectors), 1)
        self.assertEqual(idx.vectors[0][0], 1)
        self.assertEqual(list(idx.vectors[0][1].data), [1, 2, 3])

    def test_add_vector_multiple(self):
        idx = Index()
        for i in range(5):
            v = Vector(2)
            v.data = [i, i+1]
            idx.add_vector(v, i)
        self.assertEqual(len(idx.vectors), 5)
        self.assertEqual(len(idx.id_to_index), 5)

    def test_add_vector_auto_generates_id(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v1)
        idx.add_vector(v2)
        self.assertEqual(idx.vectors[0][0], 0)
        self.assertEqual(idx.vectors[1][0], 1)

    def test_add_vector_populates_id_to_index(self):
        idx = Index()
        v = Vector(3)
        v.data = [1, 2, 3]
        idx.add_vector(v, 42)
        self.assertEqual(idx.id_to_index[42], 0)

    def test_add_vector_duplicate_id_raises(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v1, 1)
        with self.assertRaises(ValueError):
            idx.add_vector(v2, 1)

    def test_add_vector_non_vector_raises(self):
        idx = Index()
        with self.assertRaises(TypeError):
            idx.add_vector([1, 2, 3], 1)

    def test_add_vector_invalid_id_type_raises(self):
        idx = Index()
        v = Vector(3)
        with self.assertRaises(TypeError):
            idx.add_vector(v, "abc")

    def test_add_vector_float_id_raises(self):
        idx = Index()
        v = Vector(3)
        with self.assertRaises(TypeError):
            idx.add_vector(v, 1.5)

    def test_add_vector_empty_vector(self):
        idx = Index()
        v = Vector(0)
        idx.add_vector(v, 1)
        self.assertEqual(len(idx.vectors), 1)

    def test_add_vector_with_existing_auto_id(self):
        idx = Index()
        v1 = Vector(3)
        v1.data = [1, 2, 3]
        v2 = Vector(3)
        v2.data = [4, 5, 6]
        idx.add_vector(v1, 0)
        idx.add_vector(v2)
        self.assertEqual(idx.vectors[1][0], 1)


if __name__ == "__main__":
    unittest.main()
