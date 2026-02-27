import unittest
from neuroseek import Vector
from neuroseek.hnsw_node import HNSWNode


class TestHNSWNode(unittest.TestCase):
    def test_constructor_basic(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=0)
        self.assertEqual(node.id, 1)
        self.assertEqual(node.layer, 0)
        self.assertEqual(list(node.vector.data), [1, 2, 3])
        self.assertEqual(node.connections, {})

    def test_constructor_default_layer(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v)
        self.assertEqual(node.layer, 0)

    def test_constructor_multiple_layers(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=3)
        self.assertEqual(node.layer, 3)

    def test_add_connection_basic(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=0)
        node.add_connection(neighbor_id=2, distance=0.5, layer=0)
        connections = node.get_connections(layer=0)
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0], (2, 0.5))

    def test_add_connection_multiple(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=0)
        node.add_connection(neighbor_id=2, distance=0.5, layer=0)
        node.add_connection(neighbor_id=3, distance=0.3, layer=0)
        node.add_connection(neighbor_id=4, distance=0.7, layer=0)
        connections = node.get_connections(layer=0)
        self.assertEqual(len(connections), 3)

    def test_add_connection_different_layers(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=2)
        node.add_connection(neighbor_id=2, distance=0.5, layer=0)
        node.add_connection(neighbor_id=3, distance=0.3, layer=1)
        node.add_connection(neighbor_id=4, distance=0.7, layer=2)
        self.assertEqual(len(node.get_connections(0)), 1)
        self.assertEqual(len(node.get_connections(1)), 1)
        self.assertEqual(len(node.get_connections(2)), 1)

    def test_add_connection_without_layer_uses_node_layer(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=1)
        node.add_connection(neighbor_id=2, distance=0.5)
        connections = node.get_connections(layer=1)
        self.assertEqual(len(connections), 1)

    def test_get_connections_no_connections(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=0)
        connections = node.get_connections(layer=0)
        self.assertEqual(connections, [])

    def test_get_connections_layer_not_exist(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=0)
        connections = node.get_connections(layer=5)
        self.assertEqual(connections, [])

    def test_get_connections_default_layer(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=1)
        node.add_connection(neighbor_id=2, distance=0.5)
        connections = node.get_connections()
        self.assertEqual(len(connections), 1)

    def test_repr(self):
        v = Vector(3)
        v.data = [1, 2, 3]
        node = HNSWNode(id=1, vector=v, layer=0)
        node.add_connection(neighbor_id=2, distance=0.5, layer=0)
        r = repr(node)
        self.assertIn("id=1", r)
        self.assertIn("layer=0", r)


if __name__ == "__main__":
    unittest.main()
