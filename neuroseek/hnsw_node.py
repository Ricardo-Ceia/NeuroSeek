from neuroseek.vector import Vector


class HNSWNode:
    def __init__(self, id, vector, layer=0):
        self.id = id
        self.vector = vector
        self.layer = layer
        self.connections = {}  # layer -> list of (node_id, distance)

    def add_connection(self, neighbor_id, distance, layer=None):
        if layer is None:
            layer = self.layer
        
        if layer not in self.connections:
            self.connections[layer] = []
        
        self.connections[layer].append((neighbor_id, distance))

    def get_connections(self, layer=None):
        if layer is None:
            layer = self.layer
        return self.connections.get(layer, [])

    def __repr__(self):
        return f"HNSWNode(id={self.id}, layer={self.layer}, connections={len(self.connections)})"
