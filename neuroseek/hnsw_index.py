import random
import math
from neuroseek.vector import Vector
from neuroseek.hnsw_node import HNSWNode


class HNSWIndex:
    def __init__(self, M=16, efConstruction=200, maxLayers=16):
        self.M = M  # Number of connections per node
        self.efConstruction = efConstruction  # Search width during construction
        self.maxLayers = maxLayers
        self.layers = []  # List of dicts: layer -> {node_id: HNSWNode}
        self.id_to_node = {}  # node_id -> HNSWNode
        self.entry_point = None  # Top layer node
        self.num_vectors = 0

    def _get_random_layer(self):
        level = 0
        while random.random() < 0.5 and level < self.maxLayers:
            level += 1
        return level

    def _distance(self, v1, v2):
        return 1 - v1.cosine_similarity(v2)  # Convert similarity to distance

    def _search_layer(self, query, ef, layer):
        if not self.layers or layer >= len(self.layers) or not self.layers[layer]:
            return []

        visited = set()
        candidates = []  # [(node_id, distance)]
        results = []  # [(node_id, distance)]

        if self.entry_point:
            entry_id = self.entry_point.id
            entry_node = self.id_to_node[entry_id]
            dist = self._distance(query, entry_node.vector)
            candidates.append((entry_id, dist))
            results.append((entry_id, dist))
            visited.add(entry_id)

        while candidates:
            candidates.sort(key=lambda x: x[1])
            current_id, current_dist = candidates.pop(0)

            if results and current_dist > results[-1][1]:
                break

            current_node = self.id_to_node[current_id]
            for neighbor_id, neighbor_dist in current_node.get_connections(layer):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_node = self.id_to_node[neighbor_id]
                    dist = self._distance(query, neighbor_node.vector)

                    if len(results) < ef or dist < results[-1][1]:
                        candidates.append((neighbor_id, dist))
                        results.append((neighbor_id, dist))
                        results.sort(key=lambda x: x[1])
                        if len(results) > ef:
                            results.pop()

        return results[:ef]

    def add_vector(self, vector, id=None):
        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector, not {type(vector).__name__}")

        if id is None:
            id = self.num_vectors

        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")

        if id in self.id_to_node:
            raise ValueError(f"ID {id} already exists")

        node = HNSWNode(id=id, vector=vector, layer=0)
        self.id_to_node[id] = node
        self.num_vectors += 1

        if not self.layers:
            self.layers.append({})
            self.entry_point = node

        while len(self.layers) <= node.layer:
            self.layers.append({})

        self.layers[node.layer][id] = node

        if node.layer > 0:
            search_layers = list(range(node.layer, len(self.layers) - 1))
        else:
            search_layers = list(range(len(self.layers) - 1))

        for layer in reversed(search_layers):
            neighbors = self._search_layer(vector, self.efConstruction, layer)
            for neighbor_id, _ in neighbors[:self.M]:
                neighbor_node = self.id_to_node[neighbor_id]
                node.add_connection(neighbor_id, self._distance(vector, neighbor_node.vector), node.layer)
                neighbor_node.add_connection(id, self._distance(neighbor_node.vector, vector), node.layer)

        return id

    def get_vector(self, id):
        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")

        if id not in self.id_to_node:
            raise ValueError(f"ID {id} does not exist")

        return self.id_to_node[id].vector

    def search(self, query, top_k=5, ef=10):
        if not isinstance(query, Vector):
            raise TypeError(f"query must be a Vector, not {type(query).__name__}")

        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be an int, not {type(top_k).__name__}")

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if not self.id_to_node:
            return []

        if ef < top_k:
            ef = top_k

        results = []
        for layer in reversed(range(len(self.layers))):
            layer_results = self._search_layer(query, ef, layer)
            if layer_results:
                best_id = layer_results[0][0]
                self.entry_point = self.id_to_node[best_id]

        final_results = self._search_layer(query, ef, 0)
        final_results.sort(key=lambda x: x[1])

        return [(node_id, 1 - dist) for node_id, dist in final_results[:top_k]]

    def __len__(self):
        return self.num_vectors
