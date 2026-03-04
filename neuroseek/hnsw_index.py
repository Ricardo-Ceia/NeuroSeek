import heapq
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
        self._next_id = 0  # Monotonically increasing; never decremented on delete

    def _get_random_layer(self):
        level = 0
        while random.random() < 0.5 and level < self.maxLayers:
            level += 1
        return level

    def _distance(self, v1, v2):
        return 1 - v1.cosine_similarity(v2)  # Convert similarity to distance

    def _search_layer_from(self, query, entry_node, ef, layer, exclude_id=None):
        """Search a single layer starting from a given entry node.

        Unlike the old _search_layer, this accepts an explicit entry_node and
        never reads or writes self.entry_point, making it safe to call from
        both add_vector (construction) and search (query time).

        If entry_node does not participate in the requested layer, the search
        falls back to any node that does exist in that layer (skipping
        exclude_id, which is used to avoid the newly-inserted node during
        construction).
        """
        if not self.layers or layer >= len(self.layers) or not self.layers[layer]:
            return []

        # If the provided entry_node is not in this layer, fall back to any
        # node that is — skipping exclude_id (the node being inserted).
        if entry_node.id not in self.layers[layer]:
            fallback_id = next(
                (nid for nid in self.layers[layer] if nid != exclude_id),
                None
            )
            if fallback_id is None:
                return []  # layer has only the new node itself
            entry_node = self.id_to_node[fallback_id]

        visited = set()
        entry_dist = self._distance(query, entry_node.vector)
        # candidates: min-heap by distance (closest first)
        candidates = [(entry_dist, entry_node.id)]
        # results: max-heap by distance (farthest first, so we can prune)
        results = [(-entry_dist, entry_node.id)]
        visited.add(entry_node.id)

        while candidates:
            current_dist, current_id = heapq.heappop(candidates)

            # Worst (farthest) result so far
            worst_result_dist = -results[0][0]
            if current_dist > worst_result_dist and len(results) >= ef:
                break

            current_node = self.id_to_node[current_id]
            for neighbor_id, _ in current_node.get_connections(layer):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_node = self.id_to_node[neighbor_id]
                    dist = self._distance(query, neighbor_node.vector)

                    worst_result_dist = -results[0][0]
                    if len(results) < ef or dist < worst_result_dist:
                        heapq.heappush(candidates, (dist, neighbor_id))
                        heapq.heappush(results, (-dist, neighbor_id))
                        if len(results) > ef:
                            heapq.heappop(results)

        # Convert max-heap back to sorted list [(node_id, dist), ...] ascending
        sorted_results = sorted(
            [(nid, -neg_dist) for neg_dist, nid in results],
            key=lambda x: x[1]
        )
        return sorted_results[:ef]

    def add_vector(self, vector, id=None):
        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector, not {type(vector).__name__}")

        if id is None:
            while self._next_id in self.id_to_node:
                self._next_id += 1
            id = self._next_id
            self._next_id += 1

        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")

        if id in self.id_to_node:
            raise ValueError(f"ID {id} already exists")

        # Keep _next_id strictly ahead of every manually-supplied ID so that
        # future auto-IDs never collide with IDs that were (or will be) used
        # manually — even if those manual IDs have since been deleted.
        if id >= self._next_id:
            self._next_id = id + 1

        node_layer = self._get_random_layer()
        node = HNSWNode(id=id, vector=vector, layer=node_layer)
        self.id_to_node[id] = node
        self.num_vectors += 1

        # Expand layer list if needed
        while len(self.layers) <= node_layer:
            self.layers.append({})

        if self.entry_point is None:
            # First node — register and make it the entry point immediately
            for lyr in range(node_layer + 1):
                self.layers[lyr][id] = node
            self.entry_point = node
            return id

        ep = self.entry_point

        # Phase 1: greedily descend layers ABOVE node_layer — find the best
        # entry point for the connection-building phase without adding edges.
        # The new node is NOT yet registered in the layers during this phase so
        # it does not interfere with the search.
        for layer in reversed(range(node_layer + 1, len(self.layers))):
            layer_results = self._search_layer_from(vector, ep, ef=1, layer=layer)
            if layer_results:
                ep = self.id_to_node[layer_results[0][0]]

        # Phase 2: build connections on layers 0..node_layer.
        # Register the node layer by layer as we descend so it is visible to
        # back-connections but does not seed its own search.
        for layer in reversed(range(node_layer + 1)):
            self.layers[layer][id] = node  # register before searching this layer
            neighbors = self._search_layer_from(
                vector, ep, self.efConstruction, layer, exclude_id=id
            )
            for neighbor_id, dist in neighbors[:self.M]:
                neighbor_node = self.id_to_node[neighbor_id]
                node.add_connection(neighbor_id, dist, layer)
                # Only add a back-connection if the neighbor actually exists at this layer
                if neighbor_node.layer >= layer:
                    neighbor_node.add_connection(id, dist, layer)
            if neighbors:
                # Best neighbor in this layer is the entry point for the layer below
                ep = self.id_to_node[neighbors[0][0]]

        # If the new node reaches a higher layer than the current entry point,
        # it becomes the new global entry point.
        if node_layer >= self.entry_point.layer:
            self.entry_point = node

        return id

    def add_vectors(self, vectors, ids=None):
        if not isinstance(vectors, (list, tuple)):
            raise TypeError(f"vectors must be a list or tuple, not {type(vectors).__name__}")

        if ids is None:
            ids = [None] * len(vectors)

        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"ids must be a list, tuple, or None, not {type(ids).__name__}")

        if len(vectors) != len(ids):
            raise ValueError(f"vectors and ids must have the same length")

        for id in ids:
            if id is not None and not isinstance(id, int):
                raise TypeError(f"id must be an int or None, not {type(id).__name__}")

        returned_ids = []
        for vector, id in zip(vectors, ids):
            returned_ids.append(self.add_vector(vector, id))

        return returned_ids

    def get_vector(self, id):
        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")

        if id not in self.id_to_node:
            raise ValueError(f"ID {id} does not exist")

        return self.id_to_node[id].vector

    def delete_vector(self, id):
        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")

        if id not in self.id_to_node:
            raise ValueError(f"ID {id} does not exist")

        node = self.id_to_node[id]

        for layer in range(node.layer + 1):
            if layer < len(self.layers) and id in self.layers[layer]:
                del self.layers[layer][id]

        for neighbor_id, neighbor_node in self.id_to_node.items():
            for layer in neighbor_node.connections:
                neighbor_node.connections[layer] = [
                    (nid, dist) for nid, dist in neighbor_node.connections[layer]
                    if nid != id
                ]

        del self.id_to_node[id]
        self.num_vectors -= 1

        if self.entry_point and self.entry_point.id == id:
            self._repair_entry_point()

        return node.vector

    def _repair_entry_point(self):
        """Select a new entry point after the current one has been deleted.

        Scans layers from highest to lowest and picks any surviving node.
        Also trims trailing empty layers so len(self.layers) stays accurate.
        If the index is now empty, entry_point is set to None.
        """
        # Remove empty top layers
        while self.layers and not self.layers[-1]:
            self.layers.pop()

        # Walk remaining layers top-down for a new entry point
        self.entry_point = None
        for layer_dict in reversed(self.layers):
            if layer_dict:
                ep_id = next(iter(layer_dict))
                self.entry_point = self.id_to_node[ep_id]
                break

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

        ep = self.entry_point

        # Greedily descend from the top layer to layer 1, tracking the closest
        # node as the entry point for the next layer — no state mutation.
        for layer in reversed(range(1, len(self.layers))):
            layer_results = self._search_layer_from(query, ep, ef=1, layer=layer)
            if layer_results:
                ep = self.id_to_node[layer_results[0][0]]

        final_results = self._search_layer_from(query, ep, ef, layer=0)
        final_results.sort(key=lambda x: x[1])

        return [(node_id, 1 - dist) for node_id, dist in final_results[:top_k]]

    def __len__(self):
        return self.num_vectors
