import heapq
import random
import math
import numpy as np
from neuroseek.core.vector import Vector
from neuroseek.core.hnsw_node import HNSWNode

_INITIAL_CAPACITY = 256


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

        # --- numpy vector matrix -----------------------------------------------
        # Rows are dense; _id_to_row / _row_to_id map between node IDs and rows.
        # _free_rows holds row indices that were freed by delete_vector so they
        # can be reused without growing the matrix.
        # _dim is set on first insertion and validated on every subsequent one.
        self._dim: int = 0
        self._capacity: int = _INITIAL_CAPACITY
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._id_to_row: dict[int, int] = {}
        self._row_to_id: dict[int, int] = {}
        self._free_rows: list[int] = []

        # --- reverse adjacency for O(degree) delete ----------------------------
        # _reverse_adj[node_id] = set of node_ids that have node_id as a neighbor
        # (across all layers).  Maintained in sync with HNSWNode.connections.
        self._reverse_adj: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_random_layer(self) -> int:
        """Return a random layer using the HNSW paper formula.

        level = floor(-ln(uniform(0,1)) / ln(M))  capped at maxLayers-1.
        This produces ~log_M(N) layers instead of the O(log_2(N)) layers that
        the old coin-flip approach generated.
        """
        level = int(-math.log(random.random()) / math.log(max(self.M, 2)))
        return min(level, self.maxLayers - 1)

    def _distance(self, v1: Vector, v2: Vector) -> float:
        """Fallback scalar distance used outside the hot search path."""
        return 1.0 - v1.cosine_similarity(v2)

    def _ensure_matrix(self, dim: int) -> None:
        """Allocate or grow the numpy matrix to hold at least one more vector."""
        if self._dim == 0:
            # First insertion — fix dimension and allocate.
            self._dim = dim
            self._matrix = np.empty((_INITIAL_CAPACITY, dim), dtype=np.float32)
            self._capacity = _INITIAL_CAPACITY
        elif dim != self._dim:
            raise ValueError(
                f"Vector dimension {dim} does not match index dimension {self._dim}"
            )
        else:
            # Grow if needed (no free rows left and matrix is full).
            used = len(self._id_to_row)
            if used >= self._capacity and not self._free_rows:
                new_cap = self._capacity * 2
                new_mat = np.empty((new_cap, self._dim), dtype=np.float32)
                new_mat[: self._capacity] = self._matrix
                self._matrix = new_mat
                self._capacity = new_cap

    def _alloc_row(self, node_id: int, vec_np: np.ndarray) -> int:
        """Assign a matrix row for node_id and store the pre-normalised vector."""
        if self._free_rows:
            row = self._free_rows.pop()
        else:
            row = len(self._id_to_row)
        self._matrix[row] = vec_np
        self._id_to_row[node_id] = row
        self._row_to_id[row] = node_id
        return row

    def _free_row(self, node_id: int) -> None:
        """Release the matrix row held by node_id."""
        row = self._id_to_row.pop(node_id, None)
        if row is not None:
            self._row_to_id.pop(row, None)
            self._free_rows.append(row)

    def _query_np(self, query: Vector) -> np.ndarray:
        """Return a normalised float32 numpy array for query."""
        arr = np.array(query.data, dtype=np.float32)
        n = np.linalg.norm(arr)
        if n == 0:
            raise ValueError("Query vector has zero norm.")
        return arr / n

    def _dist_row(self, row: int, query_np: np.ndarray) -> float:
        """Cosine distance between a stored (pre-normalised) row and query_np."""
        return 1.0 - float(self._matrix[row] @ query_np)

    def _dist_rows_batch(self, rows: list[int], query_np: np.ndarray) -> np.ndarray:
        """Vectorised cosine distances for a batch of rows."""
        return 1.0 - self._matrix[rows] @ query_np  # shape (len(rows),)

    def _add_edge(self, src_id: int, dst_id: int, dist: float, layer: int) -> None:
        """Add a directed edge src→dst and update reverse adjacency."""
        self.id_to_node[src_id].add_connection(dst_id, dist, layer)
        self._reverse_adj.setdefault(dst_id, set()).add(src_id)

    def _remove_edge(self, src_id: int, dst_id: int, layer: int) -> None:
        """Remove directed edge src→dst and update reverse adjacency."""
        node = self.id_to_node[src_id]
        if layer in node.connections:
            node.connections[layer] = [
                (nid, d) for nid, d in node.connections[layer] if nid != dst_id
            ]
        rev = self._reverse_adj.get(dst_id)
        if rev is not None:
            rev.discard(src_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_layer_from(
        self,
        query: Vector,
        entry_node: HNSWNode,
        ef: int,
        layer: int,
        exclude_id: int | None = None,
        query_np: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Search a single layer starting from a given entry node.

        Uses pre-normalised numpy rows for all distance computations:
        - Entry-point distance: single dot product.
        - Candidate neighbours: batched dot product over all unvisited neighbours.

        Returns list of (node_id, distance) sorted ascending, length <= ef.
        """
        if not self.layers or layer >= len(self.layers) or not self.layers[layer]:
            return []

        # If the provided entry_node is not in this layer, fall back to any
        # node that is — skipping exclude_id (the node being inserted).
        if entry_node.id not in self.layers[layer]:
            fallback_id = next(
                (nid for nid in self.layers[layer] if nid != exclude_id),
                None,
            )
            if fallback_id is None:
                return []
            entry_node = self.id_to_node[fallback_id]

        if query_np is None:
            query_np = self._query_np(query)

        # Entry point distance — single dot product on pre-normalised row.
        ep_row = self._id_to_row.get(entry_node.id)
        if ep_row is None:
            return []
        entry_dist = self._dist_row(ep_row, query_np)

        visited = {entry_node.id}
        # candidates: min-heap by distance (closest first)
        candidates = [(entry_dist, entry_node.id)]
        # results: max-heap by distance (farthest first, so we can prune)
        results = [(-entry_dist, entry_node.id)]

        while candidates:
            current_dist, current_id = heapq.heappop(candidates)

            worst_result_dist = -results[0][0]
            if current_dist > worst_result_dist and len(results) >= ef:
                break

            current_node = self.id_to_node[current_id]
            # Collect all unvisited neighbours in one pass, then batch-compute
            # distances with a single BLAS call.
            unvisited_ids: list[int] = []
            unvisited_rows: list[int] = []
            for neighbor_id, _ in current_node.get_connections(layer):
                if neighbor_id not in visited:
                    row = self._id_to_row.get(neighbor_id)
                    if row is not None:
                        visited.add(neighbor_id)
                        unvisited_ids.append(neighbor_id)
                        unvisited_rows.append(row)

            if not unvisited_ids:
                continue

            # Batched distance computation — one matrix-vector multiply.
            dists = self._dist_rows_batch(unvisited_rows, query_np)

            worst_result_dist = -results[0][0]
            for neighbor_id, dist in zip(unvisited_ids, dists):
                dist = float(dist)
                if len(results) < ef or dist < worst_result_dist:
                    heapq.heappush(candidates, (dist, neighbor_id))
                    heapq.heappush(results, (-dist, neighbor_id))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst_result_dist = -results[0][0]

        sorted_results = sorted(
            [(nid, -neg_dist) for neg_dist, nid in results],
            key=lambda x: x[1],
        )
        return sorted_results[:ef]

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_vector(self, vector: Vector, id: int | None = None) -> int:
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

        if id >= self._next_id:
            self._next_id = id + 1

        # Pre-normalise vector and store in the numpy matrix.
        dim = len(vector)
        self._ensure_matrix(dim)
        arr = np.array(vector.data, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Cannot insert a zero-norm vector into HNSWIndex.")
        arr_norm = arr / norm
        self._alloc_row(id, arr_norm)

        node_layer = self._get_random_layer()
        node = HNSWNode(id=id, vector=vector, layer=node_layer)
        self.id_to_node[id] = node
        self._reverse_adj[id] = set()
        self.num_vectors += 1

        # Expand layer list if needed.
        while len(self.layers) <= node_layer:
            self.layers.append({})

        if self.entry_point is None:
            for lyr in range(node_layer + 1):
                self.layers[lyr][id] = node
            self.entry_point = node
            return id

        ep = self.entry_point
        # Build a single query_np once; reuse across all layer searches.
        query_np = arr_norm  # already normalised

        # Phase 1: greedy descent through layers ABOVE node_layer.
        for layer in reversed(range(node_layer + 1, len(self.layers))):
            layer_results = self._search_layer_from(
                vector, ep, ef=1, layer=layer, query_np=query_np
            )
            if layer_results:
                ep = self.id_to_node[layer_results[0][0]]

        # Phase 2: build connections on layers 0..node_layer.
        for layer in reversed(range(node_layer + 1)):
            self.layers[layer][id] = node
            neighbors = self._search_layer_from(
                vector, ep, self.efConstruction, layer,
                exclude_id=id, query_np=query_np,
            )
            for neighbor_id, dist in neighbors[: self.M]:
                neighbor_node = self.id_to_node[neighbor_id]
                self._add_edge(id, neighbor_id, dist, layer)
                # Back-connection: only if the neighbor participates at this layer
                # and hasn't yet reached its M-connection cap.
                if neighbor_node.layer >= layer:
                    existing = neighbor_node.get_connections(layer)
                    if len(existing) < self.M:
                        self._add_edge(neighbor_id, id, dist, layer)
                    else:
                        # Replace the farthest back-connection if the new node
                        # is closer (keeps the graph quality high).
                        farthest_idx = max(
                            range(len(existing)), key=lambda i: existing[i][1]
                        )
                        if dist < existing[farthest_idx][1]:
                            old_id = existing[farthest_idx][0]
                            self._remove_edge(neighbor_id, old_id, layer)
                            self._add_edge(neighbor_id, id, dist, layer)
            if neighbors:
                ep = self.id_to_node[neighbors[0][0]]

        if node_layer >= self.entry_point.layer:
            self.entry_point = node

        return id

    def add_vectors(
        self,
        vectors: list[Vector],
        ids: list[int | None] | None = None,
    ) -> list[int]:
        if not isinstance(vectors, (list, tuple)):
            raise TypeError(f"vectors must be a list or tuple, not {type(vectors).__name__}")

        if ids is None:
            ids = [None] * len(vectors)

        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"ids must be a list, tuple, or None, not {type(ids).__name__}")

        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")

        for id in ids:
            if id is not None and not isinstance(id, int):
                raise TypeError(f"id must be an int or None, not {type(id).__name__}")

        returned_ids = []
        for vector, id in zip(vectors, ids):
            returned_ids.append(self.add_vector(vector, id))

        return returned_ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_vector(self, id: int) -> Vector:
        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")

        if id not in self.id_to_node:
            raise ValueError(f"ID {id} does not exist")

        return self.id_to_node[id].vector

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_vector(self, id: int) -> Vector:
        if not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")

        if id not in self.id_to_node:
            raise ValueError(f"ID {id} does not exist")

        node = self.id_to_node[id]

        # Remove from layer dicts.
        for layer in range(node.layer + 1):
            if layer < len(self.layers) and id in self.layers[layer]:
                del self.layers[layer][id]

        # Use reverse adjacency to find exactly which nodes point TO this node,
        # then remove only those edges — O(degree × L) instead of O(N × L × M).
        for src_id in list(self._reverse_adj.get(id, set())):
            if src_id in self.id_to_node:
                src_node = self.id_to_node[src_id]
                for layer in list(src_node.connections.keys()):
                    src_node.connections[layer] = [
                        (nid, d) for nid, d in src_node.connections[layer]
                        if nid != id
                    ]
                # Clean up forward-direction reverse-adj entries.
                for layer_connections in src_node.connections.values():
                    for (nid, _) in layer_connections:
                        pass  # already cleaned above via the list comprehension

        # Remove forward-direction reverse-adj entries (this node pointed TO others).
        for layer_connections in node.connections.values():
            for (dst_id, _) in layer_connections:
                rev = self._reverse_adj.get(dst_id)
                if rev is not None:
                    rev.discard(id)

        # Free the reverse-adj entry for this node.
        self._reverse_adj.pop(id, None)

        # Release the numpy matrix row.
        self._free_row(id)

        del self.id_to_node[id]
        self.num_vectors -= 1

        if self.entry_point and self.entry_point.id == id:
            self._repair_entry_point()

        return node.vector

    def _repair_entry_point(self) -> None:
        """Select a new entry point after the current one has been deleted."""
        while self.layers and not self.layers[-1]:
            self.layers.pop()

        self.entry_point = None
        for layer_dict in reversed(self.layers):
            if layer_dict:
                ep_id = next(iter(layer_dict))
                self.entry_point = self.id_to_node[ep_id]
                break

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self, query: Vector, top_k: int = 5, ef: int = 10
    ) -> list[tuple[int, float]]:
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

        query_np = self._query_np(query)
        ep = self.entry_point

        for layer in reversed(range(1, len(self.layers))):
            layer_results = self._search_layer_from(
                query, ep, ef=1, layer=layer, query_np=query_np
            )
            if layer_results:
                ep = self.id_to_node[layer_results[0][0]]

        final_results = self._search_layer_from(
            query, ep, ef, layer=0, query_np=query_np
        )
        final_results.sort(key=lambda x: x[1])

        return [(node_id, 1.0 - dist) for node_id, dist in final_results[:top_k]]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_vectors
