"""
benchmarks/runners/faiss_runner.py
───────────────────────────────────
Benchmark runner for FAISS (faiss-cpu) — Facebook's vector search library.

Uses IndexHNSWFlat with inner-product metric on L2-normalised vectors, which
is equivalent to cosine similarity.  Same Embedder as all other vector
runners so the only variable is the index.
"""

from __future__ import annotations

import numpy as np
import faiss

from neuroseek.embedder import Embedder

from benchmarks.runners.base import BaseRunner, _rss_mb


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row so that inner product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class FAISSRunner(BaseRunner):
    name = "FAISS"

    def __init__(self, embedder: Embedder, M: int = 16) -> None:
        self._embedder = embedder
        self._M = M
        self._index: faiss.IndexHNSWFlat | None = None
        # FAISS uses sequential 0-based integer IDs; map back to pids
        self._id_to_pid: dict[int, str] = {}

    def build_index(
        self,
        passages: list[tuple[str, str]],
        vectors: "np.ndarray | None" = None,
    ) -> None:
        """Build the FAISS HNSW index.

        Parameters
        ----------
        passages:
            List of ``(pid, text)`` pairs.
        vectors:
            Optional pre-computed embedding matrix (N × dim, float32).
            When provided the Embedder is not called.
        """
        pids  = [pid  for pid, _ in passages]
        texts = [text for _, text in passages]

        if vectors is not None:
            matrix = vectors.astype(np.float32)
        else:
            emb = self._embedder.encode_batch(texts)
            matrix = np.array([v.data for v in emb], dtype=np.float32)
        matrix = _normalise(matrix)

        dim = matrix.shape[1]
        # IndexHNSWFlat with METRIC_INNER_PRODUCT on normalised vectors
        # gives cosine similarity ranking
        self._index = faiss.IndexHNSWFlat(dim, self._M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = 200
        self._index.add(matrix)
        self._index.hnsw.efSearch = 50
        self._id_to_pid = {i: pid for i, pid in enumerate(pids)}

    def query(self, text: str, top_k: int = 10) -> list[str]:
        assert self._index is not None, "call build_index first"
        vec = np.array([self._embedder.encode(text).data], dtype=np.float32)
        vec = _normalise(vec)
        k = min(top_k, self._index.ntotal)
        _, indices = self._index.search(vec, k)
        # FAISS returns -1 for empty slots — filter those out
        return [
            self._id_to_pid[int(i)]
            for i in indices[0]
            if i >= 0
        ]

    def memory_usage_mb(self) -> float:
        return _rss_mb()
