"""
benchmarks/runners/hnswlib_runner.py
─────────────────────────────────────
Benchmark runner for hnswlib — the C++ HNSW library.

Uses the same Embedder (and therefore the same model) as NeuroSeekRunner so
that the only difference being measured is the index implementation.
hnswlib operates on raw numpy float32 arrays; we use cosine space to match
NeuroSeek's cosine similarity ranking.
"""

from __future__ import annotations

import numpy as np

import hnswlib

from neuroseek.embedder import Embedder

from benchmarks.runners.base import BaseRunner, _rss_mb


class HNSWLibRunner(BaseRunner):
    name = "hnswlib"

    def __init__(self, embedder: Embedder, M: int = 16, ef_construction: int = 200) -> None:
        self._embedder = embedder
        self._M = M
        self._ef_construction = ef_construction
        self._index: hnswlib.Index | None = None
        # integer label → pid string
        self._label_to_pid: dict[int, str] = {}

    def build_index(
        self,
        passages: list[tuple[str, str]],
        vectors: "np.ndarray | None" = None,
    ) -> None:
        """Build the HNSW index.

        Parameters
        ----------
        passages:
            List of ``(pid, text)`` pairs.
        vectors:
            Optional pre-computed embedding matrix (N × dim, float32).
            When provided the Embedder is not called, saving embedding time
            when the same matrix is shared across multiple runners.
        """
        pids  = [pid  for pid, _ in passages]
        texts = [text for _, text in passages]

        if vectors is not None:
            matrix = vectors.astype(np.float32)
        else:
            # Embed all passages in one batch
            emb = self._embedder.encode_batch(texts)
            matrix = np.array([v.data for v in emb], dtype=np.float32)

        dim = matrix.shape[1]
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=len(passages),
            M=self._M,
            ef_construction=self._ef_construction,
            random_seed=42,
        )
        # Use integer labels 0..N-1; store mapping to pids
        labels = np.arange(len(passages), dtype=np.int64)
        self._index.add_items(matrix, labels)
        self._index.set_ef(50)  # ef at query time — standard default
        self._label_to_pid = {i: pid for i, pid in enumerate(pids)}

    def query(self, text: str, top_k: int = 10) -> list[str]:
        assert self._index is not None, "call build_index first"
        vec = np.array([self._embedder.encode(text).data], dtype=np.float32)
        k = min(top_k, self._index.get_current_count())
        labels, _ = self._index.knn_query(vec, k=k)
        return [self._label_to_pid[int(lbl)] for lbl in labels[0]]

    def memory_usage_mb(self) -> float:
        return _rss_mb()
