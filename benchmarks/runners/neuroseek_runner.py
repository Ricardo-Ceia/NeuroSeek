"""
benchmarks/runners/neuroseek_runner.py
──────────────────────────────────────
Benchmark runner for NeuroSeek — the reference implementation.

Uses SearchEngine directly (not the CLI) so we measure the library layer,
not subprocess overhead.  A single shared Embedder is passed in at
construction time so that embedding time is counted separately from index
build time when the harness chooses to do so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroseek.embedder import Embedder
from neuroseek.search_engine import SearchEngine
from neuroseek.core.vector import Vector

from benchmarks.runners.base import BaseRunner, _rss_mb

if TYPE_CHECKING:
    import numpy as np


class NeuroSeekRunner(BaseRunner):
    name = "NeuroSeek"

    def __init__(self, embedder: Embedder) -> None:
        """
        Parameters
        ----------
        embedder:
            Pre-loaded Embedder instance shared across all runners so that
            model load time is not counted in index build time.
        """
        self._embedder = embedder
        self._engine: SearchEngine | None = None
        # pid at list index i  →  used to map integer doc IDs back to pids
        self._id_to_pid: dict[int, str] = {}

    def build_index(
        self,
        passages: list[tuple[str, str]],
        vectors: "np.ndarray | None" = None,
    ) -> None:
        """Embed and index all passages.

        Parameters
        ----------
        passages:
            List of ``(pid, text)`` pairs.
        vectors:
            Optional pre-computed embedding matrix (N × dim, float32).
            When provided the Embedder is not called, saving embedding time
            when the same matrix is shared across multiple runners.
        """
        self._engine = SearchEngine._from_embedder(self._embedder)
        self._id_to_pid = {}

        pids = [pid for pid, _ in passages]

        if vectors is not None:
            # Wrap numpy rows into Vector objects and insert directly
            dim = vectors.shape[1]
            vec_objects = []
            for row in vectors:
                v = Vector(dim)
                v.data = row.tolist()
                vec_objects.append(v)
            assigned_ids = self._engine._index.add_vectors(vec_objects)
            # Also populate the document store with placeholder texts
            texts = [text for _, text in passages]
            for text, doc_id in zip(texts, assigned_ids):
                self._engine._store.add(text, id=doc_id)
        else:
            texts = [text for _, text in passages]
            assigned_ids = self._engine.add_batch(texts)

        for doc_id, pid in zip(assigned_ids, pids):
            self._id_to_pid[doc_id] = pid

    def query(self, text: str, top_k: int = 10) -> list[str]:
        """Return the top-*k* passage IDs most similar to *text*."""
        assert self._engine is not None, "call build_index first"
        results = self._engine.search(text, top_k=top_k)
        return [self._id_to_pid[r["id"]] for r in results]

    def memory_usage_mb(self) -> float:
        return _rss_mb()
