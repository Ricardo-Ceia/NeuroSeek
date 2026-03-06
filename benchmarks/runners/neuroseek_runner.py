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

from neuroseek.embedder import Embedder
from neuroseek.search_engine import SearchEngine

from benchmarks.runners.base import BaseRunner, _rss_mb


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

    def build_index(self, passages: list[tuple[str, str]]) -> None:
        """Embed and index all passages.

        Parameters
        ----------
        passages:
            List of ``(pid, text)`` pairs.
        """
        self._engine = SearchEngine._from_embedder(self._embedder)
        self._id_to_pid = {}

        texts = [text for _, text in passages]
        pids  = [pid  for pid, _ in passages]

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
