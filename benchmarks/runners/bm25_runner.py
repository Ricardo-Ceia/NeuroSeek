"""
benchmarks/runners/bm25_runner.py
──────────────────────────────────
Benchmark runner for BM25 (rank-bm25, BM25Okapi variant).

BM25 is a keyword-based retrieval model — no embeddings, no vectors.
It tokenises passages and queries by whitespace-lowercasing, then ranks
by term frequency / inverse document frequency with length normalisation.

This is the classic IR baseline.  Accuracy will be lower than semantic
models on paraphrase queries; speed and memory will be much better.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from benchmarks.runners.base import BaseRunner, _rss_mb


def _tokenise(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Runner(BaseRunner):
    name = "BM25"

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._pids: list[str] = []

    def build_index(self, passages: list[tuple[str, str]]) -> None:
        self._pids = [pid for pid, _ in passages]
        tokenised = [_tokenise(text) for _, text in passages]
        self._bm25 = BM25Okapi(tokenised)

    def query(self, text: str, top_k: int = 10) -> list[str]:
        assert self._bm25 is not None, "call build_index first"
        tokens = _tokenise(text)
        scores = self._bm25.get_scores(tokens)
        # argsort descending — take top_k
        k = min(top_k, len(self._pids))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._pids[i] for i in top_indices]

    def memory_usage_mb(self) -> float:
        return _rss_mb()
