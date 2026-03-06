"""
benchmarks/runners/chroma_runner.py
─────────────────────────────────────
Benchmark runner for ChromaDB — an embedded vector database.

Uses an EphemeralClient (in-memory, no files written to disk) and supplies
pre-computed embeddings from the shared Embedder so that model load time is
not counted in index build time and all vector runners use identical vectors.

ChromaDB uses cosine similarity by default when embeddings are supplied
without a distance function override; we set it explicitly to be safe.
"""

from __future__ import annotations

import numpy as np
import chromadb

from neuroseek.embedder import Embedder
from benchmarks.runners.base import BaseRunner, _rss_mb

# ChromaDB collection add() takes at most 41,666 items per batch by default;
# we chunk to stay well under that limit.
_BATCH_SIZE = 5_000


class ChromaRunner(BaseRunner):
    name = "ChromaDB"

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._collection: chromadb.Collection | None = None

    def build_index(self, passages: list[tuple[str, str]]) -> None:
        client = chromadb.EphemeralClient()
        self._collection = client.create_collection(
            name="benchmark",
            metadata={"hnsw:space": "cosine"},
        )

        pids  = [pid  for pid, _ in passages]
        texts = [text for _, text in passages]

        # Embed in one batch then add to Chroma in chunks
        vectors = self._embedder.encode_batch(texts)
        embeddings = [v.data for v in vectors]

        for start in range(0, len(passages), _BATCH_SIZE):
            end = start + _BATCH_SIZE
            self._collection.add(
                ids=pids[start:end],
                embeddings=embeddings[start:end],
            )

    def query(self, text: str, top_k: int = 10) -> list[str]:
        assert self._collection is not None, "call build_index first"
        vec = self._embedder.encode(text).data
        k = min(top_k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[vec],
            n_results=k,
            include=[],   # we only need the ids, not documents or distances
        )
        return results["ids"][0]

    def memory_usage_mb(self) -> float:
        return _rss_mb()
