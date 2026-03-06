"""
benchmarks/runners/whoosh_runner.py
─────────────────────────────────────
Benchmark runner for Whoosh — a pure-Python full-text search library.

Whoosh uses an inverted index with BM25F scoring (the default).  It operates
on an in-memory RAM storage backend so no files are written to disk during
the benchmark.  Tokenisation uses Whoosh's standard analyser (lowercase +
stop-word removal).
"""

from __future__ import annotations

import whoosh.index as windex
import whoosh.fields as wfields
import whoosh.qparser as wqparser
import whoosh.scoring as wscoring
from whoosh.filedb.filestore import RamStorage

from benchmarks.runners.base import BaseRunner, _rss_mb


class WhooshRunner(BaseRunner):
    name = "Whoosh"

    def __init__(self) -> None:
        self._ix: windex.FileIndex | None = None
        self._schema: wfields.Schema | None = None

    def build_index(self, passages: list[tuple[str, str]]) -> None:
        schema = wfields.Schema(
            pid=wfields.ID(stored=True, unique=True),
            content=wfields.TEXT(stored=False),
        )
        self._schema = schema

        # Use RAM storage — no disk I/O during the benchmark
        st = RamStorage()
        self._ix = st.create_index(schema)

        writer = self._ix.writer()
        for pid, text in passages:
            writer.add_document(pid=pid, content=text)
        writer.commit()

    def query(self, text: str, top_k: int = 10) -> list[str]:
        assert self._ix is not None, "call build_index first"
        parser = wqparser.QueryParser("content", schema=self._schema)
        q = parser.parse(text)
        with self._ix.searcher(weighting=wscoring.BM25F()) as searcher:
            results = searcher.search(q, limit=top_k)
            return [r["pid"] for r in results]

    def memory_usage_mb(self) -> float:
        return _rss_mb()
