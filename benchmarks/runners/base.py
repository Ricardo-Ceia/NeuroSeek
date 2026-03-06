"""
benchmarks/runners/base.py
──────────────────────────
Abstract base class that every benchmark runner must implement.

Each runner wraps one retrieval system and exposes three methods:

  build_index(passages)          — ingest all passages
  query(text, top_k)             — return ranked list of passage IDs
  memory_usage_mb()              — RSS after indexing, in MB

The benchmark harness in run.py calls these methods, measures wall-clock
time and memory externally, and never inspects runner internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """Interface every benchmark runner must satisfy."""

    # Human-readable name shown in the results table.
    name: str = "unnamed"

    @abstractmethod
    def build_index(self, passages: list[tuple[str, str]]) -> None:
        """Ingest all passages into the retrieval system.

        Parameters
        ----------
        passages:
            List of ``(pid, text)`` pairs.  The runner must store the
            mapping from its internal result IDs back to ``pid`` so that
            :meth:`query` can return string passage IDs.
        """

    @abstractmethod
    def query(self, text: str, top_k: int = 10) -> list[str]:
        """Return the top-*k* passage IDs most relevant to *text*.

        Parameters
        ----------
        text:
            Query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[str]
            Passage IDs ranked best-first.  May be shorter than *top_k*
            if the index contains fewer passages.
        """

    @abstractmethod
    def memory_usage_mb(self) -> float:
        """Resident set size of the current process in megabytes.

        Called once after :meth:`build_index` completes.  Runners should
        return ``_rss_mb()`` from this module unless they have a better
        measurement available.
        """


def _rss_mb() -> float:
    """Return the RSS of the current process in MB (Linux/macOS)."""
    import os
    import resource
    # getrusage returns KB on Linux, bytes on macOS
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / 1_048_576  # bytes → MB
    return usage / 1_024  # KB → MB
