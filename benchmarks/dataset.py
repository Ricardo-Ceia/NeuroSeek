"""
benchmarks/dataset.py
─────────────────────
Download and cache the MS MARCO passage-ranking dataset subset used for
all NeuroSeek benchmarks.

What gets downloaded
--------------------
Three files from the official MS MARCO distribution:

  collection.tar.gz   ~2.9 GB  pid TAB passage  (8.8M passages total)
  queries.tar.gz      ~42 MB   qid TAB query     (dev split used)
  qrels.dev.tsv       ~1.1 MB  TREC qrels format (59k dev judgements)

The collection is streamed directly from the URL — the HTTP connection is
closed as soon as ``max_passages`` lines have been saved, so only a small
fraction of the 2.9 GB archive is ever downloaded.  Queries and qrels are
small enough to download in full.

Only the first ``max_passages`` rows of the collection are kept in memory
(default 10 000).  Queries are filtered to those that have at least one
relevant passage inside that subset, so accuracy metrics are meaningful.

Public API
----------
  load(data_dir, max_passages, max_queries) -> Dataset

  Dataset.passages  : list[Passage]   — (pid: str, text: str)
  Dataset.queries   : list[Query]     — (qid: str, text: str)
  Dataset.qrels     : Qrels           — dict[qid, dict[pid, int]]

The files are downloaded once and cached in ``data_dir``.  Subsequent calls
return immediately from the local cache.
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import NamedTuple


# ---------------------------------------------------------------------------
# URLs (official MS MARCO distribution)
# ---------------------------------------------------------------------------

_COLLECTION_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
)
_QUERIES_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
)
_QRELS_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv"
)

# Cached filenames inside data_dir
_COLLECTION_CACHE = "collection.tsv"
_QUERIES_CACHE = "queries.dev.tsv"
_QRELS_CACHE = "qrels.dev.tsv"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Passage(NamedTuple):
    pid: str
    text: str


class Query(NamedTuple):
    qid: str
    text: str


# qrels: qid -> {pid -> relevance_score}
Qrels = dict[str, dict[str, int]]


@dataclass
class Dataset:
    passages: list[Passage]
    queries: list[Query]
    qrels: Qrels

    def __repr__(self) -> str:
        return (
            f"Dataset("
            f"passages={len(self.passages)}, "
            f"queries={len(self.queries)}, "
            f"qrel_pairs={sum(len(v) for v in self.qrels.values())}"
            f")"
        )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    """Download *url* to *dest*, printing a progress indicator."""
    print(f"  Downloading {os.path.basename(dest)} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1_048_576
    print(f"  Saved {dest} ({size_mb:.1f} MB)", flush=True)


def _ensure_collection(data_dir: str, max_passages: int) -> str:
    """Return path to the cached flat TSV collection file.

    Streams the collection tarball directly from the URL without saving it to
    disk.  The HTTP connection is closed as soon as ``max_passages`` lines have
    been written, so we never download the full 2.9 GB archive.
    """
    cache_path = os.path.join(data_dir, _COLLECTION_CACHE)
    if os.path.exists(cache_path):
        return cache_path

    print(
        f"  Streaming first {max_passages:,} passages from MS MARCO "
        f"(connection closed early — no full download required) ...",
        flush=True,
    )

    response = urllib.request.urlopen(_COLLECTION_URL)
    try:
        # Wrap the raw HTTP stream in a streaming GzipFile so we decompress
        # on the fly without buffering the whole archive.
        gz_stream = gzip.GzipFile(fileobj=response)
        # tarfile can open a streaming GzipFile directly in 'r|' (pipe) mode.
        with tarfile.open(fileobj=gz_stream, mode="r|") as tar:
            for member in tar:
                if not member.name.endswith(".tsv"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                with open(cache_path, "w", encoding="utf-8") as out:
                    for i, raw_line in enumerate(f):
                        if i >= max_passages:
                            break
                        out.write(raw_line.decode("utf-8", errors="replace"))
                # Found and processed the collection member — stop iterating.
                break
    finally:
        response.close()

    print(f"  Collection subset saved to {cache_path}", flush=True)
    return cache_path


def _ensure_queries(data_dir: str) -> str:
    """Return path to the cached dev queries TSV."""
    cache_path = os.path.join(data_dir, _QUERIES_CACHE)
    if os.path.exists(cache_path):
        return cache_path

    tarball_path = os.path.join(data_dir, "queries.tar.gz")
    if not os.path.exists(tarball_path):
        _download(_QUERIES_URL, tarball_path)

    print("  Extracting dev queries ...", flush=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        # The archive contains queries.train.tsv and queries.dev.tsv
        member = next(
            m for m in tar.getmembers() if "dev" in m.name and m.name.endswith(".tsv")
        )
        f = tar.extractfile(member)
        assert f is not None, "Could not open queries member in tarball"
        with open(cache_path, "wb") as out:
            out.write(f.read())

    os.remove(tarball_path)
    print(f"  Dev queries saved to {cache_path}", flush=True)
    return cache_path


def _ensure_qrels(data_dir: str) -> str:
    """Return path to the cached dev qrels TSV."""
    cache_path = os.path.join(data_dir, _QRELS_CACHE)
    if os.path.exists(cache_path):
        return cache_path
    _download(_QRELS_URL, cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_collection(path: str, max_passages: int) -> list[Passage]:
    """Read up to *max_passages* lines from the flat TSV collection cache."""
    passages: list[Passage] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_passages:
                break
            line = line.rstrip("\n")
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            pid, text = parts
            if pid and text:
                passages.append(Passage(pid=pid, text=text))
    return passages


def _parse_queries(path: str) -> dict[str, str]:
    """Return qid -> query text mapping."""
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t", 1)
            if len(parts) == 2:
                qid, text = parts
                if qid and text:
                    queries[qid] = text
    return queries


def _parse_qrels(path: str) -> Qrels:
    """Parse TREC qrels format: ``qid 0 pid relevance``.

    Returns qid -> {pid -> relevance} mapping.
    Relevance scores are integers (1 = relevant for MS MARCO dev).
    """
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _zero, pid, rel = parts[0], parts[1], parts[2], parts[3]
            rel_int = int(rel)
            if rel_int > 0:  # only keep relevant judgements
                qrels.setdefault(qid, {})[pid] = rel_int
    return qrels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(
    data_dir: str | None = None,
    max_passages: int = 10_000,
    max_queries: int = 200,
) -> Dataset:
    """Download (if needed) and return the MS MARCO benchmark dataset.

    Parameters
    ----------
    data_dir:
        Directory for cached files.  Defaults to
        ``<repo_root>/benchmarks/data``.
    max_passages:
        How many passages to load from the collection.  Passages are taken
        from the top of the file (deterministic order).  Default 10 000.
    max_queries:
        Maximum number of dev queries to include.  Only queries with at
        least one relevant passage inside the loaded subset are kept.
        Default 200.

    Returns
    -------
    Dataset
        ``passages``, ``queries``, and ``qrels`` ready for benchmarking.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # --- passages -----------------------------------------------------------
    collection_path = _ensure_collection(data_dir, max_passages)
    passages = _parse_collection(collection_path, max_passages)
    passage_ids = {p.pid for p in passages}
    print(f"  Loaded {len(passages):,} passages", flush=True)

    # --- qrels --------------------------------------------------------------
    qrels_path = _ensure_qrels(data_dir)
    all_qrels = _parse_qrels(qrels_path)

    # Keep only qrels whose relevant pids exist in our passage subset
    qrels: Qrels = {}
    for qid, pid_map in all_qrels.items():
        filtered = {pid: rel for pid, rel in pid_map.items() if pid in passage_ids}
        if filtered:
            qrels[qid] = filtered

    # --- queries ------------------------------------------------------------
    queries_path = _ensure_queries(data_dir)
    all_queries = _parse_queries(queries_path)

    # Only keep queries that have at least one judgement in our qrels subset
    eligible_qids = set(qrels.keys())
    queries: list[Query] = []
    for qid, text in all_queries.items():
        if qid in eligible_qids:
            queries.append(Query(qid=qid, text=text))
        if len(queries) >= max_queries:
            break

    print(
        f"  Loaded {len(queries)} queries with relevance judgements "
        f"inside the passage subset",
        flush=True,
    )

    return Dataset(passages=passages, queries=queries, qrels=qrels)
