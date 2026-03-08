"""
benchmarks/run.py
─────────────────
Entry point for the NeuroSeek benchmark suite.

Runs all six retrieval systems against a MS MARCO passage subset and prints
a results table covering speed (index build time, query latency p50/p95,
throughput) accuracy (Recall@10, MRR, NDCG@10) and memory.

Usage
-----
    # from the repo root
    python -m benchmarks.run

    # optional flags
    python -m benchmarks.run --passages 10000 --queries 200 --top-k 10
    python -m benchmarks.run --runners neuroseek hnswlib faiss  # subset
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Type

import numpy as np

from neuroseek.embedder import Embedder

from benchmarks.dataset import load, Dataset
from benchmarks.metrics import evaluate, mean_scores
from benchmarks.runners.base import BaseRunner
from benchmarks.runners.neuroseek_runner import NeuroSeekRunner
from benchmarks.runners.hnswlib_runner import HNSWLibRunner
from benchmarks.runners.faiss_runner import FAISSRunner
from benchmarks.runners.bm25_runner import BM25Runner
from benchmarks.runners.whoosh_runner import WhooshRunner
from benchmarks.runners.chroma_runner import ChromaRunner

# All runners that accept an embedder (vector-based)
_VECTOR_RUNNERS: list[Type[BaseRunner]] = [
    NeuroSeekRunner,
    HNSWLibRunner,
    FAISSRunner,
    ChromaRunner,
]

# All runners that need no embedder (keyword-based)
_KEYWORD_RUNNERS: list[Type[BaseRunner]] = [
    BM25Runner,
    WhooshRunner,
]

_ALL_RUNNER_NAMES = ["neuroseek", "hnswlib", "faiss", "chromadb", "bm25", "whoosh"]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_queries(
    runner: BaseRunner,
    queries: list[tuple[str, str]],
    top_k: int,
) -> tuple[list[str], list[float]]:
    """Run all queries, return (list_of_pid_lists, list_of_latencies_seconds).

    Returns a parallel list of results (one per query) and latencies.
    """
    all_results: list[list[str]] = []
    latencies: list[float] = []
    for _qid, text in queries:
        t0 = time.perf_counter()
        pids = runner.query(text, top_k=top_k)
        latencies.append(time.perf_counter() - t0)
        all_results.append(pids)
    return all_results, latencies


# ---------------------------------------------------------------------------
# Per-runner benchmark
# ---------------------------------------------------------------------------

def _run_one(
    runner: BaseRunner,
    passages: list[tuple[str, str]],
    dataset: Dataset,
    top_k: int,
    vectors: "np.ndarray | None" = None,
) -> dict:
    """Benchmark a single runner. Returns a result dict."""
    print(f"  [{runner.name}] building index ...", flush=True)
    t0 = time.perf_counter()
    if vectors is not None:
        runner.build_index(passages, vectors)  # type: ignore[call-arg]
    else:
        runner.build_index(passages)
    build_time = time.perf_counter() - t0
    mem_mb = runner.memory_usage_mb()

    print(f"  [{runner.name}] running {len(dataset.queries)} queries ...", flush=True)
    query_pairs = [(q.qid, q.text) for q in dataset.queries]
    all_results, latencies = _time_queries(runner, query_pairs, top_k)

    # Accuracy — evaluate each query that has qrels
    per_query_metrics = []
    for (qid, _), retrieved in zip(query_pairs, all_results):
        relevant = dataset.qrels.get(qid, {})
        if relevant:
            per_query_metrics.append(evaluate(retrieved, relevant, k=top_k))

    avg = mean_scores(per_query_metrics) if per_query_metrics else {}

    # Latency stats (ms)
    lats_ms = [l * 1000 for l in latencies]
    p50 = statistics.median(lats_ms)
    p95 = sorted(lats_ms)[int(len(lats_ms) * 0.95)] if lats_ms else 0.0
    qps  = len(latencies) / sum(latencies) if latencies else 0.0

    return {
        "name":        runner.name,
        "build_s":     build_time,
        "p50_ms":      p50,
        "p95_ms":      p95,
        "qps":         qps,
        "mem_mb":      mem_mb,
        f"recall@{top_k}": avg.get(f"recall@{top_k}", float("nan")),
        "mrr":         avg.get("mrr",          float("nan")),
        f"ndcg@{top_k}":   avg.get(f"ndcg@{top_k}",   float("nan")),
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], top_k: int) -> None:
    r_col  = f"R@{top_k}"
    nd_col = f"nDCG@{top_k}"

    headers = [
        "System", "Build(s)", "p50(ms)", "p95(ms)", "QPS",
        "RAM(MB)", r_col, "MRR", nd_col,
    ]
    col_w = [12, 9, 8, 8, 7, 8, 7, 7, 8]

    def fmt_row(vals: list) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    sep = "  ".join("-" * w for w in col_w)

    print()
    print(fmt_row(headers))
    print(sep)
    for r in results:
        def _f(v, decimals=2):
            return f"{v:.{decimals}f}" if isinstance(v, float) and v == v else "n/a"

        row = [
            r["name"],
            _f(r["build_s"]),
            _f(r["p50_ms"]),
            _f(r["p95_ms"]),
            _f(r["qps"], 1),
            _f(r["mem_mb"], 0),
            _f(r[f"recall@{top_k}"]),
            _f(r["mrr"]),
            _f(r[f"ndcg@{top_k}"]),
        ]
        print(fmt_row(row))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroSeek benchmark suite")
    parser.add_argument("--passages", type=int, default=10_000,
                        help="Number of MS MARCO passages to index (default 10000)")
    parser.add_argument("--queries",  type=int, default=200,
                        help="Max dev queries to evaluate (default 200)")
    parser.add_argument("--top-k",    type=int, default=10,
                        help="Retrieval depth for all metrics (default 10)")
    parser.add_argument("--runners",  nargs="+", choices=_ALL_RUNNER_NAMES,
                        default=_ALL_RUNNER_NAMES,
                        help="Which runners to include (default: all)")
    args = parser.parse_args()

    print(f"\nNeuroSeek Benchmark")
    print(f"  passages : {args.passages:,}")
    print(f"  queries  : {args.queries}")
    print(f"  top-k    : {args.top_k}")
    print(f"  runners  : {', '.join(args.runners)}")
    print()

    # --- dataset ------------------------------------------------------------
    print("Loading dataset ...")
    dataset = load(max_passages=args.passages, max_queries=args.queries)
    passages = [(p.pid, p.text) for p in dataset.passages]
    print(f"  {dataset}\n")

    # --- shared embedder (loaded once) --------------------------------------
    _VECTOR_RUNNER_NAMES = {"neuroseek", "hnswlib", "faiss", "chromadb"}
    needs_embedder = any(n in args.runners for n in _VECTOR_RUNNER_NAMES)
    embedder: Embedder | None = None
    passage_vectors: np.ndarray | None = None

    if needs_embedder:
        print("Loading embedding model ...", flush=True)
        embedder = Embedder()
        print(f"  model={embedder.model_name}  dim={embedder.dimension}")

        # Pre-compute passage embeddings once — shared across all vector runners
        print(f"  Pre-computing embeddings for {len(passages):,} passages ...", flush=True)
        t_emb = time.perf_counter()
        texts = [text for _, text in passages]
        vecs = embedder.encode_batch(texts)
        passage_vectors = np.array([v.data for v in vecs], dtype=np.float32)
        print(f"  done in {time.perf_counter() - t_emb:.1f}s\n")

    # --- build runners ------------------------------------------------------
    runners: list[BaseRunner] = []
    name_map = {
        "neuroseek": lambda: NeuroSeekRunner(embedder),
        "hnswlib":   lambda: HNSWLibRunner(embedder),
        "faiss":     lambda: FAISSRunner(embedder),
        "chromadb":  lambda: ChromaRunner(embedder),
        "bm25":      lambda: BM25Runner(),
        "whoosh":    lambda: WhooshRunner(),
    }
    for name in _ALL_RUNNER_NAMES:
        if name in args.runners:
            runners.append(name_map[name]())

    # --- run ----------------------------------------------------------------
    results = []
    for runner in runners:
        print(f"Benchmarking {runner.name} ...", flush=True)
        # Pass pre-computed vectors to vector runners to avoid re-embedding.
        # runner.name values: "NeuroSeek", "hnswlib", "FAISS", "ChromaDB".
        # args.runners uses lowercase, so _VECTOR_RUNNER_NAMES is also lowercase —
        # but runner.name is the canonical display name set on each runner class.
        _VECTOR_RUNNER_DISPLAY_NAMES = {"NeuroSeek", "hnswlib", "FAISS", "ChromaDB"}
        use_vectors = passage_vectors if runner.name in _VECTOR_RUNNER_DISPLAY_NAMES else None
        try:
            result = _run_one(runner, passages, dataset, args.top_k, vectors=use_vectors)
            results.append(result)
            print(
                f"  done  build={result['build_s']:.2f}s  "
                f"p50={result['p50_ms']:.1f}ms  "
                f"R@{args.top_k}={result[f'recall@{args.top_k}']:.2f}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    # --- print table --------------------------------------------------------
    _print_table(results, args.top_k)


if __name__ == "__main__":
    main()
