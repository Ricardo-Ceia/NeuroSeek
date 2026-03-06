"""
benchmarks/metrics.py
─────────────────────
Information-retrieval evaluation metrics used by the NeuroSeek benchmark.

All functions are pure — no I/O, no side effects.  Each takes:

  retrieved : list[str]
      Ranked list of passage IDs returned by a retrieval system, best
      match first.

  relevant  : dict[str, int]
      Ground-truth relevance map for the query: pid -> relevance score.
      For MS MARCO dev, scores are always 1 (binary relevance).

Public API
----------
  recall_at_k(retrieved, relevant, k)  -> float  in [0, 1]
  mrr(retrieved, relevant, k)          -> float  in [0, 1]
  ndcg_at_k(retrieved, relevant, k)    -> float  in [0, 1]
  evaluate(retrieved, relevant, k)     -> dict[str, float]

References
----------
  Recall@K  : fraction of relevant docs found in the top-K results.
  MRR       : 1/rank of the first relevant result (capped at K).
  NDCG@K    : normalised discounted cumulative gain at K, using
              log2(rank+1) discounting.  Handles graded relevance.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of relevant passages that appear in the top-*k* results.

    Parameters
    ----------
    retrieved:
        Ranked list of passage IDs (best first).  Only the first *k* are
        considered.
    relevant:
        Ground-truth map of pid -> relevance score (> 0 means relevant).
    k:
        Cut-off depth.

    Returns
    -------
    float
        0.0 if there are no relevant passages; otherwise the fraction of
        relevant passages found in ``retrieved[:k]``.
    """
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0

    top_k = set(retrieved[:k])
    hits = sum(1 for pid in relevant if pid in top_k)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Mean Reciprocal Rank — reciprocal of the first relevant result's rank.

    Parameters
    ----------
    retrieved:
        Ranked list of passage IDs (best first).
    relevant:
        Ground-truth map of pid -> relevance score.
    k:
        Maximum rank to consider.  Results beyond rank *k* are ignored.

    Returns
    -------
    float
        1/rank of the first hit, or 0.0 if no relevant passage appears in
        ``retrieved[:k]``.
    """
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0

    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in relevant and relevant[pid] > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank *k*.

    Uses the standard log2(rank + 1) discount.  Supports graded relevance
    (e.g. relevance scores of 0, 1, 2, 3).

    Parameters
    ----------
    retrieved:
        Ranked list of passage IDs (best first).
    relevant:
        Ground-truth map of pid -> relevance score (non-negative integers).
    k:
        Cut-off depth.

    Returns
    -------
    float
        nDCG@k in [0, 1].  Returns 0.0 if there are no relevant passages.
    """
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0

    # Actual DCG — sum over the top-k retrieved results
    dcg = 0.0
    for rank, pid in enumerate(retrieved[:k], start=1):
        rel = relevant.get(pid, 0)
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    # Ideal DCG — top-k sorted by descending relevance
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, start=1)
        if rel > 0
    )

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate(
    retrieved: list[str],
    relevant: dict[str, int],
    k: int = 10,
) -> dict[str, float]:
    """Compute all three metrics for a single query.

    Parameters
    ----------
    retrieved:
        Ranked list of passage IDs (best first).
    relevant:
        Ground-truth map of pid -> relevance score.
    k:
        Cut-off depth applied to all metrics.

    Returns
    -------
    dict[str, float]
        Keys: ``"recall@k"``, ``"mrr"``, ``"ndcg@k"``.
    """
    return {
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


def mean_scores(results: list[dict[str, float]]) -> dict[str, float]:
    """Average per-query metric dicts into a single summary dict.

    Parameters
    ----------
    results:
        List of per-query dicts as returned by :func:`evaluate`.

    Returns
    -------
    dict[str, float]
        Macro-averaged scores.  Returns an empty dict if *results* is empty.
    """
    if not results:
        return {}
    keys = results[0].keys()
    return {
        key: sum(r[key] for r in results) / len(results)
        for key in keys
    }
