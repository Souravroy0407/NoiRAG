"""
evaluation/retrieval_eval.py

Retrieval evaluation metrics for NoiRAG baseline experiments.
Computes: Precision@k, Recall@k, F1@k, MRR, NDCG@k.

Relevance: a retrieved chunk is relevant if its doc_name
matches the QA pair's doc_name (document-level matching).
"""

import math
from typing import List, Dict, Any


def is_relevant(retrieved_doc_name: str, gold_doc_name: str) -> bool:
    """
    Check if a retrieved chunk is relevant to the QA pair.
    Matching logic: doc_name from retrieved chunk must match
    the doc_name from the QA pair (strip domain prefix if present).

    Examples:
        gold = "finance/3M_2023Q2_10Q"
        retrieved = "3M_2023Q2_10Q"  → True (stem matches)
        retrieved = "AES_2022_10K"   → False
    """
    # Strip domain prefix if present (e.g. "finance/3M_2023Q2_10Q" → "3M_2023Q2_10Q")
    gold_stem = gold_doc_name.split("/")[-1] if "/" in gold_doc_name else gold_doc_name
    ret_stem = retrieved_doc_name.split("/")[-1] if "/" in retrieved_doc_name else retrieved_doc_name
    return gold_stem == ret_stem


# ── Core Metrics ──────────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[Dict[str, Any]], gold_doc_name: str, k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if is_relevant(r["doc_name"], gold_doc_name))
    return relevant / len(top_k)


def recall_at_k(
    retrieved: List[Dict[str, Any]],
    gold_doc_name: str,
    k: int,
    total_relevant: int = 1,
) -> float:
    """
    Fraction of relevant docs found in top-k.
    For single-doc QA (our case), total_relevant = 1.
    """
    top_k = retrieved[:k]
    if total_relevant == 0:
        return 0.0
    relevant = sum(1 for r in top_k if is_relevant(r["doc_name"], gold_doc_name))
    return min(relevant, total_relevant) / total_relevant


def f1_at_k(retrieved: List[Dict[str, Any]], gold_doc_name: str, k: int) -> float:
    """Harmonic mean of Precision@k and Recall@k."""
    p = precision_at_k(retrieved, gold_doc_name, k)
    r = recall_at_k(retrieved, gold_doc_name, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: List[Dict[str, Any]], gold_doc_name: str) -> float:
    """
    Mean Reciprocal Rank — inverse of the rank of first relevant result.
    Returns 0 if no relevant result found.
    """
    for i, r in enumerate(retrieved):
        if is_relevant(r["doc_name"], gold_doc_name):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[Dict[str, Any]], gold_doc_name: str, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    Binary relevance: 1 if doc matches, 0 otherwise.
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    num_relevant = 0
    for i, r in enumerate(top_k):
        rel = 1.0 if is_relevant(r["doc_name"], gold_doc_name) else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank is 1-indexed
        num_relevant += int(rel)

    # Ideal DCG: all relevant results packed at top ranks
    idcg = 0.0
    for i in range(min(num_relevant, k)):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        # No relevant results exist → NDCG undefined, return 0
        return 0.0

    return min(dcg / idcg, 1.0)


# ── Aggregate ─────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    all_results: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Evaluate retrieval across all QA pairs.

    Args:
        all_results: list of dicts, each containing:
            - "retrieved": list of retriever results (with "doc_name")
            - "gold_doc_name": the expected doc name from QA pair

        k_values: list of k values to compute metrics at

    Returns:
        dict with averaged metrics, e.g.:
        {
            "P@1": 0.72, "P@3": 0.65, "P@5": 0.58,
            "R@1": 0.72, "R@3": 0.85, "R@5": 0.91,
            "F1@1": ..., "F1@3": ..., "F1@5": ...,
            "MRR": 0.81, "NDCG@1": ..., "NDCG@3": ..., "NDCG@5": ...,
        }
    """
    n = len(all_results)
    if n == 0:
        return {}

    metrics = {}

    for k in k_values:
        p_scores = []
        r_scores = []
        f1_scores = []
        ndcg_scores = []

        for item in all_results:
            retrieved = item["retrieved"]
            gold = item["gold_doc_name"]

            p_scores.append(precision_at_k(retrieved, gold, k))
            r_scores.append(recall_at_k(retrieved, gold, k))
            f1_scores.append(f1_at_k(retrieved, gold, k))
            ndcg_scores.append(ndcg_at_k(retrieved, gold, k))

        metrics[f"P@{k}"] = sum(p_scores) / n
        metrics[f"R@{k}"] = sum(r_scores) / n
        metrics[f"F1@{k}"] = sum(f1_scores) / n
        metrics[f"NDCG@{k}"] = sum(ndcg_scores) / n

    # MRR (not k-dependent — uses full result list)
    mrr_scores = [mrr(item["retrieved"], item["gold_doc_name"]) for item in all_results]
    metrics["MRR"] = sum(mrr_scores) / n

    return metrics
