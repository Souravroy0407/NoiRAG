"""
evaluation/generation_eval.py

Generation evaluation metrics for NoiRAG baseline experiments.
Computes: BLEU, ROUGE-1, ROUGE-2, ROUGE-L.

Compares the generated answer against the gold (expected) answer.
"""

from typing import List, Dict, Any


# ── BLEU ──────────────────────────────────────────────────────────────────────

def _get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    """Extract n-gram counts from a list of tokens."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def bleu_score(
    generated: str,
    reference: str,
    max_n: int = 4,
) -> float:
    """
    Compute BLEU score (simple implementation without brevity penalty smoothing).
    Uses uniform weights across n-gram orders.
    """
    import math

    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(gen_tokens))) if len(gen_tokens) > 0 else 0.0

    # Precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        gen_ngrams = _get_ngrams(gen_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not gen_ngrams:
            precisions.append(0.0)
            continue

        # Clipped counts
        clipped = 0
        total = 0
        for gram, count in gen_ngrams.items():
            clipped += min(count, ref_ngrams.get(gram, 0))
            total += count

        precisions.append(clipped / total if total > 0 else 0.0)

    # Geometric mean of precisions (with smoothing for zero)
    log_avg = 0.0
    weight = 1.0 / max_n
    for p in precisions:
        if p == 0:
            return 0.0  # any zero n-gram precision → BLEU = 0
        log_avg += weight * math.log(p)

    return bp * math.exp(log_avg)


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_n(generated: str, reference: str, n: int = 1) -> Dict[str, float]:
    """
    Compute ROUGE-N (unigram or bigram overlap).
    Returns precision, recall, and F1.
    """
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    gen_ngrams = _get_ngrams(gen_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)

    if not gen_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Overlap count
    overlap = 0
    for gram, count in ref_ngrams.items():
        overlap += min(count, gen_ngrams.get(gram, 0))

    total_gen = sum(gen_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    precision = overlap / total_gen if total_gen > 0 else 0.0
    recall = overlap / total_ref if total_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(generated: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L using Longest Common Subsequence.
    Returns precision, recall, and F1.
    """
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(gen_tokens, ref_tokens)

    precision = lcs / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ── Aggregate ─────────────────────────────────────────────────────────────────

def evaluate_generation(
    all_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate generation across all QA pairs.

    Args:
        all_results: list of dicts, each containing:
            - "generated_answer": str (LLM output)
            - "gold_answer": str (expected answer)

    Returns:
        dict with averaged metrics:
        {
            "BLEU": 0.34,
            "ROUGE-1": 0.51,
            "ROUGE-2": 0.28,
            "ROUGE-L": 0.45,
        }
    """
    n = len(all_results)
    if n == 0:
        return {}

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []

    for item in all_results:
        gen = item["generated_answer"]
        gold = item["gold_answer"]

        bleu_scores.append(bleu_score(gen, gold))
        rouge1_scores.append(rouge_n(gen, gold, n=1)["f1"])
        rouge2_scores.append(rouge_n(gen, gold, n=2)["f1"])
        rougel_scores.append(rouge_l(gen, gold)["f1"])

    return {
        "BLEU":    sum(bleu_scores) / n,
        "ROUGE-1": sum(rouge1_scores) / n,
        "ROUGE-2": sum(rouge2_scores) / n,
        "ROUGE-L": sum(rougel_scores) / n,
    }
