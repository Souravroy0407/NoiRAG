"""
baseline/run_baseline.py

Baseline experiment runner for NoiseClear-RAG.

Runs the RAG retrieval pipeline on clean ground-truth text AND noisy text,
then evaluates retrieval quality to establish:
  1. The performance ceiling (clean text)
  2. The performance degradation (noisy text at various levels)

Usage:
    python -m baseline.run_baseline                        # clean baseline only
    python -m baseline.run_baseline --noisy                # clean + all noisy
    python -m baseline.run_baseline --noisy --noise-type semantic --noise-level 25
    python -m baseline.run_baseline --limit 5              # test with 5 QA pairs/domain
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.chunker.text_splitter import chunk_directory, save_chunks
from pipeline.embedder.bge_small_embedder import load_chunks, embed_chunks, build_faiss_index, save_index
from pipeline.retriever.faiss_retriever import FAISSRetriever
from evaluation.retrieval_eval import evaluate_retrieval


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = PROJECT_ROOT / "data"
GT_DIR         = DATA_DIR / "ground_truth" / "gt"
NOISY_DIR      = DATA_DIR / "noisy"
CHUNKS_DIR     = DATA_DIR / "chunks"
INDEX_DIR      = DATA_DIR / "index"
QA_DIR         = DATA_DIR / "qa"
RESULTS_DIR    = PROJECT_ROOT / "results" / "tables"
K_VALUES       = [1, 3, 5]
TOP_K          = 5
# ─────────────────────────────────────────────────────────────────────────────


def load_qa_pairs(limit: int = 0) -> List[Dict[str, Any]]:
    """Load all QA pairs from data/qa/*.jsonl files."""
    qa_pairs = []
    for jsonl_file in sorted(QA_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            domain_pairs = [json.loads(line) for line in f if line.strip()]
        if limit > 0:
            domain_pairs = domain_pairs[:limit]
        qa_pairs.extend(domain_pairs)
        print(f"  Loaded {len(domain_pairs)} QA pairs from {jsonl_file.name}")
    return qa_pairs


def ensure_chunks_and_index(name: str, source_dir: Path) -> tuple:
    """
    Ensure chunks and FAISS index exist for a given data source.
    If they already exist, skip. Otherwise, build them.

    Returns: (chunks_path, index_path, metadata_path)
    """
    chunks_path = CHUNKS_DIR / f"{name}_chunks.json"
    index_path  = INDEX_DIR / f"{name}.faiss"
    meta_path   = INDEX_DIR / f"{name}_metadata.json"

    # Check if already built
    if chunks_path.exists() and index_path.exists() and meta_path.exists():
        print(f"  ✅ Index already exists for '{name}' — skipping build")
        return chunks_path, index_path, meta_path

    # Build chunks
    print(f"  📦 Chunking {source_dir}...")
    chunks = chunk_directory(source_dir)
    save_chunks(chunks, chunks_path)

    # Build embeddings + index
    print(f"  🧬 Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_index(index, chunks, INDEX_DIR, name)

    return chunks_path, index_path, meta_path


def run_retrieval(
    retriever: FAISSRetriever,
    qa_pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run retrieval for all QA pairs."""
    results = []
    for i, qa in enumerate(qa_pairs):
        retrieved = retriever.retrieve(qa["question"])
        results.append({
            "qa": qa,
            "retrieved": retrieved,
            "gold_doc_name": qa["doc_name"],
        })
        if (i + 1) % 100 == 0:
            print(f"    Retrieved {i + 1}/{len(qa_pairs)}")
    return results


def run_experiment(
    name: str,
    source_dir: Path,
    qa_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run a single experiment (clean or noisy).
    Returns dict with retrieval metrics.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Step 1: Ensure chunks + index
    print("\n[1/3] Preparing index...")
    chunks_path, index_path, meta_path = ensure_chunks_and_index(name, source_dir)

    # Step 2: Run retrieval
    print(f"\n[2/3] Running retrieval on {len(qa_pairs)} queries...")
    retriever = FAISSRetriever(str(index_path), str(meta_path), TOP_K)
    retrieval_results = run_retrieval(retriever, qa_pairs)

    # Step 3: Evaluate retrieval
    print(f"\n[3/3] Evaluating retrieval...")
    metrics = evaluate_retrieval(retrieval_results, K_VALUES)

    # Pretty print
    print(f"\n{'─'*40}")
    print(f"  Results: {name}")
    print(f"{'─'*40}")
    for metric, value in metrics.items():
        print(f"  {metric:>10}: {value:.4f}")

    return {
        "name": name,
        "metrics": metrics,
        "num_queries": len(qa_pairs),
    }


def save_results(all_experiments: List[Dict[str, Any]], filename: str = "baseline_results.json"):
    """Save all experiment results to a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved: {output_path}")


def print_comparison_table(all_experiments: List[Dict[str, Any]]):
    """Print a comparison table across all experiments."""
    if not all_experiments:
        return

    # Collect all metric names
    all_metric_names = []
    for exp in all_experiments:
        for k in exp["metrics"]:
            if k not in all_metric_names:
                all_metric_names.append(k)

    # Header
    name_width = max(len(exp["name"]) for exp in all_experiments)
    header = f"{'Condition':<{name_width}}"
    for m in all_metric_names:
        header += f" | {m:>8}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    # Rows
    for exp in all_experiments:
        row = f"{exp['name']:<{name_width}}"
        for m in all_metric_names:
            val = exp["metrics"].get(m, 0.0)
            row += f" | {val:>8.4f}"
        print(row)

    print(f"{'='*len(header)}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NoiseClear-RAG baseline experiments")
    parser.add_argument("--noisy", action="store_true",
                        help="Also run noisy experiments (default: clean only)")
    parser.add_argument("--noise-type", choices=["semantic", "formatting"],
                        help="Specific noise type (default: both)")
    parser.add_argument("--noise-level", type=int, choices=[10, 25, 50, 75],
                        help="Specific noise level (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max QA pairs per domain (0 = all)")
    args = parser.parse_args()

    print("NoiseClear-RAG — Baseline Experiment Runner")
    print(f"Project root: {PROJECT_ROOT}\n")

    # Load QA pairs
    print("Loading QA pairs...")
    qa_pairs = load_qa_pairs(args.limit)
    print(f"Total: {len(qa_pairs)} QA pairs\n")

    all_experiments = []

    # ── Experiment 1: Clean baseline ──────────────────────────────────────
    print("▸ Running CLEAN baseline...")
    clean_result = run_experiment("gt", GT_DIR, qa_pairs)
    all_experiments.append(clean_result)

    # ── Experiment 2+: Noisy baselines ────────────────────────────────────
    if args.noisy:
        types = [args.noise_type] if args.noise_type else ["semantic", "formatting"]
        levels = [args.noise_level] if args.noise_level else [10, 25, 50, 75]

        for ntype in types:
            for nlevel in levels:
                noisy_name = f"{ntype}_{nlevel}"
                noisy_dir = NOISY_DIR / noisy_name

                if not noisy_dir.exists() or not any(noisy_dir.iterdir()):
                    print(f"\n⚠️ Skipping {noisy_name} — no data in {noisy_dir}")
                    print(f"   Run: python -m baseline.noise_injector --type {ntype} --level {nlevel}")
                    continue

                result = run_experiment(noisy_name, noisy_dir, qa_pairs)
                all_experiments.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    print_comparison_table(all_experiments)
    save_results(all_experiments)

    print("\n🎉 Baseline experiments complete!")
