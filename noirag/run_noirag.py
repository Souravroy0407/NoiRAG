"""
noirag/run_noirag.py

End-to-End Experiment Runner for NoiseClear-RAG.

Runs the intelligently routed Hybrid Cleaner preprocessing pipeline on noisy text,
then runs the RAG retrieval pipeline to evaluate the performance improvement.

Usage:
    python -m noirag.run_noirag --noise-type formatting --noise-level 25
    python -m noirag.run_noirag --limit 5  # test with 5 QA pairs/domain
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Baseline execution utilities
from baseline.run_baseline import run_experiment, print_comparison_table, save_results, load_qa_pairs

# Import NoiRAG preprocessing module
from noirag.preprocessing.hybrid.hybrid_cleaner import HybridCleaner, clean_document_pages

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = PROJECT_ROOT / "data"
GT_DIR         = DATA_DIR / "ground_truth" / "gt"
NOISY_DIR      = DATA_DIR / "noisy"
CLEANED_DIR    = DATA_DIR / "cleaned" / "hybrid"
RESULTS_DIR    = PROJECT_ROOT / "results" / "tables"
# ─────────────────────────────────────────────────────────────────────────────

def apply_preprocessing(
    noise_name: str, 
    source_dir: Path, 
    output_dir: Path,
    cleaner: HybridCleaner
) -> None:
    """Cleans the noisy dataset and saves it to the output directory."""
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source noisy directory not found: {source_dir}")
        
    print(f"\n[1/4] Preprocessing Data: {noise_name}")
    print(f"      Source: {source_dir}")
    print(f"      Target: {output_dir}")
    
    total_docs = 0
    total_pages = 0
    
    for domain_dir in sorted(source_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
            
        out_domain = output_dir / domain_dir.name
        out_domain.mkdir(parents=True, exist_ok=True)
        
        json_files = sorted(domain_dir.glob("*.json"))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                pages = json.load(f)
                
            cleaned_pages = clean_document_pages(pages, cleaner)
            
            with open(out_domain / jf.name, "w", encoding="utf-8") as f:
                json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)
                
            total_docs += 1
            total_pages += len(pages)
            
        print(f"      Cleaned {domain_dir.name}: {len(json_files)} docs")

    print(f"      ✅ Preprocessing Complete: {total_docs} docs, {total_pages} pages")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full NoiRAG preprocessing and evaluation pipeline")
    parser.add_argument("--noise-type", choices=["semantic", "formatting"], required=True,
                        help="Specific noise type to test")
    parser.add_argument("--noise-level", type=int, choices=[10, 25, 50, 75], required=True,
                        help="Specific noise level percentage to test")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max QA pairs to test per domain (0 = all, faster testing)")
    parser.add_argument("--skip-cleaning", action="store_true",
                        help="Skip preprocessing phase if docs were already cleaned")
    args = parser.parse_args()

    print("============================================================")
    print(" NoiRAG End-to-End Evaluation Pipeline")
    print("============================================================")

    noise_name = f"{args.noise_type}_{args.noise_level}"
    noisy_dir = NOISY_DIR / noise_name
    cleaned_dir = CLEANED_DIR / noise_name
    
    # 1. Clean the Noisy Dataset
    if not args.skip_cleaning:
        cleaner = HybridCleaner(verbose=False)
        apply_preprocessing(noise_name, noisy_dir, cleaned_dir, cleaner)
    else:
        print(f"\n[1/4] Skipping Preprocessing. Using existing data at {cleaned_dir}")
        if not cleaned_dir.exists():
            print(f"⚠️ Warning: Cleaned directory does not exist: {cleaned_dir}")

    # 2. Load QA Pairs
    print("\n[2/4] Loading QA Evaluation Pairs...")
    qa_pairs = load_qa_pairs(args.limit)
    print(f"      Total: {len(qa_pairs)} QA pairs")

    all_experiments = []

    # 3. Clean Baseline (Ground Truth)
    print("\n[3/4] Evaluating Models...")
    gt_result = run_experiment("gt_clean_baseline", GT_DIR, qa_pairs)
    all_experiments.append(gt_result)
    
    # 4. Noisy Baseline (Uncleaned)
    noisy_result = run_experiment(f"noisy_{noise_name}", noisy_dir, qa_pairs)
    all_experiments.append(noisy_result)
    
    # 5. NoiRAG (Hybrid Cleaned)
    noirag_result = run_experiment(f"noirag_cleaned_{noise_name}", cleaned_dir, qa_pairs)
    all_experiments.append(noirag_result)

    # 6. Summary and Save
    print("\n[4/4] Final Results and Comparison:")
    print_comparison_table(all_experiments)
    
    output_meta = f"hybrid_evaluation_{noise_name}.json"
    save_results(all_experiments, filename=output_meta)

    print(f"\n🎉 NoiRAG Evaluation Complete for {noise_name}!")
