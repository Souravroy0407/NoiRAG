"""
Evaluate Statistical Cleaner on Semantic Noise.
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from noirag.preprocessing.statistical.spell_cleaner import StatisticalCleaner, clean_document_pages
from baseline.run_baseline import run_experiment, print_comparison_table, load_qa_pairs

def main():
    noisy_dir = PROJECT_ROOT / "data" / "noisy" / "semantic_25"
    cleaned_dir = PROJECT_ROOT / "data" / "cleaned" / "statistical" / "semantic_25"
    
    # 1. Clean the noisy data
    print("1. Cleaning noisy data (semantic_25) with Statistical Cleaner...")
    cleaner = StatisticalCleaner()
    
    for domain_dir in noisy_dir.iterdir():
        if not domain_dir.is_dir(): continue
        out_domain = cleaned_dir / domain_dir.name
        out_domain.mkdir(parents=True, exist_ok=True)
        
        for json_file in domain_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                pages = json.load(f)
            
            cleaned_pages = clean_document_pages(pages, cleaner)
            
            with open(out_domain / json_file.name, 'w', encoding='utf-8') as f:
                json.dump(cleaned_pages, f, indent=2, ensure_ascii=False)
                
    print(f"   Saved cleaned data to {cleaned_dir}")
    
    # 2. Run retrieval evaluation
    print("\n2. Running retrieval pipeline on cleaned data...")
    # Using testing set to compare
    qa_pairs = load_qa_pairs()
    
    # Run experiment on Cleaned Data
    cleaned_result = run_experiment("statistical_semantic_25", cleaned_dir, qa_pairs)
    
    # Run experiment on Noisy Data for comparison
    noisy_result = run_experiment("noisy_semantic_25", noisy_dir, qa_pairs)
    
    # Run experiment on Ground Truth for comparison
    gt_result = run_experiment("gt", PROJECT_ROOT / "data" / "ground_truth" / "gt", qa_pairs)
    
    print("\n3. Comparing Baseline vs Noisy vs Cleaned:")
    print_comparison_table([gt_result, noisy_result, cleaned_result])

if __name__ == "__main__":
    main()
