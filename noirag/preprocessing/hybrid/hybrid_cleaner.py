"""
Hybrid Cleaner for NoiRAG.
Orchestrates text cleaning by routing text blocks through Rule-Based and Statistical
cleaners based on the heuristic scores from the Quality Scorer.
"""
import argparse
import sys
import json
from typing import List, Dict, Any, Tuple

from noirag.preprocessing.hybrid.quality_scorer import QualityScorer
from noirag.preprocessing.rule_based.cleaner import RuleBasedCleaner
from noirag.preprocessing.statistical.spell_cleaner import StatisticalCleaner

class HybridCleaner:
    def __init__(
        self, 
        formatting_threshold: float = 0.05,
        semantic_threshold: float = 0.10,
        verbose: bool = False
    ):
        """
        Args:
            formatting_threshold: Score above which Rule-Based cleaner is applied.
            semantic_threshold:  OOV ratio above which Statistical cleaner is applied.
            verbose: If True, prints routing decisions during execution.
        """
        self.formatting_threshold = formatting_threshold
        self.semantic_threshold = semantic_threshold
        self.verbose = verbose
        
        self.scorer = QualityScorer()
        self.rule_cleaner = RuleBasedCleaner()
        self.stat_cleaner = StatisticalCleaner()
        
    def clean(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluates and cleans the text.
        Returns the cleaned text and the routing metadata (scores and applied cleaners).
        """
        if not text:
            return text, {}
            
        scores = self.scorer.score(text)
        
        # Determine Routing
        apply_rule = (scores["garbage_density"] > self.formatting_threshold or 
                      scores["formatting_anomaly_rate"] > self.formatting_threshold)
                      
        apply_stat = scores["oov_ratio"] > self.semantic_threshold
        
        cleaned_text = text
        applied_cleaners = []
        
        # 1. Rule-based First (Fixing line breaks helps the spell-checker)
        if apply_rule:
            cleaned_text = self.rule_cleaner.clean(cleaned_text)
            applied_cleaners.append("rule_based")
            
        # 2. Statistical Spell-Checker Second
        if apply_stat:
            cleaned_text = self.stat_cleaner.clean(cleaned_text)
            applied_cleaners.append("statistical")
            
        if self.verbose:
            print(f"Scores: OOV={scores['oov_ratio']:.3f}, Garbage={scores['garbage_density']:.3f}, "
                  f"Anomaly={scores['formatting_anomaly_rate']:.3f} | Routing: {applied_cleaners}")
            
        metadata = {
            "original_scores": scores,
            "applied_cleaners": applied_cleaners
        }
        
        return cleaned_text, metadata

def clean_document_pages(pages: List[Dict[str, Any]], cleaner: HybridCleaner) -> List[Dict[str, Any]]:
    """Cleans the 'text' field of all pages in a document and attaches routing metadata."""
    for page in pages:
        if "text" in page and page["text"]:
            cleaned_text, metadata = cleaner.clean(page["text"])
            page["text"] = cleaned_text
            page["cleaning_metadata"] = metadata
    return pages

def main():
    parser = argparse.ArgumentParser(description="Clean text dynamically via the Hybrid Cleaner routing orchestrator.")
    parser.add_argument("--input", type=str, help="Input raw text file")
    parser.add_argument("--json", type=str, help="Input JSON file with list of page dicts")
    parser.add_argument("--output", type=str, default="stdout", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Print routing decisions")
    
    args = parser.parse_args()
    cleaner = HybridCleaner(verbose=args.verbose)
    
    if args.json:
        with open(args.json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data = clean_document_pages(data, cleaner)
        
        if args.output == "stdout":
            print(json.dumps(data, indent=2))
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned, metadata = cleaner.clean(text)
        if args.output == "stdout":
            if args.verbose:
                print(f"--- Metadata ---\n{json.dumps(metadata, indent=2)}\n--- Cleaned Text ---")
            print(cleaned)
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(cleaned)
    else:
        # Read from stdin
        text = sys.stdin.read()
        cleaned, metadata = cleaner.clean(text)
        if args.output == "stdout":
            if args.verbose:
                print(f"--- Metadata ---\n{json.dumps(metadata, indent=2)}\n--- Cleaned Text ---")
            print(cleaned)
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(cleaned)

if __name__ == "__main__":
    main()
