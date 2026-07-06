"""
Hybrid Cleaner for NoiRAG.
Orchestrates text cleaning by routing text blocks through Rule-Based and Statistical
cleaners based on the heuristic scores from the Quality Scorer.
"""
import argparse
import sys
import time
import json
from typing import List, Dict, Any, Tuple

from noirag.preprocessing.hybrid.quality_scorer import QualityScorer
from noirag.preprocessing.rule_based.cleaner import RuleBasedCleaner
from noirag.preprocessing.statistical.spell_cleaner import StatisticalCleaner
from noirag.preprocessing.hybrid.llm_cleaner import LLMCleaner
from noirag.preprocessing.hybrid.cost_profiler import CostProfiler

class HybridCleaner:
    def __init__(
        self, 
        formatting_threshold: float = 0.05,
        semantic_threshold: float = 0.15,
        llm_threshold: float = 0.75,
        llm_backend: str = "groq",
        api_key: str = None,
        verbose: bool = False
    ):
        """
        Args:
            formatting_threshold: Score above which Rule-Based cleaner is applied.
            semantic_threshold: OOV ratio above which Statistical cleaner is applied.
            llm_threshold: Overall score above which the heavy LLM cleaner is executed.
            llm_backend: Backend to use for LLMCleaner ("groq", "ollama", or "openai")
            api_key: Optional API key for LLMCleaner
            verbose: If True, prints routing decisions during execution.
        """
        self.formatting_threshold = formatting_threshold
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold
        self.verbose = verbose
        
        self.scorer = QualityScorer()
        self.rule_cleaner = RuleBasedCleaner()
        self.stat_cleaner = StatisticalCleaner()
        self.llm_cleaner = LLMCleaner(backend=llm_backend, api_key=api_key)
        self.profiler = CostProfiler()
        
    def clean(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluates and cleans the text.
        Returns the cleaned text and the routing metadata (scores and applied cleaners).
        """
        if not text:
            return text, {}
        
        t0 = time.perf_counter()
        scores = self.scorer.score(text)
        
        # Determine Routing
        apply_llm = scores["overall_score"] > self.llm_threshold
        apply_rule = (scores["garbage_density"] > self.formatting_threshold or 
                      scores["formatting_anomaly_rate"] > self.formatting_threshold)
        apply_stat = scores["oov_ratio"] > self.semantic_threshold
        
        cleaned_text = text
        applied_cleaners = []
        
        # 0. Catastrophic Noise -> Try LLM first, fall back to Rule+Stat if it fails
        if apply_llm:
            llm_result = self.llm_cleaner.clean(cleaned_text)
            # If LLM succeeded (returned different text), use it
            if llm_result != cleaned_text:
                cleaned_text = llm_result
                applied_cleaners.append("llm")
            else:
                # LLM failed (rate limit / error) — fall back to local cleaners
                cleaned_text = self.rule_cleaner.clean(cleaned_text)
                cleaned_text = self.stat_cleaner.clean(cleaned_text)
                applied_cleaners.append("rule+stat (llm_fallback)")
        else:
            # 1. Rule-based First (Fixing line breaks helps the spell-checker)
            if apply_rule:
                cleaned_text = self.rule_cleaner.clean(cleaned_text)
                applied_cleaners.append("rule_based")
                
            # 2. Statistical Spell-Checker Second
            if apply_stat:
                cleaned_text = self.stat_cleaner.clean(cleaned_text)
                applied_cleaners.append("statistical")
        
        # ── Content Preservation Guard ──────────────────────────────────
        # If cleaning removed too much content (>15% of text lost),
        # the original noisy text is better than a gutted cleaned version.
        original_len = len(text.strip())
        cleaned_len = len(cleaned_text.strip())
        if original_len > 0 and cleaned_len / original_len < 0.85:
            cleaned_text = text  # revert to original
            applied_cleaners = ["bypassed (content_guard)"]
        # ────────────────────────────────────────────────────────────────
            
        if self.verbose:
            print(f"Scores: Overall={scores['overall_score']:.3f}, OOV={scores['oov_ratio']:.3f}, "
                  f"Garbage={scores['garbage_density']:.3f}"
                  f" | Routing: {applied_cleaners}")
            
        elapsed = time.perf_counter() - t0
        self.profiler.record(text, applied_cleaners, elapsed)
        
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
