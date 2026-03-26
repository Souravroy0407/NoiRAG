"""
Quality Scorer for NoiRAG.
Calculates a 'noisiness' score for text chunks to guide the Hybrid Cleaner routing strategy.
"""
import re
import importlib.resources
from typing import Dict

try:
    from symspellpy import SymSpell, Verbosity
except ImportError:
    SymSpell = None

class QualityScorer:
    def __init__(self):
        # We reuse the exact same garbage strings the RuleBasedCleaner looks for
        self.garbage_strings = [
            "•", "■", "▪", "▸", "◆", " | ", " — ", "...", "***",
            "  —  ", ">> ", "<< ", "## ", "[...]", "(?)", "{~}", "//", "---", "==="
        ]
        
        # Load symspell dictionary for Out-Of-Vocabulary (OOV) checking
        self.sym_spell = None
        if SymSpell:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dictionary_path = str(importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt")
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def _should_check_vocab(self, token: str) -> bool:
        """Determines if a token is valid for OOV checking."""
        clean_token = token.strip('.,!?;:\'\"()[]{}')
        
        # Ignored: Very short, Contains numbers, All uppercase (Acronyms)
        if len(clean_token) <= 3 or any(char.isdigit() for char in clean_token) or clean_token.isupper():
            return False
            
        return True

    def _calculate_oov_ratio(self, text: str) -> float:
        """Returns the ratio of out-of-vocabulary words (semantic noise heuristic)."""
        if not self.sym_spell:
            return 0.0
            
        words = re.split(r'\s+', text)
        valid_words_count = 0
        oov_words_count = 0
        
        for w in words:
            if not w.strip():
                continue
            
            clean_word = w.strip('.,!?;:\'\"()[]{}')
            if self._should_check_vocab(clean_word):
                valid_words_count += 1
                
                # Check if it exists exactly in the dictionary
                # .lookup with edit dist 0 acts as a vocabulary check
                suggestions = self.sym_spell.lookup(clean_word.lower(), Verbosity.TOP, max_edit_distance=0)
                if not suggestions:
                    oov_words_count += 1
                    
        if valid_words_count == 0:
            return 0.0
            
        return oov_words_count / valid_words_count

    def _calculate_garbage_density(self, text: str) -> float:
        """Returns the relative density of specific OCR garbage strings (formatting noise heuristic)."""
        if not text:
            return 0.0
            
        garbage_count = sum(text.count(g) for g in self.garbage_strings)
        # Normalize arbitrarily: length 100 with 1 garbage string = 1% density.
        # Cap at 1.0
        density = (garbage_count * 10) / len(text)
        return min(density, 1.0)
        
    def _calculate_formatting_anomaly_rate(self, text: str) -> float:
        """Returns heuristic score based on strange whitespace and very short broken lines."""
        if not text:
            return 0.0
            
        score = 0.0
        
        # Penalty for multiple spaces
        multiple_spaces = len(re.findall(r' {2,}', text))
        score += multiple_spaces * 0.05
        
        # Penalty for arbitrary broken lines mid-sentence (line doesn't end in punctuation)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        broken_lines = 0
        for i in range(len(lines) - 1):
            if lines[i] and lines[i][-1] not in '.!?:;"\'':
                if lines[i+1] and lines[i+1][0].islower():
                    broken_lines += 1
                    
        score += broken_lines * 0.1
        
        return min(score, 1.0)

    def score(self, text: str) -> Dict[str, float]:
        """
        Calculates the noisiness of the text.
        Returns a dict of sub-scores and the combined normalized score (0.0=clean, 1.0=noisy).
        """
        if not text.strip():
            return {
                "overall_score": 0.0,
                "oov_ratio": 0.0,
                "garbage_density": 0.0,
                "formatting_anomaly_rate": 0.0
            }

        oov_ratio = self._calculate_oov_ratio(text)
        garbage_density = self._calculate_garbage_density(text)
        anomaly_rate = self._calculate_formatting_anomaly_rate(text)
        
        # Weighted combination 
        # Semantic errors (typos) are heavily weighted because they break embedding tokens
        # Garbage strings are easily cleaned
        overall = (oov_ratio * 0.5) + (garbage_density * 0.2) + (anomaly_rate * 0.3)
        overall = min(overall, 1.0)
        
        return {
            "overall_score": round(overall, 4),
            "oov_ratio": round(oov_ratio, 4),
            "garbage_density": round(garbage_density, 4),
            "formatting_anomaly_rate": round(anomaly_rate, 4)
        }
