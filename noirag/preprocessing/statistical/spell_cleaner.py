"""
Statistical text cleaner for NoiRAG.
Uses symmetric delete spell-checking to fix character-level semantic noise.
"""
import re
import argparse
import sys
import json
import importlib.resources
from typing import List, Dict, Any

try:
    from symspellpy import SymSpell, Verbosity
except ImportError:
    print("Error: symspellpy is required for StatisticalCleaner. Run `pip install symspellpy`")
    sys.exit(1)

class StatisticalCleaner:
    def __init__(self, max_dictionary_edit_distance: int = 3, prefix_length: int = 7):
        # Initialize SymSpell
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_dictionary_edit_distance, 
            prefix_length=prefix_length
        )
        
        # Load default English dictionary from symspellpy
        dictionary_path = str(importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt")
        
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
    def _should_ignore_token(self, token: str) -> bool:
        """Determines if a token should be skipped for spell checking."""
        # Clean punctuation from edges for checking
        clean_token = token.strip('.,!?;:\'\"()[]{}')
        
        # 1. Very short (likely a particle, initial, or not worth risking)
        # Note: We still spellcheck small common words if they are in dict, 
        # but symspell handles it. Actually symspell can be aggressive on 1-2 char words.
        # So let's skip < 3 char words unless they have a known error pattern, but easier to just skip.
        if len(clean_token) < 3:
            return True
            
        # 2. Contains numbers (years, metrics, financial data)
        if any(char.isdigit() for char in clean_token):
            return True
            
        # 3. All uppercase (Acronyms like EBITDA, LSTM)
        if clean_token.isupper():
            return True
            
        return False

    def correct_word(self, token: str) -> str:
        """Corrects a single token, preserving punctuation and case where possible."""
        # Extract leading/trailing punctuation
        match = re.match(r'^([^a-zA-Z0-9]*)(.*?)([^a-zA-Z0-9]*)$', token)
        if not match:
            return token
            
        prefix, core_word, suffix = match.groups()
        
        if not core_word or self._should_ignore_token(core_word):
            return token
            
        is_title = core_word.istitle()
        is_upper = core_word.isupper()
        
        # We query in lowercase
        suggestions = self.sym_spell.lookup(
            core_word.lower(), 
            Verbosity.TOP,          # Return only the top suggestion
            max_edit_distance=3     # Up to 3 edits
        )
        
        if not suggestions:
            return token # No correction found
            
        best_correction = suggestions[0].term
        
        # Restore case
        if is_upper:
            best_correction = best_correction.upper()
        elif is_title:
            best_correction = best_correction.capitalize()
            
        return f"{prefix}{best_correction}{suffix}"

    def clean(self, text: str) -> str:
        """Executes the statistical cleaner on the entire text."""
        if not text:
            return text
            
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if not line.strip():
                cleaned_lines.append(line)
                continue
                
            # Split by whitespace, keeping track of spaces. 
            # A simple split() loses exact spacing, which `RuleBasedCleaner` standardizes anyway.
            # But let's use a regex to preserve existing spaces exactly just in case.
            tokens = re.split(r'(\s+)', line)
            
            cleaned_tokens = []
            for token in tokens:
                if not token.strip(): # It's whitespace
                    cleaned_tokens.append(token)
                else:
                    cleaned_tokens.append(self.correct_word(token))
                    
            cleaned_lines.append("".join(cleaned_tokens))
            
        return '\n'.join(cleaned_lines)

def clean_document_pages(pages: List[Dict[str, Any]], cleaner: StatisticalCleaner) -> List[Dict[str, Any]]:
    """Cleans the 'text' field of all pages in a document."""
    for page in pages:
        if "text" in page and page["text"]:
            page["text"] = cleaner.clean(page["text"])
    return pages

def main():
    parser = argparse.ArgumentParser(description="Clean text via statistical edit-distance spell checking.")
    parser.add_argument("--input", type=str, help="Input raw text file")
    parser.add_argument("--json", type=str, help="Input JSON file with list of page dicts")
    parser.add_argument("--output", type=str, default="stdout", help="Output file")
    
    args = parser.parse_args()
    cleaner = StatisticalCleaner()
    
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
        
        cleaned = cleaner.clean(text)
        if args.output == "stdout":
            print(cleaned)
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(cleaned)
    else:
        # Read from stdin
        text = sys.stdin.read()
        cleaned = cleaner.clean(text)
        if args.output == "stdout":
            print(cleaned)
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(cleaned)

if __name__ == "__main__":
    main()
