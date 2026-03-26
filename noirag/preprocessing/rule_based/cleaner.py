"""
Rule-based text cleaner for NoiRAG.
Overrides formatting noise and standardizes text for downstream RAG components.
"""
import re
import string
import unicodedata
import argparse
import sys
import json
from typing import List, Dict, Any

class RuleBasedCleaner:
    def __init__(self):
        # Precise list of garbage strings injected by our baseline injector
        # Sorted by length descending to prevent partial replacements (e.g. "..." before "[...]")
        raw_garbage = [
            "•", "■", "▪", "▸", "◆", " | ", " — ", "...", "***",
            "\n\n", "  ", "\t", "  —  ", ">> ", "<< ", "## ",
            "[...]", "(?)", "{~}", "//", "---", "==="
        ]
        self.garbage_strings = sorted(raw_garbage, key=len, reverse=True)
        
    def remove_garbage_strings(self, text: str) -> str:
        """Removes known OCR artifacts and garbage strings."""
        for g in self.garbage_strings:
            # We replace garbage with space to avoid merging words inadvertently,
            # except for structural things like \n\n or \t if we handle them in whitespace.
            # Actually, the injector inserted these literally.
            if g in ["\n\n", "  ", "\t"]:
                continue # handled in whitespace normalization
            text = text.replace(g, " ")
        return text

    def normalize_unicode(self, text: str) -> str:
        """Applies NFKC Unicode normalization."""
        return unicodedata.normalize('NFKC', text)
        
    def fix_punctuation_spacing(self, text: str) -> str:
        """Adds missing spaces after commas and periods."""
        # Add space after comma if followed by a letter
        text = re.sub(r',([a-zA-Z])', r', \1', text)
        # Add space after period if followed by a letter
        text = re.sub(r'\.([a-zA-Z])', r'. \1', text)
        return text

    def repair_broken_lines(self, text: str) -> str:
        """Heuristic to merge lines that were arbitrarily broken mid-sentence."""
        lines = text.split('\n')
        if not lines:
            return text
            
        merged = [lines[0]]
        for i in range(1, len(lines)):
            curr_line = lines[i]
            prev_line = merged[-1]
            
            if not curr_line.strip():
                merged.append(curr_line)
                continue
                
            if not prev_line.strip():
                merged.append(curr_line)
                continue
                
            # If prev_line doesn't end with sentence-ending punctuation 
            # and curr_line doesn't start with an uppercase letter, they might be broken.
            # The injector breaks randomly (e.g. `line[:mid] + "\n" + line[mid:]`).
            prev_last_char = prev_line.rstrip()[-1] if prev_line.rstrip() else ''
            curr_first_char = curr_line.lstrip()[0] if curr_line.lstrip() else ''
            
            if prev_last_char and prev_last_char not in '.!?:"\'' and curr_first_char and curr_first_char.islower():
                # Merge them
                merged[-1] = prev_line + curr_line
            else:
                merged.append(curr_line)
            
        return '\n'.join(merged)

    def normalize_whitespace(self, text: str) -> str:
        """Collapses multiple spaces, trims lines, and limits consecutive newlines."""
        # Collapse multiple spaces and tabs into a single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        lines = [line.strip() for line in text.split('\n')]
        
        cleaned_lines = []
        for line in lines:
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                # Allow a single empty line between paragraphs
                cleaned_lines.append("")
                
        # Remove trailing empty line if it exists
        if cleaned_lines and cleaned_lines[-1] == "":
            cleaned_lines.pop()
            
        return '\n'.join(cleaned_lines)
        
    def remove_duplicate_lines(self, text: str) -> str:
        """Removes consecutive duplicate lines."""
        lines = text.split('\n')
        if not lines:
            return text
            
        cleaned = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i].strip() and lines[i].strip() == lines[i-1].strip():
                continue # Skip duplicate
            cleaned.append(lines[i])
        return '\n'.join(cleaned)

    def clean(self, text: str) -> str:
        """Executes the full rule-based cleaning pipeline."""
        if not text:
            return text
            
        text = self.normalize_unicode(text)
        text = self.remove_garbage_strings(text)
        text = self.fix_punctuation_spacing(text)
        text = self.repair_broken_lines(text)
        text = self.remove_duplicate_lines(text)
        text = self.normalize_whitespace(text)
        return text

def clean_document_pages(pages: List[Dict[str, Any]], cleaner: RuleBasedCleaner) -> List[Dict[str, Any]]:
    """Cleans the 'text' field of all pages in a document."""
    for page in pages:
        if "text" in page and page["text"]:
            page["text"] = cleaner.clean(page["text"])
    return pages

def main():
    parser = argparse.ArgumentParser(description="Clean text via rule-based heuristics.")
    parser.add_argument("--input", type=str, help="Input raw text file")
    parser.add_argument("--json", type=str, help="Input JSON file with list of page dicts")
    parser.add_argument("--output", type=str, default="stdout", help="Output file")
    
    args = parser.parse_args()
    cleaner = RuleBasedCleaner()
    
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
