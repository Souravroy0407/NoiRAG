"""
baseline/noise_injector.py

Injects controlled OCR-like noise into clean ground-truth JSON docs.

Two noise types:
  - Semantic: character-level errors (swaps, deletions, insertions, substitutions)
  - Formatting: structure-level errors (merged lines, broken whitespace, garbage text)

Four noise levels: 10%, 25%, 50%, 75% of words affected.

Usage:
    python -m baseline.noise_injector                     # all types + levels
    python -m baseline.noise_injector --type semantic --level 25
    python -m baseline.noise_injector --limit 5           # test with 5 docs per domain
"""

import json
import random
import string
import copy
from pathlib import Path
from typing import List, Dict, Any


# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
GT_DIR         = PROJECT_ROOT / "data" / "ground_truth" / "gt"
NOISY_DIR      = PROJECT_ROOT / "data" / "noisy"
NOISE_TYPES    = ["semantic", "formatting"]
NOISE_LEVELS   = [10, 25, 50, 75]
RANDOM_SEED    = 42
# ─────────────────────────────────────────────────────────────────────────────


# ── Semantic Noise (character-level OCR errors) ──────────────────────────────

# Common OCR character confusions
CHAR_CONFUSIONS = {
    'a': ['o', 'e', 'á'],    'b': ['d', 'h', '6'],
    'c': ['e', 'o', '('],    'd': ['b', 'cl', 'a'],
    'e': ['c', 'o', 'é'],    'f': ['t', 'r'],
    'g': ['q', '9', 'y'],    'h': ['b', 'n', 'li'],
    'i': ['l', '1', '!'],    'j': ['i', ']'],
    'k': ['lc', 'x'],        'l': ['1', 'i', '|'],
    'm': ['rn', 'nn', 'in'], 'n': ['ri', 'u', 'h'],
    'o': ['0', 'a', 'c'],    'p': ['b', 'q'],
    'q': ['g', '9'],         'r': ['n', 'f', 'v'],
    's': ['5', '$'],          't': ['f', '+', '1'],
    'u': ['v', 'n', 'ü'],    'v': ['u', 'w', 'y'],
    'w': ['vv', 'uu'],       'x': ['k', '%'],
    'y': ['v', 'g'],         'z': ['2', 's'],
    'O': ['0', 'Q'],         'I': ['1', 'l', '|'],
    'S': ['5', '$'],         'B': ['8', 'D'],
}


def _corrupt_word_semantic(word: str, rng: random.Random) -> str:
    """Apply a random character-level OCR error to a word."""
    if len(word) < 2:
        return word

    error_type = rng.choice(["swap", "delete", "insert", "substitute", "confuse"])

    if error_type == "swap" and len(word) >= 3:
        # Swap two adjacent characters
        idx = rng.randint(0, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    elif error_type == "delete" and len(word) >= 3:
        # Delete a random character
        idx = rng.randint(1, len(word) - 2)  # keep first/last
        return word[:idx] + word[idx + 1:]

    elif error_type == "insert":
        # Insert a random character
        idx = rng.randint(1, len(word) - 1)
        char = rng.choice(string.ascii_lowercase)
        return word[:idx] + char + word[idx:]

    elif error_type == "substitute" and len(word) >= 2:
        # Replace a random character
        idx = rng.randint(0, len(word) - 1)
        char = rng.choice(string.ascii_lowercase)
        return word[:idx] + char + word[idx + 1:]

    elif error_type == "confuse":
        # Use OCR confusion table
        idx = rng.randint(0, len(word) - 1)
        original_char = word[idx]
        if original_char.lower() in CHAR_CONFUSIONS:
            replacement = rng.choice(CHAR_CONFUSIONS[original_char.lower()])
            if original_char.isupper():
                replacement = replacement.upper()
            return word[:idx] + replacement + word[idx + 1:]

    return word  # fallback: no change


def inject_semantic_noise(text: str, level: float, rng: random.Random) -> str:
    """
    Inject semantic (character-level) noise into text.
    level: fraction of words to corrupt (0.0–1.0)
    """
    words = text.split()
    if not words:
        return text

    num_to_corrupt = max(1, int(len(words) * level))
    indices = rng.sample(range(len(words)), min(num_to_corrupt, len(words)))

    for idx in indices:
        word = words[idx]
        # Skip very short words, numbers, and punctuation-only
        if len(word) < 2 or word.isdigit() or not any(c.isalpha() for c in word):
            continue
        words[idx] = _corrupt_word_semantic(word, rng)

    return " ".join(words)


# ── Formatting Noise (structure-level OCR errors) ────────────────────────────

GARBAGE_STRINGS = [
    "•", "■", "▪", "▸", "◆", " | ", " — ", "...", "***",
    "\n\n", "  ", "\t", "  —  ", ">> ", "<< ", "## ",
    "[...]", "(?)", "{~}", "//", "---", "===",
]


def inject_formatting_noise(text: str, level: float, rng: random.Random) -> str:
    """
    Inject formatting (structure-level) noise into text.
    level: fraction of lines/positions to corrupt (0.0–1.0)
    """
    lines = text.split("\n")
    if not lines:
        return text

    num_to_corrupt = max(1, int(len(lines) * level))
    indices = rng.sample(range(len(lines)), min(num_to_corrupt, len(lines)))

    for idx in indices:
        line = lines[idx]
        if not line.strip():
            continue

        error = rng.choice([
            "merge_next", "break_mid", "garbage_insert",
            "extra_whitespace", "strip_spaces",
        ])

        if error == "merge_next" and idx + 1 < len(lines) and lines[idx + 1].strip():
            # Merge with next line (missing line break)
            lines[idx] = line.rstrip() + " " + lines[idx + 1].lstrip()
            lines[idx + 1] = ""

        elif error == "break_mid" and len(line) > 10:
            # Break line at arbitrary position
            mid = rng.randint(len(line) // 4, 3 * len(line) // 4)
            lines[idx] = line[:mid] + "\n" + line[mid:]

        elif error == "garbage_insert":
            # Insert garbage text
            garbage = rng.choice(GARBAGE_STRINGS)
            pos = rng.randint(0, max(0, len(line) - 1))
            lines[idx] = line[:pos] + garbage + line[pos:]

        elif error == "extra_whitespace":
            # Random extra whitespace
            words = line.split()
            new_words = []
            for w in words:
                spaces = " " * rng.randint(1, 4)
                new_words.append(w + spaces)
            lines[idx] = "".join(new_words)

        elif error == "strip_spaces":
            # Remove meaningful whitespace
            lines[idx] = line.replace("  ", " ").replace(" ,", ",").replace(" .", ".")

    # Remove empty lines created by merges
    result = "\n".join(line for line in lines if line is not None)
    return result


# ── Main Injection Logic ─────────────────────────────────────────────────────

def inject_noise_to_doc(
    doc_pages: List[Dict[str, Any]],
    noise_type: str,
    level: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Inject noise into all pages of a document."""
    noisy_pages = copy.deepcopy(doc_pages)
    inject_fn = inject_semantic_noise if noise_type == "semantic" else inject_formatting_noise

    for page in noisy_pages:
        if page.get("text", "").strip():
            page["text"] = inject_fn(page["text"], level, rng)

    return noisy_pages


def process_all(
    noise_type: str = "semantic",
    level: int = 10,
    limit: int = 0,
    verbose: bool = True,
):
    """
    Process all ground-truth docs and save noisy versions.

    Args:
        noise_type: "semantic" or "formatting"
        level: noise percentage (10, 25, 50, 75)
        limit: max docs per domain (0 = all)
    """
    level_frac = level / 100.0
    output_dir = NOISY_DIR / f"{noise_type}_{level}"
    rng = random.Random(RANDOM_SEED + level)  # deterministic per level

    if not GT_DIR.exists():
        raise FileNotFoundError(f"Ground truth not found: {GT_DIR}")

    # Process each domain
    domains = sorted(d.name for d in GT_DIR.iterdir() if d.is_dir())
    total_docs = 0
    total_pages = 0

    for domain in domains:
        domain_dir = GT_DIR / domain
        json_files = sorted(domain_dir.glob("*.json"))

        if limit > 0:
            json_files = json_files[:limit]

        out_domain = output_dir / domain
        out_domain.mkdir(parents=True, exist_ok=True)

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                pages = json.load(f)

            noisy_pages = inject_noise_to_doc(pages, noise_type, level_frac, rng)

            out_path = out_domain / jf.name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(noisy_pages, f, ensure_ascii=False, indent=2)

            total_docs += 1
            total_pages += len(pages)

        if verbose:
            print(f"  {domain}: {len(json_files)} docs → {out_domain}")

    if verbose:
        print(f"\n✅ {noise_type}_{level}: {total_docs} docs, {total_pages} pages → {output_dir}")

    return total_docs


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inject OCR noise into ground-truth docs")
    parser.add_argument("--type", choices=NOISE_TYPES, default=None,
                        help="Noise type (default: both)")
    parser.add_argument("--level", type=int, choices=NOISE_LEVELS, default=None,
                        help="Noise level %% (default: all levels)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max docs per domain (0 = all)")
    args = parser.parse_args()

    types = [args.type] if args.type else NOISE_TYPES
    levels = [args.level] if args.level else NOISE_LEVELS

    print(f"Noise injection: types={types}, levels={levels}")
    if args.limit:
        print(f"Limiting to {args.limit} docs per domain")
    print(f"Source: {GT_DIR}\n")

    for ntype in types:
        for nlevel in levels:
            print(f"── {ntype} @ {nlevel}% ──")
            process_all(ntype, nlevel, args.limit)
            print()

    print("🎉 All noise injection complete!")
