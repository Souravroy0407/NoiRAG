"""
pipeline/chunker/text_splitter.py

Reads cleaned .json document files (gt/ or MinerU/ format),
concatenates all pages into one document, then splits into
overlapping chunks ready for bge-small-en-v1.5 embedding into FAISS.

Doc format: [{page_idx, text}, ...]
"""

import json
from pathlib import Path
from typing import List, Dict, Any


# ── Config ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512  # words per chunk (≈ 400 tokens, fits bge-small-en-v1.5's 512-token limit)
CHUNK_OVERLAP = 64   # words overlap between consecutive chunks
# ─────────────────────────────────────────────────────────────────────────────


def load_doc(json_path: str | Path) -> str:
    """Load a NoiRAG JSON doc and return all page text concatenated."""
    with open(json_path, "r", encoding="utf-8") as f:
        pages: List[Dict[str, Any]] = json.load(f)

    pages = sorted(pages, key=lambda p: p.get("page_idx", 0))
    full_text = "\n".join(p["text"] for p in pages if p.get("text", "").strip())
    return full_text


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Word-level sliding window chunker. Returns a list of chunk strings."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def chunk_file(
    json_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Load one JSON doc file and return a list of chunk dicts.

    Each chunk dict:
        {
            "doc_name": str,   # stem of the source file e.g. "2301.00001"
            "chunk_id": int,   # 0-indexed chunk number
            "text":     str,   # chunk content
        }
    """
    json_path = Path(json_path)
    full_text = load_doc(json_path)
    raw_chunks = split_into_chunks(full_text, chunk_size, overlap)

    doc_name = json_path.stem
    return [
        {"doc_name": doc_name, "chunk_id": i, "text": chunk}
        for i, chunk in enumerate(raw_chunks)
    ]


def chunk_directory(
    input_dir: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process all .json files in a directory.
    Returns a flat list of all chunk dicts across all docs.
    """
    input_dir = Path(input_dir)
    all_chunks: List[Dict[str, Any]] = []

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    for i, jf in enumerate(json_files):
        chunks = chunk_file(jf, chunk_size, overlap)
        all_chunks.extend(chunks)
        if verbose and (i + 1) % 100 == 0:
            print(f"  Chunked {i+1}/{len(json_files)} files — {len(all_chunks)} chunks so far")

    if verbose:
        print(f"\nDone: {len(json_files)} docs → {len(all_chunks)} chunks")
        print(f"Avg chunks/doc: {len(all_chunks)/len(json_files):.1f}")

    return all_chunks


def save_chunks(chunks: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Save chunk list to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk NoiRAG JSON docs")
    parser.add_argument("input_dir",   help="e.g. data/gt")
    parser.add_argument("output_file", help="e.g. data/chunks/gt_chunks.json")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap",    type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    print(f"Chunking: {args.input_dir}  (chunk_size={args.chunk_size}, overlap={args.overlap})")
    chunks = chunk_directory(args.input_dir, args.chunk_size, args.overlap)
    save_chunks(chunks, args.output_file)