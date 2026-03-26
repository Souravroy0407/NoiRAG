"""
pipeline/embedder/bge_small_embedder.py

Loads chunks from a JSON file, embeds them using BAAI/bge-small-en-v1.5
(runs locally via sentence-transformers — no API key needed),
and saves a FAISS index + metadata file for retrieval.

Inputs:  data/chunks/gt_chunks.json
Outputs: data/index/gt.faiss
         data/index/gt_metadata.json
"""

import json
import sys
import time
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64         # Local model — can handle larger batches
EMBED_DIM  = 384        # bge-small-en-v1.5 output dimension
# ─────────────────────────────────────────────────────────────────────────────


def load_chunks(chunks_path: str | Path) -> List[Dict[str, Any]]:
    """Load chunks from JSON file."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_time(seconds: float) -> str:
    """Format seconds into mm:ss or hh:mm:ss."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def embed_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Embed all chunk texts using BAAI/bge-small-en-v1.5.
    Returns a float32 numpy array of shape (num_chunks, 384).
    """
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]
    total = len(texts)
    num_batches = (total + batch_size - 1) // batch_size
    print(f"\nEmbedding {total} chunks in {num_batches} batches of {batch_size}...\n")

    all_embeddings = []
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]

        batch_emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_embeddings.append(batch_emb)

        # ── Live progress ─────────────────────────────────────────
        done = min(i + batch_size, total)
        pct = done / total * 100
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0

        bar_len = 30
        filled = int(bar_len * done // total)
        bar = "█" * filled + "░" * (bar_len - filled)

        sys.stdout.write(
            f"\r  [{bar}] {done:,}/{total:,} chunks "
            f"({pct:.1f}%) | "
            f"Elapsed: {_format_time(elapsed)} | "
            f"ETA: {_format_time(eta)}   "
        )
        sys.stdout.flush()

    print()  # newline after progress bar
    embeddings = np.vstack(all_embeddings).astype("float32")
    total_time = time.time() - start_time
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Total time: {_format_time(total_time)}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat inner-product index from embeddings.
    Vectors are already normalized, so IP ≈ cosine similarity.
    """
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"FAISS index built — {index.ntotal} vectors")
    return index


def save_index(
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    output_dir: str | Path,
    name: str = "gt",
) -> None:
    """Save FAISS index and metadata (doc_name + chunk_id per vector)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_path = output_dir / f"{name}.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index → {index_path}")

    # Save metadata (one entry per vector, same order as index)
    metadata = [
        {"vector_id": i, "doc_name": c["doc_name"], "chunk_id": c["chunk_id"]}
        for i, c in enumerate(chunks)
    ]
    meta_path = output_dir / f"{name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata     → {meta_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Auto-resolve project root (two levels up from this file)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Default paths — just run: python bge_small_embedder.py
    CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "gt_chunks.json"
    OUTPUT_DIR  = PROJECT_ROOT / "data" / "index"
    NAME        = "gt"

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Chunks file  : {CHUNKS_FILE}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Index name   : {NAME}\n")

    chunks     = load_chunks(CHUNKS_FILE)
    embeddings = embed_chunks(chunks, BATCH_SIZE)
    index      = build_faiss_index(embeddings)
    save_index(index, chunks, OUTPUT_DIR, NAME)

    print(f"\n✅ Done! Files saved:")
    print(f"  {OUTPUT_DIR / f'{NAME}.faiss'}")
    print(f"  {OUTPUT_DIR / f'{NAME}_metadata.json'}")

