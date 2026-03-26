"""
pipeline/retriever/faiss_retriever.py

Loads a FAISS index and metadata, embeds a query using
BAAI/bge-small-en-v1.5 (same model used for indexing),
and retrieves the top-k most relevant chunks.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K      = 5  # number of chunks to retrieve per query
# ─────────────────────────────────────────────────────────────────────────────


class FAISSRetriever:
    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k

        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        print(f"  Index loaded — {self.index.ntotal} vectors")

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"  Metadata loaded — {len(self.metadata)} entries")

        # Load BGE-small model (same as embedder — 384-dim)
        print(f"Loading model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("  Model ready!")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string using bge-small-en-v1.5."""
        vector = self.model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
        ).astype("float32")
        return vector

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks for a query.

        Returns a list of dicts:
            {
                "rank":      int,   # 1 = most relevant
                "score":     float, # cosine similarity score
                "doc_name":  str,   # source document
                "chunk_id":  int,   # chunk number within doc
                "vector_id": int,   # position in FAISS index
            }
        """
        query_vector = self.embed_query(query)
        scores, indices = self.index.search(query_vector, self.top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:  # FAISS returns -1 if not enough results
                continue
            meta = self.metadata[idx]
            results.append({
                "rank":      rank,
                "score":     float(score),
                "doc_name":  meta["doc_name"],
                "chunk_id":  meta["chunk_id"],
                "vector_id": meta["vector_id"],
            })

        return results


def load_chunk_text(
    chunks_path: str | Path,
    doc_name: str,
    chunk_id: int,
) -> str:
    """Helper to fetch the actual text of a retrieved chunk."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    for chunk in chunks:
        if chunk["doc_name"] == doc_name and chunk["chunk_id"] == chunk_id:
            return chunk["text"]
    return ""


# ── CLI (quick test) ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FAISS retriever")
    parser.add_argument("query",         help="Query string to search")
    parser.add_argument("--index",       default="data/index/gt.faiss")
    parser.add_argument("--metadata",    default="data/index/gt_metadata.json")
    parser.add_argument("--chunks",      default="data/chunks/gt_chunks.json")
    parser.add_argument("--top-k",       type=int, default=TOP_K)
    args = parser.parse_args()

    retriever = FAISSRetriever(args.index, args.metadata, args.top_k)
    results   = retriever.retrieve(args.query)

    print(f"\nTop {args.top_k} results for: '{args.query}'\n")
    for r in results:
        text = load_chunk_text(args.chunks, r["doc_name"], r["chunk_id"])
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Doc: {r['doc_name']} | Chunk: {r['chunk_id']}")
        print(f"  {text[:200]}...")
        print()