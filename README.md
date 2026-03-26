# NoiRAG: Noise-Resilient Retrieval-Augmented Generation

## Project Overview

Real-world RAG (Retrieval-Augmented Generation) systems often fail when processing messy PDFs full of OCR errors, arbitrary line breaks, and formatting garbage. **NoiRAG** is a lightweight, intelligent preprocessing engine built to solve this. It intercepts and cleans noisy documents *before* they are embedded into the vector database, recovering lost retrieval performance without relying on expensive and slow Large Language Models (LLMs) for every single document.

## Architecture & Modules

The project is structured into several key components:

*   **`baseline/`**: Environment testing tools containing `noise_injector.py` to artificially corrupt clean documents and `run_baseline.py` to measure uncleaned performance.
*   **`data/`**: Storage for datasets. Holds the `ground_truth` (perfect data), `noisy` (corrupted data), and `cleaned` (repaired by NoiRAG) JSONs, as well as the `.faiss` vector indices.
*   **`noirag/preprocessing/`**: The core NoiRAG engine containing Rule-Based, Statistical, and Hybrid orchestration logic.
*   **`noirag/pipeline/`**: The standard RAG retrieval implementation utilizing BGE-Small and FAISS.
*   **`noirag/run_noirag.py`**: The master experiment runner that executes both preprocessing and retrieval evaluation automatically.
*   **`results/tables/`**: Output store containing JSON tables showing exact metric degradation and recovery.

## How NoiRAG Preprocessing Works

The core innovation of NoiRAG is its intelligent pre-embedding workflow:

1.  **Quality Scorer (`quality_scorer.py`)**: Every text chunk passes through a heuristic evaluator that calculates an Out-of-Vocabulary (OOV) ratio and Garbage Density Score to generate a final Noisiness score between 0.0 and 1.0.
2.  **Hybrid Orchestrator (`hybrid_cleaner.py`)**: Acts as a router. Clean chunks bypass processing to save computation. Chunks with high formatting or semantic noise are routed to the appropriate cleaners.
3.  **Targeted Repair Execution**:
    *   **Rule-Based Cleaner**: Fixes structural damage using Regex (removing non-linguistic markers, standardizing unicode, merging arbitrary line breaks).
    *   **Statistical Cleaner**: Uses `symspellpy` for high-speed edit-distance spell-checking, employing Conservative Fences (ignoring small words, numbers, acronyms) to protect valid terminology.

## RAG Evaluation Metrics

We evaluate performance using the following metrics against a FAISS index embedded with `BAAI/bge-small-en-v1.5`:

*   **P@1 (Precision at 1)**: Primary benchmark for accuracy. Does the #1 retrieved chunk contain the correct answer?
*   **R@5 (Recall at 5)**: Out of the top 5 chunks, does at least one contain the answer?
*   **MRR (Mean Reciprocal Rank)**: Measures how close to the top spot the correct answer appears.
*   **NDCG@5**: Evaluates the overall quality and relevance ranking of the top 5 retrieved list.

## Results & Performance Recovery

When tested against a dataset injected with 25% Semantic Noise, NoiRAG achieved the following P@1 scores:

*   **Ground Truth**: 68.2%
*   **Noisy Baseline**: 63.5%
*   **NoiRAG Cleaned Dataset**: **65.6%**

*By processing the noisy data through NoiRAG's conservative preprocessing engine, we successfully recovered 45% of the performance lost to the noise (from 63.5% to 65.6%) at zero cost rapidly, without making a single slow, expensive LLM API call.*
