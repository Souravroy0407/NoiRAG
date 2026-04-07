# NoiRAG: Noise-Aware Retrieval-Augmented Generation

## Project Overview

Real-world RAG (Retrieval-Augmented Generation) systems often fail when processing messy PDFs full of OCR errors, arbitrary line breaks, and formatting garbage. **NoiRAG** is a lightweight, intelligent preprocessing engine built to solve this. It intercepts and cleans noisy documents *before* they are embedded into the vector database, recovering lost retrieval performance using a **Hybrid Triage Architecture** that dynamically routes text through the optimal cleaning strategy.

## Architecture & Modules

```
NoiRAG/
├── baseline/               # Noise injection & uncleaned baseline measurement
│   ├── noise_injector.py   # Artificially corrupts clean documents
│   └── run_baseline.py     # Measures raw (uncleaned) retrieval performance
├── noirag/
│   ├── preprocessing/
│   │   ├── rule_based/     # Regex-based formatting repair
│   │   ├── statistical/    # SymSpellPy edit-distance spell correction
│   │   └── hybrid/         # Hybrid Orchestrator + Quality Scorer + LLM Cleaner
│   │       ├── hybrid_cleaner.py   # Routes chunks to the right cleaner
│   │       ├── quality_scorer.py   # Scores noisiness (OOV + Garbage Density)
│   │       └── llm_cleaner.py      # Local Ollama LLM for severe corruption
│   ├── tests/              # Unit tests for all cleaners
│   └── run_noirag.py       # Master experiment runner
├── pipeline/               # Standard RAG components
│   ├── chunker/            # Text splitting
│   ├── embedder/           # BGE-Small embedding
│   ├── retriever/          # FAISS vector retrieval
│   └── generator/          # LLM answer generation
├── evaluation/             # Retrieval & generation evaluation metrics
├── configs/config.yaml     # Central configuration
├── data/                   # Datasets (ground truth, noisy, cleaned)
└── results/tables/         # Benchmark evaluation outputs (JSON)
```

## How NoiRAG Preprocessing Works

The core innovation of NoiRAG is its **Hybrid Triage Architecture**:

1.  **Quality Scorer (`quality_scorer.py`)**: Every text chunk passes through a heuristic evaluator that calculates an Out-of-Vocabulary (OOV) ratio and Garbage Density Score to generate a final Noisiness score between 0.0 and 1.0.
2.  **Hybrid Orchestrator (`hybrid_cleaner.py`)**: Acts as an intelligent router:
    *   **Clean chunks** (score < 0.05) → bypass processing entirely.
    *   **Formatting noise** (garbage > 0.05) → routed to the Rule-Based Cleaner.
    *   **Semantic noise** (OOV > 0.10) → routed to the Statistical Cleaner.
    *   **Severely corrupted** (score > 0.60) → routed to the Local LLM Cleaner.
3.  **Targeted Repair Execution**:
    *   **Rule-Based Cleaner**: Fixes structural damage using Regex (removing non-linguistic markers, standardizing unicode, merging arbitrary line breaks).
    *   **Statistical Cleaner**: Uses `symspellpy` for high-speed edit-distance spell-checking, employing Conservative Fences (ignoring small words, numbers, acronyms) to protect valid terminology.
    *   **LLM Cleaner**: Uses a local Ollama model (`qwen2.5:0.5b`) as a final firewall for the most damaged text, running entirely offline with zero API costs.

## RAG Evaluation Metrics

We evaluate performance using the following metrics against a FAISS index embedded with `BAAI/bge-small-en-v1.5`:

*   **P@1 (Precision at 1)**: Does the #1 retrieved chunk contain the correct answer?
*   **R@5 (Recall at 5)**: Out of the top 5 chunks, does at least one contain the answer?
*   **MRR (Mean Reciprocal Rank)**: Measures how close to the top spot the correct answer appears.
*   **NDCG@5**: Evaluates the overall quality and relevance ranking of the top 5 retrieved list.

## Results & Performance Recovery

| Metric | Ground Truth | Noisy (75% Semantic) | NoiRAG Cleaned |
|--------|-------------|---------------------|----------------|
| **P@1** | 100% | **0%** | **100%** |
| **MRR** | 1.000 | 0.361 | **1.000** |
| **NDCG@5** | 0.968 | 0.541 | **0.941** |

*NoiRAG achieved **100% recovery** of the P@1 score on 75% semantic noise, restoring a completely broken RAG pipeline back to near-perfect accuracy.*

## Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/shreyabag028/NoiRAG.git
cd NoiRAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install Ollama for LLM cleaning
# Download from https://ollama.com, then:
ollama pull qwen2.5:0.5b

# 4. Run the evaluation pipeline
python -m noirag.run_noirag --noise-type semantic --noise-level 75

# 5. Run tests
pytest noirag/tests/ -v
```

## Environment Variables

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_here
```
