# NoiRAG: Noise-Resilient Retrieval-Augmented Generation
## Project Overview & Architecture Report

---

### **1. Project Purpose**
Real-world RAG (Retrieval-Augmented Generation) systems often fail when processing messy PDFs full of OCR errors, arbitrary line breaks, and formatting garbage. **NoiRAG** is a lightweight, intelligent preprocessing engine built to solve this. It intercepts and cleans noisy documents *before* they are embedded into the vector database, recovering lost retrieval performance without relying on expensive and slow Large Language Models (LLMs) for every single document.

---

### **2. Folder Structure & Modules**

*   **`baseline/`** 
    *   **Purpose:** The environment testing tools. Contains `noise_injector.py` to artificially corrupt clean documents and `run_baseline.py` to measure uncleaned performance.
*   **`data/`** 
    *   **Purpose:** Storage for datasets. Holds the `ground_truth` (perfect data), `noisy` (corrupted data), and `cleaned` (repaired by NoiRAG) JSONs, as well as the `.faiss` vector indices.
*   **`noirag/preprocessing/`** 
    *   **Purpose:** The core NoiRAG engine containing Rule-Based, Statistical, and Hybrid orchestration logic.
*   **`noirag/pipeline/`** 
    *   **Purpose:** The standard RAG retrieval implementation utilizing BGE-Small and FAISS.
*   **`noirag/run_noirag.py`** 
    *   **Purpose:** The master experiment runner that executes both preprocessing and retrieval evaluation automatically.
*   **`results/tables/`**
    *   **Purpose:** Output store containing JSON tables showing exact metric degradation and recovery.

---

### **3. Deep Dive: What NoiRAG Does in Preprocessing**
The core innovation of NoiRAG is its pre-embedding workflow, which intelligently decides **how** to clean text rather than blindly applying expensive operations. 

1.  **Stage 1: The Quality Scorer (`quality_scorer.py`)**
    *   Every single text chunk first passes through a heuristic evaluator.
    *   It calculates an "Out-of-Vocabulary (OOV) Ratio" using an English dictionary to detect semantic spelling noise.
    *   It calculates a "Garbage Density Score" by scanning for OCR artifacts (like `■`, `[...]`) and weird spacing (multiple spaces, unexpected broken lines).
    *   This generates a final Noisiness score between **`0.0` (Clean) and `1.0` (Heavily Corrupted)**.
2.  **Stage 2: The Hybrid Orchestrator (`hybrid_cleaner.py`)**
    *   Instead of running all cleaners on all text, the orchestrator acts as a router.
    *   If the Noisiness score is extremely low, it **bypasses the chunk**, saving computation time and preventing accidental corruption of inherently clean data.
    *   If the formatting score > 0.05, it routes the text to the Rule-Based Cleaner.
    *   If the semantic score > 0.10, it routes the text to the Statistical Cleaner.
3.  **Stage 3: Targeted Repair Execution**
    *   **The Rule-Based Cleaner:** Fixes structural damage using Regex. It removes non-linguistic OCR markers (`===`, `~`), standardizes unicode characters, and intelligently merges arbitrary line breaks (`comput-\ners` -> `computers`) back into fluid sentences.
    *   **The Statistical Cleaner:** Uses the high-speed `symspellpy` symmetric-delete algorithm to perform edit-distance spell-checking (`computars` -> `computers`). Crucially, it employs **Conservative Fences**—it permanently ignores words under 3 characters, words with numbers (`2023`, `1.5M`), and all-caps acronyms (`EBITDA`) to guarantee it does not destroy valid, domain-specific terminology.

---

### **4. Understanding Our RAG Evaluation Metrics**
To objectively prove NoiRAG works, we chunk our text, embed it using the `BAAI/bge-small-en-v1.5` AI model into a FAISS index, and test it against a database of specific Q&A pairs. 

We measure success by evaluating if our model successfully "retrieved" the exact paragraph that contains the answer. We use 4 primary metrics:

*   **P@1 (Precision at 1):** The percentage of time the absolute #1 most relevant chunk returned by FAISS contained the correct answer to the question. *(This represents our primary benchmark for accuracy).*
*   **R@5 (Recall at 5):** Out of the top 5 chunks returned by the system, does *at least one* of them contain the correct answer? *(Crucial for LLM generation, as any LLM will read the top 5 chunks into its context window).*
*   **MRR (Mean Reciprocal Rank):** Measures *how far down* the list the correct answer appeared. If the answer is #1, MRR is `1.0`. If it is ranked #2, MRR is `0.5`, #3 is `0.33`, etc. Higher MRR means the correct paragraph is consistently closer to the top spot.
*   **NDCG@5 (Normalized Discounted Cumulative Gain):** Similar to MRR, but it evaluates the overall *quality* and *relevance* ranking of the entire top 5 retrieved list, punishing the system heavily if highly relevant paragraphs are ranked near the bottom.

---

### **5. Results: Our Final Score Output Analysis**
When testing the pipeline against a dataset injected with 25% Semantic Noise (typos), we observed the following **P@1 (Precision at 1) scores:**

*   **Ground Truth (The Perfect Ceiling):** 68.2%
*   **Noisy Baseline (The Degraded Floor):** 63.5%
*   **NoiRAG Cleaned Dataset (Our Solution):** 65.6%

**What this means:** The injected noise destroyed 4.7% of the total available accuracy (from 68.2 down to 63.5). By passing the noisy data through our highly conservative NoiRAG preprocessing engine, **we recovered exactly 45% of the performance lost to the noise** (recovering +2.1% back to 65.6%). We accomplished this using zero-cost, lightning-fast algorithmic heuristics without making a single slow, expensive API call to an LLM. 

---

### **6. Future Scope: How to Recover the Remaining 55%**
To push our performance recovery even higher in the future, we designed the following scalable upgrades:
1.  **Aggressive Edit Distances:** We recently increased the `symspellpy` max-edit distance from `2` to `3`. This allows the cleaner to repair more severely corrupted terminology.
2.  **Context-Aware Spell Checking (N-Grams):** Upgrading the Statistical Cleaner to evaluate words based on their surrounding context (using a localized Transformer model) rather than checking words in isolation. 
3.  **LLM Routing (The Final Threshold):** The Hybrid Orchestrator's greatest strength is dynamic routing. In the future, if a chunk achieves a critically high noise score (e.g., Scorer > 0.40), the orchestrator can specifically route that single chunk to a fast LLM (like GPT-4o-mini) with the prompt: *"Clean the OCR jargon from this sentence exactly without altering its meaning."* This isolates expensive LLM calls structurally, ensuring they are only ever executed when absolutely necessary on the most damaged documents.
