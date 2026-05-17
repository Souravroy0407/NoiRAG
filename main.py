"""
NoiRAG — Streamlit Dashboard
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import json
import sys
import time
import os
from pathlib import Path

# Fix for Streamlit's file watcher crashing with PyTorch
try:
    import torch
    import sys
    if 'torch.classes' in sys.modules:
        del sys.modules['torch.classes']
except ImportError:
    pass

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page Config ──
st.set_page_config(
    page_title="NoiRAG Dashboard",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──
DATA_DIR = PROJECT_ROOT / "data"
GT_DIR = DATA_DIR / "ground_truth" / "gt"
NOISY_DIR = DATA_DIR / "noisy"
CLEANED_DIR = DATA_DIR / "cleaned" / "hybrid"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
QA_DIR = DATA_DIR / "qa"

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px; padding: 20px;
    text-align: center; margin: 5px;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #818cf8; }
.metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
.cost-box {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 12px; padding: 20px; margin: 8px 0;
}
.cost-saved { font-size: 1.8rem; font-weight: 700; color: #34d399; }
.route-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 500; margin: 2px;
}
.badge-bypass { background: #1e3a5f; color: #60a5fa; }
.badge-rule { background: #3b2f1e; color: #fbbf24; }
.badge-stat { background: #1e3b2f; color: #34d399; }
.badge-llm { background: #3b1e2f; color: #f472b6; }
.section-header {
    font-size: 1.4rem; font-weight: 600;
    border-bottom: 2px solid rgba(99,102,241,0.3);
    padding-bottom: 8px; margin-bottom: 16px;
}
.compare-panel {
    border-radius: 10px; padding: 16px;
    font-size: 0.85rem; line-height: 1.6;
    max-height: 400px; overflow-y: auto;
}
.panel-gt { background: #0f1f0f; border: 1px solid #22c55e40; }
.panel-noisy { background: #1f0f0f; border: 1px solid #ef444440; }
.panel-clean { background: #0f0f1f; border: 1px solid #6366f140; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("# 🧹 NoiRAG")
st.sidebar.markdown("*Noise-Aware RAG Engine*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "▶️ Run Pipeline",
    "💰 Cost Report",
    "📊 Benchmarks",
    "🔍 Text Comparison",
    "⚙️ Architecture",
])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🧹 NoiRAG Dashboard")
    st.markdown("**Noise-Aware Retrieval-Augmented Generation** — An intelligent preprocessing engine that recovers retrieval accuracy from noisy, OCR-damaged documents.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">2,002</div><div class="metric-label">QA Pairs Tested</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">7</div><div class="metric-label">Document Domains</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Cleaning Strategies</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">$0</div><div class="metric-label">API Costs</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. **Quality Scorer** — Evaluates every chunk's noisiness (OOV ratio + Garbage Density)
    2. **Hybrid Orchestrator** — Routes chunks to the right cleaner based on scores
    3. **Targeted Repair** — Rule-Based (formatting), Statistical (typos), or LLM (severe)
    4. **Evaluation** — Measures retrieval recovery using P@1, MRR, NDCG@5
    """)
    st.info("👈 Use the sidebar to navigate between features.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "▶️ Run Pipeline":
    st.markdown("# ▶️ Run NoiRAG Pipeline")
    st.markdown("Run the full preprocessing + evaluation pipeline from the GUI.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        noise_type = st.selectbox("Noise Type", ["semantic", "formatting"])
    with col2:
        noise_level = st.selectbox("Noise Level (%)", [10, 25, 50, 75])
    with col3:
        limit = st.number_input("QA Limit per Domain (0 = all)", min_value=0, max_value=1000, value=5)

    skip_cleaning = st.checkbox("Skip cleaning (use existing cleaned data)")

    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        noise_name = f"{noise_type}_{noise_level}"
        noisy_dir = NOISY_DIR / noise_name
        cleaned_dir = CLEANED_DIR / noise_name

        if not noisy_dir.exists():
            st.error(f"❌ Noisy data not found: `{noisy_dir}`")
            st.info("Run noise injection first: `python -m baseline.noise_injector`")
        else:
            progress = st.progress(0, text="Initializing...")
            log_area = st.empty()
            logs = []

            def log(msg):
                logs.append(msg)
                log_area.code("\n".join(logs[-20:]), language="text")

            try:
                # Step 1: Preprocessing
                try:
                    from codecarbon import EmissionsTracker
                    tracker = EmissionsTracker(project_name="noirag_streamlit", log_level="error")
                    tracker.start()
                except ImportError:
                    tracker = None
                    log("CodeCarbon not installed. Skipping emissions tracking.")
                    
                if not skip_cleaning:
                    progress.progress(10, text="Loading cleaners (cached)...")
                    log("Loading HybridCleaner...")
                    
                    @st.cache_resource
                    def get_cleaner():
                        from noirag.preprocessing.hybrid.hybrid_cleaner import HybridCleaner
                        return HybridCleaner(verbose=False)
                        
                    cleaner = get_cleaner()
                    from noirag.preprocessing.hybrid.hybrid_cleaner import clean_document_pages
                    cleaner.profiler.start()

                    progress.progress(20, text="Preprocessing noisy documents...")
                    
                    # Count total json files first for accurate progress
                    all_json_files = []
                    for domain_dir in sorted(noisy_dir.iterdir()):
                        if domain_dir.is_dir():
                            all_json_files.extend(list(domain_dir.glob("*.json")))
                    
                    total_docs = len(all_json_files)
                    processed_docs = 0
                    
                    for domain_dir in sorted(noisy_dir.iterdir()):
                        if not domain_dir.is_dir():
                            continue
                        out_domain = cleaned_dir / domain_dir.name
                        out_domain.mkdir(parents=True, exist_ok=True)
                        for jf in sorted(domain_dir.glob("*.json")):
                            with open(jf, "r", encoding="utf-8") as f:
                                pages = json.load(f)
                            cleaned_pages = clean_document_pages(pages, cleaner)
                            with open(out_domain / jf.name, "w", encoding="utf-8") as f:
                                json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)
                            processed_docs += 1
                            
                            # Update progress between 20 and 40
                            if total_docs > 0:
                                current_pct = 20 + int((processed_docs / total_docs) * 20)
                                progress.progress(current_pct, text=f"Preprocessing: {processed_docs}/{total_docs} docs...")
                                
                        log(f"  Cleaned {domain_dir.name}: done")

                    cleaner.profiler.stop()
                    log(f"✅ Preprocessing complete: {total_docs} docs")

                    # Save cost report
                    cost_report = cleaner.profiler.generate_report()
                    cost_path = RESULTS_DIR / f"cost_report_{noise_name}.json"
                    cost_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cost_path, "w") as f:
                        json.dump(cost_report, f, indent=2)
                    log(f"💾 Cost report saved: {cost_path.name}")
                else:
                    log("Skipping preprocessing — using existing cleaned data")
                    # Make sure the cleaned data actually exists!
                    if not cleaned_dir.exists() or not list(cleaned_dir.rglob("*.json")):
                        st.error(f"❌ Cannot skip cleaning: No cleaned .json files found in `{cleaned_dir}`.")
                        st.info("Please uncheck 'Skip cleaning' to generate the cleaned data first.")
                        st.stop()

                # Step 2: Load QA
                progress.progress(40, text="Loading QA pairs...")
                from baseline.run_baseline import load_qa_pairs, run_experiment, save_results
                qa_pairs = load_qa_pairs(limit if limit > 0 else 0)
                log(f"Loaded {len(qa_pairs)} QA pairs")

                # Step 3: Evaluate
                progress.progress(50, text="Evaluating Ground Truth...")
                gt_result = run_experiment("gt_clean_baseline", GT_DIR, qa_pairs)
                log(f"GT P@1: {gt_result['metrics']['P@1']:.4f}")

                progress.progress(65, text="Evaluating Noisy baseline...")
                noisy_result = run_experiment(noise_name, noisy_dir, qa_pairs)
                log(f"Noisy P@1: {noisy_result['metrics']['P@1']:.4f}")

                progress.progress(80, text="Evaluating NoiRAG cleaned... (This embeds 13,000 chunks if not cached!)")
                noirag_result = run_experiment(f"noirag_cleaned_{noise_name}", cleaned_dir, qa_pairs)
                log(f"NoiRAG P@1: {noirag_result['metrics']['P@1']:.4f}")

                # Calculate p-value
                progress.progress(90, text="Calculating statistical significance & emissions...")
                all_exp = [gt_result, noisy_result, noirag_result]
                
                try:
                    from scipy import stats
                    gt_mrr = gt_result["metrics"].get("mrr_scores_list", [])
                    for exp in all_exp:
                        if "gt" in exp["name"].lower(): continue
                        noisy_mrr = exp["metrics"].get("mrr_scores_list", [])
                        if gt_mrr and noisy_mrr and len(gt_mrr) == len(noisy_mrr):
                            stat, p_value = stats.ttest_rel(gt_mrr, noisy_mrr)
                            exp["metrics"]["p_value_vs_gt"] = float(p_value)
                            exp["metrics"]["is_significant_degradation"] = bool(p_value < 0.05)
                except ImportError:
                    pass

                for exp in all_exp:
                    if "mrr_scores_list" in exp["metrics"]:
                        del exp["metrics"]["mrr_scores_list"]
                        
                if tracker:
                    emissions = tracker.stop()
                    if emissions is not None:
                        log(f"🌱 Carbon Emissions: {emissions:.6f} kg CO2eq")
                        all_exp.append({
                            "name": "sustainability",
                            "metrics": {"carbon_emissions_kg_co2eq": emissions},
                            "num_queries": 0
                        })

                # Step 4: Save
                progress.progress(95, text="Saving results...")
                save_results(all_exp, filename=f"hybrid_evaluation_{noise_name}.json")

                progress.progress(100, text="✅ Complete!")
                log("🎉 Pipeline complete!")
                st.success("Pipeline finished! Check the **Benchmarks** and **Cost Report** tabs.")

            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COST REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Cost Report":
    st.markdown("# 💰 Cost & Efficiency Report")
    st.markdown("See how much money and how many API calls NoiRAG avoids.")
    st.markdown("---")

    # Find available cost reports
    cost_files = sorted(RESULTS_DIR.glob("cost_report_*.json"))

    if not cost_files:
        st.warning("No cost reports found. Run the pipeline first from the **▶️ Run Pipeline** tab.")
    else:
        selected = st.selectbox("Select Report", [f.stem.replace("cost_report_", "") for f in cost_files])
        report_path = RESULTS_DIR / f"cost_report_{selected}.json"

        with open(report_path, "r") as f:
            report = json.load(f)

        if "error" in report:
            st.error(report["error"])
        else:
            s = report["summary"]
            avoided = report["llm_calls_avoided"]
            mini = report["counterfactual_cost"]["gpt_4o_mini"]
            premium = report["counterfactual_cost"]["gpt_4o"]
            routing = report["routing_breakdown"]

            # Top metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="cost-box"><div class="cost-saved">${mini["total_cost_usd"]:.4f}</div><div class="metric-label">Saved vs GPT-4o-mini</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="cost-box"><div class="cost-saved">${premium["total_cost_usd"]:.4f}</div><div class="metric-label">Saved vs GPT-4o</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="cost-box"><div class="cost-saved">{avoided["percentage"]:.1f}%</div><div class="metric-label">API Calls Avoided</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Routing breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔀 Routing Breakdown")
                labels = {"bypassed": "Bypassed", "rule_only": "Rule-Based", "stat_only": "Statistical", "rule+stat": "Rule+Stat", "llm": "LLM"}
                for cat, data in routing.items():
                    label = labels.get(cat, cat)
                    st.markdown(f"**{label}**: {data['count']:,} chunks ({data['percentage']}%)")
                    st.progress(data['percentage'] / 100)

            with col2:
                st.markdown("### 📋 Summary")
                st.metric("Total Chunks", f"{s['total_chunks']:,}")
                st.metric("Estimated Tokens", f"{s['total_estimated_tokens']:,}")
                st.metric("NoiRAG Time", f"{s['noirag_preprocessing_time_seconds']:.1f}s")
                st.metric("NoiRAG Cost", "$0.00 (local)")
                
                eval_path = RESULTS_DIR / f"hybrid_evaluation_{selected}.json"
                if eval_path.exists():
                    with open(eval_path, "r") as f:
                        res = json.load(f)
                    for exp in res:
                        if exp["name"] == "sustainability":
                            carbon = exp["metrics"].get("carbon_emissions_kg_co2eq", 0)
                            st.metric("🌱 Carbon Emissions", f"{carbon:.6f} kg CO2eq")

            st.markdown("---")
            st.markdown("### 🌐 Network Independence")
            st.success("✅ Runs fully offline — zero external API dependencies")
            st.success("✅ No rate limits, no API keys, no data leaves your machine")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Benchmarks":
    st.markdown("# 📊 Benchmark Results")
    st.markdown("Compare retrieval performance across Ground Truth, Noisy, and NoiRAG Cleaned.")
    st.markdown("---")

    # Find available evaluation results
    eval_files = sorted(RESULTS_DIR.glob("hybrid_evaluation_*.json"))

    if not eval_files:
        st.warning("No evaluation results found. Run the pipeline first.")
    else:
        selected = st.selectbox("Select Experiment", [f.stem.replace("hybrid_evaluation_", "") for f in eval_files])
        eval_path = RESULTS_DIR / f"hybrid_evaluation_{selected}.json"

        with open(eval_path, "r") as f:
            results = json.load(f)

        # Key metrics comparison
        key_metrics = ["P@1", "MRR", "NDCG@5", "R@5"]
        
        st.markdown("### Key Metrics Comparison")
        cols = st.columns(len(key_metrics))
        for i, metric in enumerate(key_metrics):
            with cols[i]:
                st.markdown(f"**{metric}**")
                for exp in results:
                    name = exp["name"]
                    val = exp["metrics"].get(metric, 0)
                    if "gt" in name.lower():
                        icon = "🟢"
                    elif "noisy" in name.lower():
                        icon = "🔴"
                    else:
                        icon = "🔵"
                    short_name = name.split("_")[0] if "gt" in name else ("Noisy" if "noisy" in name else "NoiRAG")
                    st.markdown(f"{icon} {short_name}: **{val:.4f}**")

        st.markdown("---")

        # Recovery calculation
        if len(results) >= 3:
            gt_p1 = results[0]["metrics"].get("P@1", 0)
            noisy_p1 = results[1]["metrics"].get("P@1", 0)
            cleaned_p1 = results[2]["metrics"].get("P@1", 0)

            lost = gt_p1 - noisy_p1
            recovered = cleaned_p1 - noisy_p1
            recovery_pct = (recovered / lost * 100) if lost > 0 else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Accuracy Lost to Noise", f"{lost*100:.2f}%", delta=f"-{lost*100:.2f}%", delta_color="inverse")
            with c2:
                st.metric("Recovered by NoiRAG", f"{recovered*100:.2f}%", delta=f"+{recovered*100:.2f}%")
            with c3:
                st.metric("Recovery Rate", f"{recovery_pct:.1f}%")

        st.markdown("---")

        # Full metrics table
        st.markdown("### Full Metrics Table")
        import pandas as pd
        rows = []
        for exp in results:
            if exp["name"] == "sustainability": continue
            row = {"Condition": exp["name"], "Queries": exp.get("num_queries", 0)}
            for k, v in exp["metrics"].items():
                if k != "mrr_scores_list" and "p_value" not in k and "is_significant" not in k:
                    row[k] = round(v, 4) if isinstance(v, float) else v
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🔬 Statistical Significance (vs Ground Truth)")
        st.markdown("A p-value `< 0.05` means the degradation is statistically significant.")
        
        c1, c2 = st.columns(2)
        
        noisy_exp = next((e for e in results if "noisy" in e["name"].lower()), None)
        clean_exp = next((e for e in results if "noirag_cleaned" in e["name"].lower()), None)
        
        with c1:
            st.markdown("#### 🔴 Noisy Baseline")
            if noisy_exp and "p_value_vs_gt" in noisy_exp["metrics"]:
                p_noisy = noisy_exp["metrics"]["p_value_vs_gt"]
                is_sig = noisy_exp["metrics"]["is_significant_degradation"]
                color = "#ef4444" if is_sig else "#22c55e"
                status = "Significant Degradation" if is_sig else "No Significant Damage"
                st.markdown(f"**p-value:** `{p_noisy:.6f}`")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
            else:
                st.markdown("No p-value data available.")
                
        with c2:
            st.markdown("#### 🔵 NoiRAG Cleaned")
            if clean_exp and "p_value_vs_gt" in clean_exp["metrics"]:
                p_clean = clean_exp["metrics"]["p_value_vs_gt"]
                is_sig = clean_exp["metrics"]["is_significant_degradation"]
                color = "#ef4444" if is_sig else "#22c55e"
                status = "Still Significantly Degraded" if is_sig else "Statistically Indistinguishable from Perfect Data! 🎉"
                st.markdown(f"**p-value:** `{p_clean:.6f}`")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
            else:
                st.markdown("No p-value data available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TEXT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Text Comparison":
    st.markdown("# 🔍 Text Comparison")
    st.markdown("See the difference between Ground Truth, Noisy, and NoiRAG Cleaned text.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        noise_type = st.selectbox("Noise Type", ["semantic", "formatting"], key="cmp_type")
    with col2:
        noise_level = st.selectbox("Noise Level", [10, 25, 50, 75], key="cmp_level")

    noise_name = f"{noise_type}_{noise_level}"
    noisy_dir = NOISY_DIR / noise_name
    cleaned_dir = CLEANED_DIR / noise_name

    # Get available domains
    if GT_DIR.exists():
        domains = sorted([d.name for d in GT_DIR.iterdir() if d.is_dir()])
        domain = st.selectbox("Domain", domains)

        gt_domain = GT_DIR / domain
        noisy_domain = noisy_dir / domain
        cleaned_domain = cleaned_dir / domain

        gt_files = sorted(gt_domain.glob("*.json")) if gt_domain.exists() else []

        if gt_files:
            selected_file = st.selectbox("Document", [f.stem for f in gt_files])

            gt_path = gt_domain / f"{selected_file}.json"
            noisy_path = noisy_domain / f"{selected_file}.json"
            cleaned_path = cleaned_domain / f"{selected_file}.json"

            # Load texts
            def load_page_text(path, page_idx=0):
                if not path.exists():
                    return "(file not found)"
                with open(path, "r", encoding="utf-8") as f:
                    pages = json.load(f)
                if page_idx < len(pages):
                    return pages[page_idx].get("text", "(no text)")
                return "(page not found)"

            # Page selector
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_pages = json.load(f)
            page_idx = st.slider("Page", 0, max(0, len(gt_pages) - 1), 0)

            gt_text = load_page_text(gt_path, page_idx)
            noisy_text = load_page_text(noisy_path, page_idx)
            cleaned_text = load_page_text(cleaned_path, page_idx)

            # Side by side comparison
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### 🟢 Ground Truth")
                st.markdown(f'<div class="compare-panel panel-gt">{gt_text[:2000]}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("#### 🔴 Noisy")
                st.markdown(f'<div class="compare-panel panel-noisy">{noisy_text[:2000]}</div>', unsafe_allow_html=True)
            with c3:
                st.markdown("#### 🔵 NoiRAG Cleaned")
                st.markdown(f'<div class="compare-panel panel-clean">{cleaned_text[:2000]}</div>', unsafe_allow_html=True)

            # Quick quality score
            st.markdown("---")
            if st.button("🔬 Score This Chunk"):
                from noirag.preprocessing.hybrid.quality_scorer import QualityScorer
                scorer = QualityScorer()

                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    scores = scorer.score(gt_text)
                    st.markdown("**GT Scores**")
                    st.json(scores)
                with sc2:
                    scores = scorer.score(noisy_text)
                    st.markdown("**Noisy Scores**")
                    st.json(scores)
                with sc3:
                    scores = scorer.score(cleaned_text)
                    st.markdown("**Cleaned Scores**")
                    st.json(scores)
        else:
            st.warning(f"No documents found in {gt_domain}")
    else:
        st.error(f"Ground truth directory not found: {GT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Architecture":
    st.markdown("# ⚙️ Hybrid Triage Architecture")
    st.markdown("How NoiRAG intelligently routes each text chunk to the optimal cleaner.")
    st.markdown("---")

    st.markdown("""
    ```mermaid
    graph LR
        A[📄 Noisy Chunk] --> B[📊 Quality Scorer]
        B --> C{🔀 Hybrid Orchestrator}
        C -->|Score < 0.05| D[✅ Bypass]
        C -->|Garbage > 0.05| E[🔧 Rule-Based Cleaner]
        C -->|OOV > 0.10| F[📈 Statistical Cleaner]
        C -->|Score > 0.60| G[🤖 LLM Cleaner]
        D --> H[✅ Clean Chunk]
        E --> H
        F --> H
        G --> H
        H --> I[🔍 FAISS Index]
    ```
    """)

    st.markdown("---")
    st.markdown("### Routing Thresholds")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### ✅ Bypass")
        st.markdown("**Score < 0.05**")
        st.markdown("Clean text — skip processing entirely. Prevents accidental corruption.")
        st.markdown("**Cost:** Free")
    with c2:
        st.markdown("#### 🔧 Rule-Based")
        st.markdown("**Garbage > 0.05**")
        st.markdown("Regex fixes: garbage strings, unicode normalization, broken line merging.")
        st.markdown("**Cost:** Free, ~0.1ms")
    with c3:
        st.markdown("#### 📈 Statistical")
        st.markdown("**OOV > 0.10**")
        st.markdown("SymSpellPy edit-distance spell checking with conservative fences.")
        st.markdown("**Cost:** Free, ~1-5ms")
    with c4:
        st.markdown("#### 🤖 LLM")
        st.markdown("**Score > 0.60**")
        st.markdown("Local Ollama (qwen2.5:0.5b) for severely corrupted text. Offline.")
        st.markdown("**Cost:** Free (local)")

    st.markdown("---")
    st.markdown("### Why This Architecture Matters")
    st.info("💡 **The key insight:** 97%+ of document noise can be fixed with zero-cost local algorithms. Only the most severely corrupted 3% needs an LLM. This eliminates API costs entirely while maintaining high accuracy recovery.")

    # Interactive demo
    st.markdown("---")
    st.markdown("### 🧪 Try It Live")
    demo_text = st.text_area(
        "Paste or type noisy text to see how NoiRAG routes it:",
        value="Ths cmputer sciense papre dsicusses ■ advnaced === machin lerning ▸ techniqes for natrual ◆ languge procesing.",
        height=100,
    )

    if st.button("🔬 Analyze & Clean", type="primary"):
        from noirag.preprocessing.hybrid.quality_scorer import QualityScorer
        from noirag.preprocessing.hybrid.hybrid_cleaner import HybridCleaner

        scorer = QualityScorer()
        scores = scorer.score(demo_text)

        st.markdown("#### Quality Scores")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Overall", f"{scores['overall_score']:.4f}")
        with sc2:
            st.metric("OOV Ratio", f"{scores['oov_ratio']:.4f}")
        with sc3:
            st.metric("Garbage Density", f"{scores['garbage_density']:.4f}")
        with sc4:
            st.metric("Formatting Anomaly", f"{scores['formatting_anomaly_rate']:.4f}")

        cleaner = HybridCleaner(verbose=False)
        cleaned, metadata = cleaner.clean(demo_text)

        st.markdown("#### Routing Decision")
        applied = metadata.get("applied_cleaners", [])
        if not applied:
            st.success("✅ **Bypassed** — Text is clean enough, no processing needed.")
        else:
            for c in applied:
                colors = {"rule_based": "🔧", "statistical": "📈", "llm": "🤖"}
                st.info(f"{colors.get(c, '❓')} Applied: **{c}**")

        st.markdown("#### Result")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Before (Noisy)**")
            st.code(demo_text, language="text")
        with r2:
            st.markdown("**After (Cleaned)**")
            st.code(cleaned, language="text")


# ── Sidebar Footer ──
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Team NoiRAG**")
st.sidebar.markdown("[GitHub](https://github.com/shreyabag028/NoiRAG)")
