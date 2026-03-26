"""
Extract QA pairs for finance, paper (academic), textbook domains
from OHR-Bench.parquet and save as clean .jsonl files.
Run from project root: python extract_qa.py
"""

import pandas as pd
import json
from pathlib import Path

PARQUET_PATH = "data/raw_pdfs/OHR-Bench.parquet"
OUTPUT_DIR   = Path("data/qa")

# Maps parquet domain name → our standardized name
DOMAIN_MAP = {
    "finance":  "finance",
    "paper":    "academic",
    "textbook": "textbook",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)
df_filtered = df[df["domain"].isin(DOMAIN_MAP.keys())]

print(f"Total rows for selected domains: {len(df_filtered)}")

domain_stats = {}

for parquet_domain, our_domain in DOMAIN_MAP.items():
    domain_rows = df_filtered[df_filtered["domain"] == parquet_domain]
    output_path = OUTPUT_DIR / f"{our_domain}.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in domain_rows.iterrows():
            qas = row["qas"]
            if qas is None:
                continue
            if isinstance(qas, str):
                qas = json.loads(qas)

            questions    = qas.get("questions", [])
            answers      = qas.get("answers", [])
            contexts     = qas.get("evidence_context", [])
            page_nos     = qas.get("evidence_page_no", [])
            ids          = qas.get("ID", [])
            answer_forms = qas.get("answer_form", [])
            ev_sources   = qas.get("evidence_source", [])

            for i in range(len(questions)):
                entry = {
                    "id":               str(ids[i]) if i < len(ids) else "",
                    "doc_name":         row["doc_name"],
                    "domain":           our_domain,
                    "question":         str(questions[i]),
                    "answer":           str(answers[i]) if i < len(answers) else "",
                    "evidence_context": str(contexts[i]) if i < len(contexts) else "",
                    "evidence_page_no": int(page_nos[i]) if i < len(page_nos) else -1,
                    "answer_form":      str(answer_forms[i]) if i < len(answer_forms) else "",
                    "evidence_source":  str(ev_sources[i]) if i < len(ev_sources) else "",
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

    domain_stats[our_domain] = count
    print(f"  ✅ {our_domain}.jsonl → {count} QA pairs saved to {output_path}")

print(f"\nDone! Total QA pairs extracted: {sum(domain_stats.values())}")
print(f"Files saved in: {OUTPUT_DIR.resolve()}")
