"""
NoiRAG Cost & Latency Profiler

Tracks per-chunk preprocessing decisions and calculates the counterfactual cost
of cleaning the same data via cloud LLM APIs (OpenAI GPT-4o-mini).

The goal is to prove, with hard numbers, how much money and how many API calls
NoiRAG avoids by using intelligent local routing instead of blindly sending
every document to a paid LLM endpoint.

Usage:
    Automatically integrated into HybridCleaner. Access via `cleaner.profiler`.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from pathlib import Path


# ── OpenAI Pricing (as of 2025) ─────────────────────────────────────────────
# Source: https://openai.com/api/pricing
# These are used ONLY for counterfactual estimation, not actual API calls.
PRICING = {
    "gpt-4o-mini": {
        "input_per_1m_tokens": 0.15,    # $0.15 per 1M input tokens
        "output_per_1m_tokens": 0.60,   # $0.60 per 1M output tokens
        "avg_latency_ms": 500,          # Conservative avg per-request latency
    },
    "gpt-4o": {
        "input_per_1m_tokens": 2.50,    # $2.50 per 1M input tokens
        "output_per_1m_tokens": 10.00,  # $10.00 per 1M output tokens
        "avg_latency_ms": 800,          # Conservative avg per-request latency
    },
}

# Standard token approximation: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChunkRecord:
    """Record of a single chunk's preprocessing pass."""
    chunk_index: int
    char_count: int
    estimated_tokens: int
    cleaners_applied: List[str]
    route_category: str       # "bypassed", "rule_only", "stat_only", "rule+stat", "llm"
    elapsed_seconds: float


class CostProfiler:
    """
    Collects per-chunk preprocessing data and generates cost/latency reports.
    
    This profiler answers the question:
    "If we had sent every single chunk to OpenAI's API for cleaning instead
     of using NoiRAG's local routing, how much would it have cost?"
    """

    def __init__(self):
        self.records: List[ChunkRecord] = []
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def start(self):
        """Mark the start of the preprocessing phase."""
        self._start_time = time.perf_counter()

    def stop(self):
        """Mark the end of the preprocessing phase."""
        self._end_time = time.perf_counter()

    def record(self, text: str, cleaners_applied: List[str], elapsed_seconds: float):
        """
        Record one chunk's preprocessing result.
        
        Args:
            text: The original (pre-cleaning) text of the chunk.
            cleaners_applied: List of cleaner names applied (e.g., ["rule_based", "statistical"]).
            elapsed_seconds: Wall-clock time taken by HybridCleaner.clean() for this chunk.
        """
        char_count = len(text)
        estimated_tokens = max(1, char_count // CHARS_PER_TOKEN)
        
        # Classify the routing decision
        route_category = self._classify_route(cleaners_applied)
        
        self.records.append(ChunkRecord(
            chunk_index=len(self.records),
            char_count=char_count,
            estimated_tokens=estimated_tokens,
            cleaners_applied=cleaners_applied,
            route_category=route_category,
            elapsed_seconds=elapsed_seconds,
        ))

    @staticmethod
    def _classify_route(cleaners: List[str]) -> str:
        """Classify a chunk's routing decision into a human-readable category."""
        if not cleaners:
            return "bypassed"
        if cleaners == ["llm"]:
            return "llm"
        if cleaners == ["rule_based"]:
            return "rule_only"
        if cleaners == ["statistical"]:
            return "stat_only"
        if "rule_based" in cleaners and "statistical" in cleaners:
            return "rule+stat"
        # Fallback for any unexpected combination
        return "+".join(cleaners)

    def _estimate_openai_cost(self, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Estimate what it would cost to clean ALL chunks via an OpenAI model.
        
        Assumptions:
            - Every chunk is sent as a separate API request.
            - Input tokens = chunk text + ~50 tokens for the system prompt.
            - Output tokens ≈ input tokens (the LLM returns the cleaned text,
              roughly the same length).
            - No batching, no caching, no parallelism (worst-case sequential).
        """
        pricing = PRICING[model]
        system_prompt_tokens = 80  # Our system prompt is ~320 chars ≈ 80 tokens
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        for rec in self.records:
            input_tokens = rec.estimated_tokens + system_prompt_tokens
            output_tokens = rec.estimated_tokens  # Output ≈ same length as input
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        
        input_cost = (total_input_tokens / 1_000_000) * pricing["input_per_1m_tokens"]
        output_cost = (total_output_tokens / 1_000_000) * pricing["output_per_1m_tokens"]
        total_cost = input_cost + output_cost
        
        # Estimated wall time if every chunk was a sequential API call
        total_api_calls = len(self.records)
        estimated_wall_time_s = (total_api_calls * pricing["avg_latency_ms"]) / 1000
        
        return {
            "model": model,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "total_api_calls_required": total_api_calls,
            "estimated_wall_time_seconds": round(estimated_wall_time_s, 1),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate the full cost & latency report as a dictionary."""
        if not self.records:
            return {"error": "No chunks were profiled."}
        
        total_chunks = len(self.records)
        total_chars = sum(r.char_count for r in self.records)
        total_tokens = sum(r.estimated_tokens for r in self.records)
        total_noirag_time = self._end_time - self._start_time if self._end_time else 0.0
        
        # Routing breakdown
        route_counts: Dict[str, int] = {}
        for rec in self.records:
            route_counts[rec.route_category] = route_counts.get(rec.route_category, 0) + 1
        
        # LLM calls avoided = everything that was NOT routed to LLM
        llm_chunks = route_counts.get("llm", 0)
        llm_avoided = total_chunks - llm_chunks
        llm_avoided_pct = (llm_avoided / total_chunks * 100) if total_chunks else 0
        
        # Cost estimates for both models
        cost_4o_mini = self._estimate_openai_cost("gpt-4o-mini")
        cost_4o = self._estimate_openai_cost("gpt-4o")
        
        return {
            "summary": {
                "total_chunks": total_chunks,
                "total_characters": total_chars,
                "total_estimated_tokens": total_tokens,
                "noirag_preprocessing_time_seconds": round(total_noirag_time, 2),
                "noirag_cost_usd": 0.00,
            },
            "routing_breakdown": {
                category: {
                    "count": count,
                    "percentage": round(count / total_chunks * 100, 1),
                }
                for category, count in sorted(route_counts.items())
            },
            "llm_calls_avoided": {
                "avoided": llm_avoided,
                "total": total_chunks,
                "percentage": round(llm_avoided_pct, 1),
            },
            "counterfactual_cost": {
                "gpt_4o_mini": cost_4o_mini,
                "gpt_4o": cost_4o,
            },
            "savings": {
                "vs_gpt_4o_mini_usd": round(cost_4o_mini["total_cost_usd"], 6),
                "vs_gpt_4o_usd": round(cost_4o["total_cost_usd"], 6),
            },
        }

    def print_report(self):
        """Print a formatted terminal report."""
        report = self.generate_report()
        
        if "error" in report:
            print(f"\n⚠️  Profiler: {report['error']}")
            return
        
        s = report["summary"]
        r = report["routing_breakdown"]
        avoided = report["llm_calls_avoided"]
        mini = report["counterfactual_cost"]["gpt_4o_mini"]
        premium = report["counterfactual_cost"]["gpt_4o"]
        
        w = 64  # Box width
        
        print()
        print("╔" + "═" * w + "╗")
        print("║" + "NoiRAG Cost & Efficiency Report".center(w) + "║")
        print("╠" + "═" * w + "╣")
        
        # ── Chunk Summary
        print("║" + f"  Total Chunks Processed:  {s['total_chunks']:>10,}".ljust(w) + "║")
        print("║" + f"  Total Tokens (est.):     {s['total_estimated_tokens']:>10,}".ljust(w) + "║")
        print("║" + f"  NoiRAG Processing Time:  {s['noirag_preprocessing_time_seconds']:>8.1f}s".ljust(w) + "║")
        print("║" + "─" * w + "║")
        
        # ── Routing Breakdown
        print("║" + "  Routing Breakdown:".ljust(w) + "║")
        
        labels = {
            "bypassed": "Bypassed (clean)",
            "rule_only": "Rule-Based only",
            "stat_only": "Statistical only",
            "rule+stat": "Rule + Statistical",
            "llm": "LLM (severe)",
        }
        
        for category, data in r.items():
            label = labels.get(category, category)
            line = f"    {label:<22} {data['count']:>6,}  ({data['percentage']:>5.1f}%)"
            print("║" + line.ljust(w) + "║")
        
        print("║" + "─" * w + "║")
        
        # ── API Calls Avoided
        print("║" + f"  🚫 LLM API Calls Avoided:".ljust(w) + "║")
        avoided_line = f"    {avoided['avoided']:,} / {avoided['total']:,}  ({avoided['percentage']:.1f}%)"
        print("║" + f"    {avoided_line}".ljust(w) + "║")
        
        print("║" + "─" * w + "║")
        
        # ── Cost Comparison
        print("║" + "  💰 Cost Comparison (if all chunks sent to LLM):".ljust(w) + "║")
        print("║" + f"    GPT-4o-mini would cost:   ${mini['total_cost_usd']:>8.4f}".ljust(w) + "║")
        print("║" + f"    GPT-4o would cost:        ${premium['total_cost_usd']:>8.4f}".ljust(w) + "║")
        print("║" + f"    NoiRAG actual cost:        ${'0.0000':>8}  (local)".ljust(w) + "║")
        
        print("║" + "─" * w + "║")
        
        # ── Network Independence
        print("║" + "  📡 Network Independence:".ljust(w) + "║")
        print("║" + "    ✅ Runs fully offline — zero external API dependencies".ljust(w) + "║")
        print("║" + "    ✅ No rate limits, no API keys, no data leaves machine".ljust(w) + "║")
        
        print("╚" + "═" * w + "╝")
        print()

    def save_report(self, path: Path):
        """Save the full report as a JSON file."""
        report = self.generate_report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Cost report saved: {path}")
