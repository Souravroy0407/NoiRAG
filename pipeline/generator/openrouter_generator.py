"""
pipeline/generator/openrouter_generator.py

Takes retrieved chunks + a user query, builds a prompt,
and calls Llama-3-8B-Instruct via the OpenRouter API
to generate a grounded answer.

Requires:
    OPENROUTER_API_KEY  in .env or environment
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL           = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "meta-llama/llama-3-8b-instruct"
MAX_TOKENS         = 512
TEMPERATURE        = 0.1   # low = more factual
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are a precise, factual question-answering assistant. "
    "Answer the question using ONLY the provided context. "
    "If the context does not contain enough information, say "
    "\"I cannot answer this based on the provided context.\" "
    "Be concise and direct."
)


def build_prompt(query: str, context_chunks: List[str]) -> str:
    """Build the user prompt from query + retrieved chunks."""
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        f"### Context\n{context_block}\n\n"
        f"### Question\n{query}\n\n"
        f"### Answer"
    )


def generate(
    query: str,
    context_chunks: List[str],
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    """
    Generate an answer using OpenRouter.

    Returns:
        {
            "answer":       str,   # generated answer text
            "model":        str,   # model used
            "prompt_tokens": int,  # tokens used for prompt
            "completion_tokens": int,  # tokens used for completion
            "total_tokens": int,
        }
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    user_prompt = build_prompt(query, context_chunks)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    response = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    choice = data["choices"][0]["message"]["content"].strip()
    usage  = data.get("usage", {})

    return {
        "answer":            choice,
        "model":             data.get("model", model),
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens", 0),
    }


# ── CLI (quick test) ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenRouter generator")
    parser.add_argument("query", help="Question to answer")
    parser.add_argument(
        "--context", nargs="+", default=["No context provided."],
        help="Context chunks (space-separated strings)",
    )
    parser.add_argument("--model",       default=MODEL)
    parser.add_argument("--max-tokens",  type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Query: {args.query}")
    print(f"Context chunks: {len(args.context)}\n")

    result = generate(
        query=args.query,
        context_chunks=args.context,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"Answer: {result['answer']}")
    print(f"\nTokens — prompt: {result['prompt_tokens']}, "
          f"completion: {result['completion_tokens']}, "
          f"total: {result['total_tokens']}")
