"""
LLM Cleaner for NoiRAG — Dual-Backend Support (Ollama / OpenAI)

Supports two backends for text repair:
  1. "ollama"  → Free, local, offline. Uses qwen2.5:0.5b.
  2. "openai"  → Paid, cloud-based. Uses gpt-4o-mini.

To switch: Change the single variable below ↓
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ╔══════════════════════════════════════════════════════════════╗
# ║  SWITCH BACKEND HERE — Change "ollama" to "openai" or "groq" ║
# ╚══════════════════════════════════════════════════════════════╝
BACKEND = "groq"
# ════════════════════════════════════════════════════════════════


class LLMCleaner:

    # Backend configurations
    BACKENDS = {
        "ollama": {
            "model": "qwen2.5:0.5b",
            "url": "http://localhost:11434/api/chat",
        },
        "openai": {
            "model": "gpt-4o-mini",
            "url": "https://api.openai.com/v1/chat/completions",
        },
        "groq": {
            "model": "llama-3.1-8b-instant",
            "url": "https://api.groq.com/openai/v1/chat/completions",
        },
    }

    def __init__(self, backend: str = BACKEND):
        """
        Args:
            backend: "ollama" or "openai"
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'ollama' or 'openai'.")

        self.backend = backend
        config = self.BACKENDS[backend]
        self.model = config["model"]
        self.api_url = config["url"]

        # Only needed for OpenAI or Groq
        self.api_key = None
        if backend == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                print("⚠ WARNING: OPENAI_API_KEY not found in .env file!")
        elif backend == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                print("⚠ WARNING: GROQ_API_KEY not found in .env file!")

        self.system_prompt = (
            "You are a strict, automated OCR-repair engine. Your ONLY purpose is to fix OCR errors, typos, "
            "and formatting damage (like broken lines or garbage characters) in the user's text. "
            "CRITICAL INSTRUCTIONS: "
            "1. Output ONLY the perfectly repaired text. "
            "2. DO NOT add any conversational padding like 'Here is the fixed text:'. "
            "3. DO NOT change the meaning, tone, or restructure paragraphs unnecessarily. "
            "4. Leave valid proper nouns and acronyms completely untouched."
        )

        print(f"LLM Cleaner initialized → Backend: {self.backend} | Model: {self.model}")

    def clean(self, text: str) -> str:
        """Send the text to the active LLM backend for repair."""
        if not text.strip():
            return text

        try:
            if self.backend == "ollama":
                return self._clean_ollama(text)
            elif self.backend in ["openai", "groq"]:
                return self._clean_openai(text) # Groq uses the exact same OpenAI API structure
        except requests.exceptions.RequestException as e:
            print(f"LLM Cleaner [{self.backend}] connection failed: {e}")
            return text

    # ── Ollama Backend ───────────────────────────────────────
    def _clean_ollama(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": f"{self.system_prompt}\n\nPlease repair this raw OCR extraction:\n\n{text}"}
            ],
            "stream": False,
            "options": {"temperature": 0.0}
        }
        response = requests.post(self.api_url, json=payload, timeout=1200)
        if response.status_code == 200:
            return response.json()['message']['content'].strip()
        else:
            print(f"Ollama Error: {response.status_code} - {response.text}")
            return text

    # ── OpenAI/Groq Backend ───────────────────────────────────────
    def _clean_openai(self, text: str) -> str:
        import time
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Please repair this raw OCR extraction:\n\n{text}"}
            ],
            "temperature": 0.0,
            "max_tokens": 2048
        }
        
        max_retries = 5
        for attempt in range(max_retries):
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            elif response.status_code == 429:
                print(f"API Error 429: Rate limit reached. Sleeping for 35 seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(35)
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return text
                
        print("Max retries reached. Returning original uncleaned text.")
        return text
