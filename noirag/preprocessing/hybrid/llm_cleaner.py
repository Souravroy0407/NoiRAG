"""
LLM Cleaner for NoiRAG using Local Ollama Engine.
Uses a local, lightweight LLM to fix heavily corrupted document chunks.
This acts as the final firewall in the Hybrid Orchestration pipeline with zero rate limits.
"""
import requests

class LLMCleaner:
    def __init__(self, model: str = "qwen2.5:0.5b"):
        """
        Args:
            model: The local Ollama model ID to use.
        """
        self.model = model
        # Ollama's local default port
        self.api_url = "http://localhost:11434/api/chat"
        
        self.system_prompt = (
            "You are a strict, automated OCR-repair engine. Your ONLY purpose is to fix OCR errors, typos, "
            "and formatting damage (like broken lines or garbage characters) in the user's text. "
            "CRITICAL INSTRUCTIONS: "
            "1. Output ONLY the perfectly repaired text. "
            "2. DO NOT add any conversational padding like 'Here is the fixed text:'. "
            "3. DO NOT change the meaning, tone, or restructure paragraphs unnecessarily. "
            "4. Leave valid proper nouns and acronyms completely untouched."
        )

    def clean(self, text: str) -> str:
        """
        Executes the Ollama prompt to fix the text locally.
        """
        if not text.strip():
            return text
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": f"{self.system_prompt}\n\nPlease repair this raw OCR extraction:\n\n{text}"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0 # We want deterministic, safe output
            } 
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            
            if response.status_code == 200:
                data = response.json()
                cleaned_text = data['message']['content'].strip()
                return cleaned_text
            else:
                print(f"Local LLM Error: {response.status_code} - {response.text}")
                return text
                
        except requests.exceptions.RequestException as e:
            # If Ollama is turned off or crashed, return original text safely
            print(f"Failed to connect to local Ollama instance. Is Ollama running? Error: {e}")
            return text
