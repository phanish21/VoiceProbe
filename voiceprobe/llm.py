import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class MockLLM:
    """Fake model for offline tests. Returns canned replies."""

    name = "mock"

    def chat(self, system: str, messages: list[dict]) -> str:
        time.sleep(0.05)
        last = messages[-1]["content"].lower() if messages else ""
        if "first in first out" in last:
            return "Hmm, are you sure? Think about which element a stack removes first."
        if "last in first out" in last or "lifo" in last:
            return "Correct. Can you give a real-world example of a stack?"
        return "Welcome! Let's start: what is a stack in computer science?"

class GroqLLM:
    """Real model via Groq's free tier (OpenAI-compatible API)."""

    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model="openai/gpt-oss-120b", cache_dir=".cache", retries=3, use_cache=True):
        import httpx  # imported here so mock mode never needs it

        self.httpx = httpx
        self.model = model
        self.retries = retries
        self.use_cache = use_cache
        self.key = os.environ["GROQ_API_KEY"]
        self.cache = Path(cache_dir)
        self.cache.mkdir(exist_ok=True)

    def chat(self, system: str, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_file = self.cache / f"{key}.json"
        if self.use_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

        for attempt in range(self.retries):
            r = self.httpx.post(
                self.URL,
                json=payload,
                timeout=30,
                headers={"Authorization": f"Bearer {self.key}"},
            )
            if r.status_code == 429:  # rate limited: wait, then retry
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            if self.use_cache:
                cache_file.write_text(json.dumps({"text": text}), encoding="utf-8")
            return text
        raise RuntimeError("Rate limited after retries")


def get_llm(mock: bool, use_cache: bool = True):
    return MockLLM() if mock else GroqLLM(use_cache=use_cache)