"""
HypothesisForge - Environment-Driven LLM Connector
Loads LLM configuration from .env / environment variables for Gemini API, local Ollama, or Rule-Based mode.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import requests
from .prompt_engine.runner import LLMCallable


def _load_dotenv(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Reads key=value pairs from a .env file without external dependencies."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    values: Dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def build_gemini_callable(api_key: str, model: str = "gemini-2.5-flash") -> LLMCallable:
    """Builds a callable for the Google Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def call_gemini(system_prompt: str, user_prompt: str) -> str:
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        resp = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    return call_gemini


def build_ollama_callable(base_url: str = "http://localhost:11434", model: str = "llama3.1") -> LLMCallable:
    """Builds a callable for a local Ollama server."""
    url = f"{base_url.rstrip('/')}/api/chat"

    def call_ollama(system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    return call_ollama


def get_llm_callable(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
) -> Optional[LLMCallable]:
    """Returns an LLMCallable based on explicit arguments, or None for Rule-Based mode."""
    p = provider.lower().strip()
    if p in {"gemini", "gemini api"} and api_key and api_key.strip():
        return build_gemini_callable(api_key=api_key.strip(), model=model or "gemini-2.5-flash")
    if p in {"ollama", "ollama (local)"}:
        return build_ollama_callable(base_url=ollama_url, model=model or "llama3.1")
    return None


def get_llm_callable_from_env(env_path: Optional[Path] = None) -> Tuple[str, Optional[LLMCallable]]:
    """
    Reads .env and os.environ to configure the active LLM provider cleanly without UI secrets.
    Returns (display_mode_label, llm_callable_or_none).
    """
    dotenv = _load_dotenv(env_path)

    def get_var(name: str, default: str = "") -> str:
        return os.environ.get(name, dotenv.get(name, default)).strip()

    provider = get_var("LLM_PROVIDER", "rule_based").lower()
    gemini_key = get_var("GEMINI_API_KEY", "")
    gemini_model = get_var("GEMINI_MODEL", "gemini-2.5-flash")
    ollama_url = get_var("OLLAMA_URL", "http://localhost:11434")
    ollama_model = get_var("OLLAMA_MODEL", "llama3.1")

    if provider == "gemini" and gemini_key:
        return f"Gemini ({gemini_model})", build_gemini_callable(api_key=gemini_key, model=gemini_model)
    if provider == "ollama":
        return f"Ollama ({ollama_model})", build_ollama_callable(base_url=ollama_url, model=ollama_model)
    return "Rule-Based Analysis", None
