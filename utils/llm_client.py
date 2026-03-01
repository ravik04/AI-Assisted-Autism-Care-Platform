"""
LLM Client — OpenAI GPT-4o / GPT-4o-mini wrapper with retry & fallback.
=========================================================================
Provides a unified interface for LLM calls used by Clinical and Therapy agents.
Reads OPENAI_API_KEY from environment variable.
"""

import os
import json
import time
from typing import Optional, Dict, Any

# Load .env file (OPENAI_API_KEY)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Try to import openai
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ── Configuration ──────────────────────────────────────────────────────
DEFAULT_MODEL = "gpt-4o-mini"          # cost-efficient; swap to "gpt-4o" for higher quality
MAX_RETRIES = 2
RETRY_DELAY = 1.0                      # seconds between retries
TEMPERATURE = 0.3                      # low temp → deterministic clinical output
MAX_TOKENS = 1500


def _get_client() -> Optional[Any]:
    """Create OpenAI client if API key is available."""
    if not HAS_OPENAI:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def llm_generate(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    json_mode: bool = False,
) -> Optional[str]:
    """
    Call the OpenAI Chat Completions API.

    Returns the assistant message content, or None if the call fails
    (no API key, network error, etc.). Callers should implement fallback logic.
    """
    client = _get_client()
    if client is None:
        return None

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [LLM] Attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return None


def llm_generate_json(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Optional[Dict]:
    """Call LLM with JSON mode enabled. Returns parsed dict or None."""
    raw = llm_generate(
        system_prompt, user_prompt,
        model=model, temperature=temperature,
        max_tokens=max_tokens, json_mode=True,
    )
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        import re
        m = re.search(r'```(?:json)?\s*([\s\S]+?)```', raw)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        print(f"  [LLM] Failed to parse JSON response")
        return None


def is_llm_available() -> bool:
    """Check whether an LLM backend is configured and reachable."""
    return _get_client() is not None
