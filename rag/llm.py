"""
One small wrapper so the rest of the code doesn't care which LLM is used.

Provider is picked from .env:
  LLM_PROVIDER=claude  -> Anthropic Claude (needs ANTHROPIC_API_KEY)
  LLM_PROVIDER=groq    -> Groq Llama        (needs GROQ_API_KEY)
  not set              -> Claude if ANTHROPIC_API_KEY exists, else Groq if GROQ_API_KEY exists,
                          else no LLM (the engine then answers in "extractive" mode)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

log = logging.getLogger("llm")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_ANSWER_MODEL", "llama-3.3-70b-versatile")


def _pick_provider():
    wanted = os.getenv("LLM_PROVIDER", "").strip().lower()
    if wanted == "claude" or (not wanted and ANTHROPIC_API_KEY):
        return "claude" if ANTHROPIC_API_KEY else None
    if wanted == "groq" or (not wanted and GROQ_API_KEY):
        return "groq" if GROQ_API_KEY else None
    return None


PROVIDER = _pick_provider()
_client = None


def _get_client():
    global _client
    if _client is None:
        if PROVIDER == "claude":
            import anthropic
            _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif PROVIDER == "groq":
            from groq import Groq
            _client = Groq(api_key=GROQ_API_KEY)
    return _client


def available():
    return PROVIDER is not None


def describe():
    if PROVIDER == "claude":
        return f"Claude ({CLAUDE_MODEL})"
    if PROVIDER == "groq":
        return f"Groq ({GROQ_MODEL})"
    return "no LLM key - extractive mode"


def parts():
    """(provider, model) for display, e.g. ("Claude", "claude-opus-5-5"); (None, None) without a key."""
    if PROVIDER == "claude":
        return "Claude", CLAUDE_MODEL
    if PROVIDER == "groq":
        return "Groq", GROQ_MODEL
    return None, None


def chat(system, user, max_tokens=1500):
    """Send one system + user message, return the reply text ('' on failure)."""
    client = _get_client()
    if client is None:
        return ""
    try:
        if PROVIDER == "claude":
            # Server-side fallback re-runs the request on another model if a safety classifier declines it.
            response = client.beta.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if response.stop_reason == "refusal":
                log.warning("Claude declined the request: %s", response.stop_details)
                return ""
            return "".join(b.text for b in response.content if b.type == "text").strip()

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:  # network, rate limit, bad key - caller falls back
        log.warning("LLM call failed (%s): %s", PROVIDER, e)
        return ""
