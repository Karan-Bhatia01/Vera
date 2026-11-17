"""
llm_provider.py
================
Single shared Groq client used by the agents and the insight explainers.
Plain chat completions with rate-limit retries — no LangChain, no CrewAI.
"""

import os
import re
import time

from src.core.connections import get_groq_client
from src.logger import logging

# Default model, overridable via env or per call.
_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Hard ceiling on any single LLM call so a stalled request can never hang a
# background job forever — callers fall back to defaults when this fires.
_DEFAULT_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "45"))

_RATE_LIMIT_RETRIES = int(os.environ.get("GROQ_RATE_LIMIT_RETRIES", "4"))
_MAX_RETRY_WAIT = 15.0  # never sleep longer than this on a single backoff


def _retry_after_seconds(message: str) -> float | None:
    """Pull the 'try again in 1.774s' hint out of a Groq 429 message, if present."""
    m = re.search(r"try again in ([\d.]+)\s*s", message, re.IGNORECASE)
    return float(m.group(1)) if m else None


def chat_completion(messages: list, model: str | None = None, timeout: float = _DEFAULT_TIMEOUT, **kwargs):
    """Send one chat request and return the reply message.

    Groq rate limits (HTTP 429) are retried with backoff, honoring the server's
    suggested wait. Any other error is raised so the caller can fall back.
    """
    client = get_groq_client(timeout=timeout)

    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model or _MODEL,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )
            return response.choices[0].message
        except Exception as e:
            msg = str(e)
            is_rate_limit = "rate_limit" in msg.lower() or "429" in msg
            if not is_rate_limit or attempt >= _RATE_LIMIT_RETRIES:
                raise

            wait = _retry_after_seconds(msg)
            wait = min(_MAX_RETRY_WAIT, wait + 0.5 if wait else 2.0 * (2 ** attempt))
            logging.warning(
                "Groq rate-limited (attempt %d/%d) — retrying in %.1fs",
                attempt + 1, _RATE_LIMIT_RETRIES, wait,
            )
            time.sleep(wait)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    model: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_tokens: int = 1024,
) -> str:
    """Single system + user completion. Returns the raw text response.

    Give prompts that must return a large JSON object a generous `max_tokens`,
    otherwise the reply is cut off mid-object and fails to parse.
    """
    message = chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return message.content.strip()
