"""
llm_provider.py
================
Single shared Groq client used by all agents.
Uses the openai-compatible Groq API directly — no LangChain, no tool loops.
One call per decision, fast.
"""

import os
import json
import re
import time

from groq import Groq

from src.logger import logging


# Default model. `groq/compound` is Groq's agentic system (multi-step tool use) —
# powerful but slow. Overridable per call via the `model` arg; a fast chat model
# like "llama-3.3-70b-versatile" is far quicker for plain JSON generation.
_MODEL = os.environ.get("GROQ_MODEL", "groq/compound")

# Hard ceiling on any single LLM call so a stalled request can never hang a
# background job forever — callers fall back to defaults when this fires.
_DEFAULT_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "45"))


def get_groq_client(timeout: float = _DEFAULT_TIMEOUT) -> Groq:
    return Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=timeout, max_retries=2)


_RATE_LIMIT_RETRIES = int(os.environ.get("GROQ_RATE_LIMIT_RETRIES", "4"))
_MAX_RETRY_WAIT = 15.0  # never sleep longer than this on a single backoff


def _retry_after_seconds(message: str) -> float | None:
    """Pull the 'try again in 1.774s' hint out of a Groq 429 message, if present."""
    m = re.search(r"try again in ([\d.]+)\s*s", message, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    model: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_tokens: int = 1024,
) -> str:
    """Single completion call. Returns the raw text response.

    `max_tokens` caps the response length — set it generously for prompts that
    must return a large JSON object, otherwise the reply is truncated mid-object
    and fails to parse.

    Transient Groq rate limits (HTTP 429 / tokens-per-minute) are retried with
    backoff, honoring the server's suggested wait when given. Other errors (and
    exhausted retries) raise — callers should catch and fall back so a degraded
    LLM never hangs or hard-fails the request.
    """
    client = get_groq_client(timeout=timeout)
    last_err: Exception | None = None

    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model or _MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 — classify by message below
            last_err = e
            msg = str(e)
            is_rate_limit = "rate_limit" in msg or "429" in msg or "Rate limit" in msg
            if not is_rate_limit or attempt >= _RATE_LIMIT_RETRIES:
                raise
            # Honor the suggested wait, else exponential backoff. Add a small
            # margin so the TPM window has actually rolled over.
            wait = _retry_after_seconds(msg)
            wait = min(_MAX_RETRY_WAIT, (wait + 0.5) if wait else 2.0 * (2 ** attempt))
            logging.warning(
                "Groq rate-limited (attempt %d/%d) — retrying in %.1fs",
                attempt + 1, _RATE_LIMIT_RETRIES, wait,
            )
            time.sleep(wait)

    # Unreachable in practice, but keep the type checker happy.
    raise last_err  # type: ignore[misc]


def parse_json_from_response(text: str) -> dict | list:
    """
    Extract JSON from an LLM response that may include markdown fences
    or surrounding prose.
    """
    text = text.strip()

    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find first { or [ and last matching } or ]
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return {}