"""
Shared LLM client, used by every module that needs to call an LLM
(answer generation, HyDE query rewriting, groundedness checking).

Centralizing this in one place means a provider swap (this project has
already gone Gemini -> Anthropic -> Gemini -> Groq, due to free-tier rate
limits vs. billing requirements vs. free-tier limits still being too tight
even with pacing + backoff) is a one-file change, not a find-and-replace
across the codebase.

Groq: OpenAI-compatible chat completions API, hosts open models
(GPT-OSS 120B here) at very fast inference, with a free tier that's
meaningfully more generous than Gemini's for this project's call volume.
"""

import time
import logging

import groq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    LLM_MAX_RETRIES,
    LLM_BASE_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)

_client = None


def get_client() -> groq.Groq:
    """Lazily construct the Groq client once and reuse it (avoids re-init
    overhead if this is called from many places)."""
    global _client
    if _client is None:
        _client = groq.Groq(api_key=GROQ_API_KEY)
    return _client


def call_llm(prompt: str, model: str = GROQ_MODEL, max_tokens: int = 1024) -> str:
    """
    Call the LLM with retry/backoff on rate limits.

    Groq's free tier limits are per-model (requests/min and tokens/min);
    this retry loop handles transient 429s the same way it did for
    Gemini's tighter free tier, though it's expected to trigger far less
    often given Groq's more generous limits for this project's volume.
    """
    client = get_client()

    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                reasoning_effort=GROQ_REASONING_EFFORT,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError(
                    f"LLM returned empty content (model={model}, max_tokens={max_tokens}) — "
                    "likely exhausted its token budget on reasoning; consider raising max_tokens."
                )
            return content
        except groq.RateLimitError:
            wait = LLM_BASE_BACKOFF_SECONDS * (attempt + 1)
            logger.warning(
                "Rate limited, waiting %ss (attempt %s/%s)...",
                wait, attempt + 1, LLM_MAX_RETRIES,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"LLM call failed after {LLM_MAX_RETRIES} retries — check quota/billing."
    )