"""
Shared LLM client, used by every module that needs to call an LLM
(answer generation, HyDE query rewriting, groundedness checking).

Centralizing this in one place means a provider swap (this project has
already gone Gemini -> Anthropic -> Gemini -> Groq, due to free-tier rate
limits vs. billing requirements vs. free-tier limits still being too tight
even with pacing + backoff) is a one-file change, not a find-and-replace
across the codebase.

Groq and Gemini are both wired up here as interchangeable providers,
selected via LLM_PROVIDER (src/config.py, or the LLM_PROVIDER env var —
switching is a restart, not a code change). Both free tiers are tight
enough that hitting one's rate limit mid-session is a real scenario for
this project: when that happens, flip LLM_PROVIDER to the other one.

Both providers turned out to have the same failure mode under the hood:
they're reasoning models that can spend their whole output-token budget on
hidden reasoning and return empty/truncated content for longer-output
tasks (groundedness's per-claim report, in particular) unless reasoning
effort is explicitly turned down — see GROQ_REASONING_EFFORT and
GEMINI_THINKING_LEVEL in config.py.
"""

import time
import logging

import groq
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
    LLM_MAX_ATTEMPTS,
    LLM_RATE_LIMIT_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

_groq_client = None
_gemini_client = None


class LLMRateLimitedError(Exception):
    """Raised when the active provider is still rate-limited after the one
    quick retry — callers (the API layer, in particular) should catch this
    specifically and surface a friendly "try again shortly" message rather
    than a generic 500, instead of leaving the caller hanging on repeated
    exponential-backoff retries."""


class _RateLimited(Exception):
    """Internal signal from a provider call to call_llm()'s retry loop —
    keeps the retry logic identical across providers instead of
    duplicating it in each _call_* function."""


def get_groq_client() -> groq.Groq:
    """Lazily construct the Groq client once and reuse it (avoids re-init
    overhead if this is called from many places)."""
    global _groq_client
    if _groq_client is None:
        _groq_client = groq.Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client


def _call_groq(prompt: str, model: str, max_tokens: int) -> str:
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=GROQ_REASONING_EFFORT,
            messages=[{"role": "user", "content": prompt}],
        )
    except groq.RateLimitError:
        raise _RateLimited() from None
    return (response.choices[0].message.content or "").strip()


def _call_gemini(prompt: str, model: str, max_tokens: int) -> str:
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_level=genai_types.ThinkingLevel[GEMINI_THINKING_LEVEL],
                ),
            ),
        )
    except genai_errors.ClientError as e:
        if e.code == 429:
            raise _RateLimited() from None
        raise
    return (response.text or "").strip()


_PROVIDERS = {
    "groq": (_call_groq, GROQ_MODEL),
    "gemini": (_call_gemini, GEMINI_MODEL),
}


def call_llm(prompt: str, model: str | None = None, max_tokens: int = 1024) -> str:
    """
    Call the active provider (LLM_PROVIDER). On a rate limit: one quick
    retry after a short fixed delay, then raise LLMRateLimitedError —
    deliberately not exponential backoff. A live demo request shouldn't
    hang for minutes; failing fast lets the caller show a friendly message
    (or a human flip LLM_PROVIDER) instead.

    `model` overrides the active provider's default model if given.
    """
    call_fn, default_model = _PROVIDERS[LLM_PROVIDER]
    model = model or default_model

    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            content = call_fn(prompt, model, max_tokens)
        except _RateLimited:
            if attempt + 1 >= LLM_MAX_ATTEMPTS:
                raise LLMRateLimitedError(
                    f"{LLM_PROVIDER} is rate-limited — try again shortly, or switch LLM_PROVIDER."
                ) from None
            logger.warning(
                "Rate limited (%s), retrying once in %ss...",
                LLM_PROVIDER, LLM_RATE_LIMIT_RETRY_DELAY_SECONDS,
            )
            time.sleep(LLM_RATE_LIMIT_RETRY_DELAY_SECONDS)
            continue

        if not content:
            raise RuntimeError(
                f"LLM returned empty content (provider={LLM_PROVIDER}, model={model}, "
                f"max_tokens={max_tokens}) — likely exhausted its token budget on reasoning; "
                "consider raising max_tokens."
            )
        return content
