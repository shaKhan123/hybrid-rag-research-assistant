"""
Groundedness checking: verify a generated answer's claims are actually
supported by the retrieved sources, not hallucinated.

Validated earlier in this project against a deliberately fabricated
answer (containing a fake method name, a fabricated benchmark statistic,
and an invented technical requirement) — the checker correctly flagged
all four unsupported claims individually, each with a specific reason,
not just a single pass/fail verdict.

The prompt requires an explicit final "VERDICT: GROUNDED" or
"VERDICT: NOT GROUNDED" line — this is what makes is_grounded a reliable
machine-readable signal for the LangGraph retry-loop routing decision,
rather than something we'd have to parse out of free-text explanation.
"""

from src.generation.llm_client import call_llm
from src.generation.prompts import build_source_context

_GROUNDEDNESS_PROMPT_TEMPLATE = """You are a fact-checker. Given the sources and a generated answer, \
identify any claims in the answer that are NOT directly supported by the sources. \
Be strict — if a claim is a reasonable inference but not explicitly stated, flag it too.

Sources:
{context}

Answer to check:
{answer}

For each claim in the answer, respond with:
- SUPPORTED: [claim] — [which source supports it]
- UNSUPPORTED: [claim] — [why it's not backed by the sources]

End your response with exactly one line: either "VERDICT: GROUNDED" or "VERDICT: NOT GROUNDED"."""


def _parse_verdict(report: str) -> bool:
    """Extract the grounded/not-grounded verdict from the LAST line of the
    report, rather than searching the whole text — a substring search for
    "GROUNDED" would false-positive on "UNSUPPORTED" claim lines."""
    last_line = report.strip().splitlines()[-1].upper()
    return "NOT GROUNDED" not in last_line and "GROUNDED" in last_line


def check_groundedness(answer: str, chunks: list[dict]) -> dict:
    """
    Verify answer's claims against chunks.

    Returns a dict with:
      - "report": full free-text explanation (per-claim SUPPORTED/UNSUPPORTED)
      - "is_grounded": bool, parsed from the report's final verdict line
    """
    context = build_source_context(chunks)
    prompt = _GROUNDEDNESS_PROMPT_TEMPLATE.format(context=context, answer=answer)

    # Longest expected output of the three call_llm() use sites (a per-claim
    # SUPPORTED/UNSUPPORTED breakdown plus verdict) — needs more headroom
    # than the 1024-token default.
    report = call_llm(prompt, max_tokens=2048)
    is_grounded = _parse_verdict(report)

    return {"report": report, "is_grounded": is_grounded}
