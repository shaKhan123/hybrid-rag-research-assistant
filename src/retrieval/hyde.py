"""
HyDE (Hypothetical Document Embeddings) query rewriting.

Instead of embedding the user's raw question, generate a fake/hallucinated
paragraph that plausibly answers it, then embed and retrieve using THAT
instead. The hallucinated answer is written in the same academic register
as real corpus content, so it often lands closer to genuinely relevant
chunks in embedding space than a short casual question would.

KNOWN LIMITATION (see README): HyDE is not a strict improvement. Validated
empirically in this project:
  - On a well-specified query, HyDE and raw-query retrieval returned
    different but comparably relevant results (no clear winner).
  - On an ambiguous query ("how do people fix bad retrieval?"), the LLM
    interpreted "retrieval" as human cognitive memory retrieval rather
    than information retrieval, generating a fluent but off-topic
    hypothetical answer that then actively degraded retrieval quality.
  HyDE has zero visibility into what's actually in the corpus, so
  ambiguous terms can be confidently resolved in the wrong direction.
  A corpus-anchored prompt (e.g. "...as if it were an excerpt from a
  computer science paper about information retrieval") would likely
  mitigate this — not yet implemented; use with that caveat in mind.
"""

from src.generation.llm_client import call_llm

_HYDE_PROMPT_TEMPLATE = """Write a short, plausible-sounding paragraph that could answer this question, \
as if it were an excerpt from an academic paper. It's okay if the details aren't factually \
verified — the goal is just to match the style and vocabulary of real research writing.

Question: {query}

Paragraph:"""


def generate_hyde_answer(query: str) -> str:
    """Generate a hypothetical (fabricated) answer paragraph for the given query,
    intended to be embedded and used as a retrieval query in place of the
    raw question."""
    prompt = _HYDE_PROMPT_TEMPLATE.format(query=query)
    return call_llm(prompt)
