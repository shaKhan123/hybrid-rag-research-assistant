"""
Answer generation from retrieved/reranked chunks.

Prompts the LLM to answer using ONLY the provided sources, explicitly
instructed to say so when the sources don't contain enough information
rather than guessing — validated earlier in this project to correctly
decline to fill gaps when given truncated/incomplete source chunks.

The [Source N] citation convention in the prompt isn't just for
readability — it's what makes groundedness.py's per-claim verification
possible downstream.
"""

from src.generation.llm_client import call_llm
from src.generation.prompts import build_source_context

_ANSWER_PROMPT_TEMPLATE = """Answer the question using ONLY the information in the sources below. \
If the sources don't contain enough information to answer, say so explicitly rather than guessing. \
Cite which source(s) support each claim using [Source N] notation.

Sources:
{context}

Question: {query}

Answer:"""


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate a source-grounded answer to query, using chunks as context.

    chunks: list of dicts each with a "text" key (as returned by
    hybrid_retrieve() / rerank()).
    """
    context = build_source_context(chunks)
    prompt = _ANSWER_PROMPT_TEMPLATE.format(context=context, query=query)
    return call_llm(prompt)
