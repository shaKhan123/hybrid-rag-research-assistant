"""
Shared prompt-formatting helpers used across generation modules.
"""


def build_source_context(chunks: list[dict]) -> str:
    """Format retrieved/reranked chunks into a numbered [Source N] block,
    used both when generating an answer and when checking its groundedness
    (so the two prompts reference sources identically)."""
    return "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
    )
