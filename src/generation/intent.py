"""
Front-door intent classification: distinguish a genuine research question
from chitchat (greetings, thanks, small talk) before running the
hyde -> retrieve -> rerank -> generate -> groundedness pipeline.

Chitchat doesn't need retrieval or grounding — running the full pipeline on
"hi" would burn an embedding call, a reranker pass, and an LLM generation +
groundedness check for no benefit, against a free-tier LLM quota that's
already the tightest constraint in this project.
"""

from src.generation.llm_client import call_llm

_DEFAULT_CHITCHAT_REPLY = (
    "Hi! I'm a research assistant for retrieval-augmented generation (RAG) "
    "papers on arXiv. Ask me a question about RAG methods, retrieval, "
    "reranking, or evaluation and I'll look it up in the source papers."
)

_INTENT_PROMPT_TEMPLATE = """You are the front door of a research assistant that answers questions \
about retrieval-augmented generation (RAG) research papers, using only information from a corpus \
of arXiv papers.

Classify the user's message as exactly one of:
- RESEARCH_QUESTION: a genuine question that needs looking something up in research papers to answer.
- CHITCHAT: a greeting, thanks, small talk, or anything else that isn't a real research question.

Respond in exactly this format, nothing else:

INTENT: RESEARCH_QUESTION

or

INTENT: CHITCHAT
REPLY: <a brief, friendly one-sentence reply that invites a real research question>

Message: {query}"""


def classify_intent(query: str) -> dict:
    """
    Classify a user message as chitchat or a genuine research question.

    Returns a dict:
      - "is_chitchat": bool
      - "reply": str | None — only set when is_chitchat is True
    """
    response = call_llm(_INTENT_PROMPT_TEMPLATE.format(query=query), max_tokens=200)

    # Same pragmatic pattern as groundedness._parse_verdict: check one
    # specific line for the label rather than substring-searching the
    # whole response (avoids false positives from a rewording).
    first_line = response.strip().splitlines()[0].upper()
    if "CHITCHAT" not in first_line:
        return {"is_chitchat": False, "reply": None}

    reply = _DEFAULT_CHITCHAT_REPLY
    for line in response.splitlines():
        if line.strip().upper().startswith("REPLY:"):
            reply = line.split(":", 1)[1].strip() or _DEFAULT_CHITCHAT_REPLY
            break

    return {"is_chitchat": True, "reply": reply}
