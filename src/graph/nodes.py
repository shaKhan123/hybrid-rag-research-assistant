"""
LangGraph node functions.

Each node follows the same shape: takes the current RAGState in, returns a
dict of only the fields it updates. LangGraph merges that update back into
the shared state automatically when wired into an actual StateGraph
(see pipeline.py).

Every node here was validated standalone (by hand-chaining them, before
pipeline.py existed) earlier in this project:
  - retrieve_node: confirmed correct hybrid retrieval results
  - rerank_node: confirmed promoting a fusion-rank-~5-6 chunk to rank 1
  - generate_node: confirmed declining to answer when sources were insufficient
  - groundedness_check_node: confirmed flagging all 4 claims in a
    deliberately fabricated test answer
"""

from src.config import RETRIEVE_TOP_K, RERANK_TOP_K
from src.retrieval.hybrid_search import hybrid_retrieve
from src.retrieval.rerank import rerank
from src.retrieval.hyde import generate_hyde_answer
from src.generation.answer import generate_answer
from src.generation.groundedness import check_groundedness
from src.generation.intent import classify_intent
from src.graph.state import RAGState


def classify_intent_node(state: RAGState) -> dict:
    """Classify state['query'] as chitchat vs a genuine research question.

    Chitchat gets a canned reply and skips straight to END (see
    pipeline.py's conditional edge after this node) — no retrieval,
    reranking, generation, or groundedness check needed for "hi".
    """
    print(f"[classify_intent_node] Classifying: {state['query'][:80]}")

    result = classify_intent(state["query"])

    if result["is_chitchat"]:
        print("[classify_intent_node] Chitchat — skipping the retrieval pipeline.\n")
        return {
            "is_chitchat": True,
            "answer": result["reply"],
            "is_grounded": True,
            "retry_count": 0,
        }

    print("[classify_intent_node] Research question — proceeding to retrieval.\n")
    return {"is_chitchat": False}


def hyde_node(state: RAGState) -> dict:
    """Generate a HyDE hypothetical-answer paragraph and use it as the
    retrieval query, if use_hyde is set. Otherwise pass the raw query
    through unchanged.

    KNOWN LIMITATION: HyDE has no visibility into corpus content, so
    ambiguous query terms can be confidently resolved in the wrong
    direction (see retrieval/hyde.py docstring for the documented
    "retrieval = memory" failure case found during development).
    """
    if not state.get("use_hyde"):
        return {"retrieval_query": state["query"]}

    print(f"[hyde_node] Generating HyDE paragraph for: {state['query']}")
    hyde_text = generate_hyde_answer(state["query"])
    print(f"[hyde_node] Generated ({len(hyde_text)} chars).\n")

    return {"retrieval_query": hyde_text}


def retrieve_node(state: RAGState) -> dict:
    """Hybrid retrieve top RETRIEVE_TOP_K candidates for state['retrieval_query']."""
    query_text = state.get("retrieval_query") or state["query"]
    print(f"[retrieve_node] Query: {query_text[:80]}")

    chunks = hybrid_retrieve(query_text, k=RETRIEVE_TOP_K)

    print(f"[retrieve_node] Retrieved {len(chunks)} candidates.\n")

    return {"retrieved_chunks": chunks}


def rerank_node(state: RAGState) -> dict:
    """Cross-encoder rerank state['retrieved_chunks'] down to top RERANK_TOP_K.

    Reranks against the ORIGINAL query, not the HyDE paragraph — the
    cross-encoder should judge relevance to what the user actually asked,
    not to the LLM's hypothetical answer.
    """
    print(f"[rerank_node] Reranking {len(state['retrieved_chunks'])} candidates...")

    reranked = rerank(state["query"], state["retrieved_chunks"], top_k=RERANK_TOP_K)

    print(f"[rerank_node] Kept top {len(reranked)} after reranking.\n")

    return {"reranked_chunks": reranked}


def generate_node(state: RAGState) -> dict:
    """Generate an answer from state['reranked_chunks']."""
    print("[generate_node] Generating answer...")

    answer = generate_answer(state["query"], state["reranked_chunks"])

    print(f"[generate_node] Answer generated ({len(answer)} chars).\n")

    return {"answer": answer}


def groundedness_check_node(state: RAGState) -> dict:
    """Verify state['answer'] is supported by state['reranked_chunks'].

    Increments retry_count regardless of the outcome — this is the
    counter pipeline.py's conditional edge uses to cap retries.
    """
    print("[groundedness_check_node] Checking groundedness...")

    result = check_groundedness(state["answer"], state["reranked_chunks"])

    verdict = "GROUNDED" if result["is_grounded"] else "NOT GROUNDED"
    print(f"[groundedness_check_node] Verdict: {verdict}\n")

    return {
        "groundedness_report": result["report"],
        "is_grounded": result["is_grounded"],
        "retry_count": state["retry_count"] + 1,
    }
