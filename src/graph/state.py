"""
Shared state schema for the LangGraph pipeline.

Every node reads from and writes to this single state object. Each node
takes the current state in and returns a dict of just the fields it
updates — LangGraph merges that back into the full state automatically.
"""

from typing import TypedDict, List, Optional


class RAGState(TypedDict):
    query: str
    use_hyde: bool                 # whether to rewrite the query via HyDE before retrieval
    is_chitchat: bool              # set by classify_intent_node; True skips retrieval/generation/groundedness
    retrieval_query: str           # the actual text used for retrieval (raw query or HyDE paragraph)
    retrieved_chunks: List[dict]   # raw hybrid retrieval results (top RETRIEVE_TOP_K)
    reranked_chunks: List[dict]    # after cross-encoder reranking (top RERANK_TOP_K)
    answer: str
    groundedness_report: str
    is_grounded: bool
    retry_count: int


def make_initial_state(query: str, use_hyde: bool = False) -> RAGState:
    """Build a fresh RAGState for a new query."""
    return {
        "query": query,
        "use_hyde": use_hyde,
        "is_chitchat": False,
        "retrieval_query": "",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "groundedness_report": "",
        "is_grounded": False,
        "retry_count": 0,
    }
