"""
The actual LangGraph StateGraph wiring.

Flow:
    hyde -> retrieve -> rerank -> generate -> groundedness_check
                                       ^                |
                                       |                v
                                       +-- (retry) --- routing decision
                                                         |
                                                         v (grounded, or retries exhausted)
                                                        END

The conditional edge after groundedness_check_node is the piece this
project has been building toward: if the answer isn't grounded AND we
haven't exhausted MAX_GENERATION_RETRIES, loop back to generate_node
(which will produce a fresh answer from the same reranked_chunks) instead
of just reporting the failure and stopping.

Retrieval and reranking are NOT re-run on retry — only generation. The
assumption is that if grounding failed, the problem is more likely in how
the LLM used the sources than in which sources were retrieved. Re-running
retrieval on every retry would also make each retry strictly more
expensive for no clear benefit.
"""

from langgraph.graph import StateGraph, END

from src.config import MAX_GENERATION_RETRIES
from src.graph.state import RAGState, make_initial_state
from src.graph.nodes import (
    hyde_node,
    retrieve_node,
    rerank_node,
    generate_node,
    groundedness_check_node,
)


def _route_after_groundedness_check(state: RAGState) -> str:
    """Conditional edge: retry generation if ungrounded and retries remain,
    otherwise end."""
    if state["is_grounded"]:
        return "end"
    if state["retry_count"] >= MAX_GENERATION_RETRIES:
        print(
            f"[pipeline] Not grounded after {state['retry_count']} attempt(s) — "
            f"giving up, returning best-effort answer with warning."
        )
        return "end"
    print(f"[pipeline] Not grounded (attempt {state['retry_count']}) — retrying generation.")
    return "retry"


def build_graph():
    """Construct and compile the RAG StateGraph."""
    graph = StateGraph(RAGState)

    graph.add_node("hyde", hyde_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("groundedness_check", groundedness_check_node)

    graph.set_entry_point("hyde")
    graph.add_edge("hyde", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "groundedness_check")

    graph.add_conditional_edges(
        "groundedness_check",
        _route_after_groundedness_check,
        {
            "retry": "generate",
            "end": END,
        },
    )

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Lazily compile the graph once and reuse it."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str, use_hyde: bool = False) -> RAGState:
    """Run a single query through the full pipeline, end to end."""
    graph = get_graph()
    initial_state = make_initial_state(query, use_hyde=use_hyde)
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a query through the RAG pipeline.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--hyde", action="store_true", help="Use HyDE query rewriting.")
    args = parser.parse_args()

    result = run_query(args.query, use_hyde=args.hyde)

    print("=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\nGROUNDED:", result["is_grounded"], f"(after {result['retry_count']} check(s))")
    print("\nSOURCES:")
    for i, c in enumerate(result["reranked_chunks"], start=1):
        print(f"  [{i}] {c['arxiv_id']} chunk {c['chunk_index']} "
              f"(rerank_score={c.get('rerank_score', 0):.4f})")
