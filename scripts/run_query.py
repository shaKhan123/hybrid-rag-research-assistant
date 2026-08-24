"""
CLI entrypoint: ask a single question through the full RAG pipeline.

Usage:
    python -m scripts.run_query "How does hybrid retrieval work?"
    python -m scripts.run_query "How does hybrid retrieval work?" --hyde
"""

from src.graph.pipeline import run_query


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ask a question via the RAG pipeline.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--hyde", action="store_true", help="Use HyDE query rewriting.")
    args = parser.parse_args()

    result = run_query(args.query, use_hyde=args.hyde)

    print("=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\nGROUNDED:", result["is_grounded"], f"(after {result['retry_count']} check(s))")
    if not result["is_grounded"]:
        print("\nGROUNDEDNESS REPORT:")
        print(result["groundedness_report"])
    print("\nSOURCES:")
    for i, c in enumerate(result["reranked_chunks"], start=1):
        print(f"  [{i}] {c['arxiv_id']} chunk {c['chunk_index']} "
              f"(rerank_score={c.get('rerank_score', 0):.4f})")


if __name__ == "__main__":
    main()
