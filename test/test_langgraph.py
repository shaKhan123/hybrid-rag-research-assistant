from typing import TypedDict, List
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder
from google import genai
from google.genai._gaos.lib.compat_errors import RateLimitError
from dotenv import load_dotenv
import os
import time

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "arxiv_rag_hybrid"
FINAL_K = 5  # how many chunks survive after reranking
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 15  # free tier is tight (5-20 req/min depending on model/tier)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Loading embedders + reranker...")
dense_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder("BAAI/bge-reranker-base")
print("Models loaded.\n")


def call_llm(prompt: str) -> str:
    """Call Gemini with retry/backoff on rate limits — shared by every node that needs an LLM."""
    for attempt in range(MAX_RETRIES):
        try:
            interaction = gemini.interactions.create(model="gemini-3.6-flash", input=prompt)
            return interaction.output_text.strip()
        except RateLimitError:
            wait = BASE_BACKOFF_SECONDS * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)

    raise RuntimeError("Gemini call failed after max retries — check quota/billing.")


class RAGState(TypedDict):
    query: str
    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]
    answer: str
    groundedness_report: str
    is_grounded: bool
    retry_count: int


def retrieve_node(state: RAGState) -> dict:
    """LangGraph node: hybrid retrieve top 20 candidates for state['query']."""
    print(f"[retrieve_node] Query: {state['query']}")

    dense_vec = dense_embedder.embed_query(state["query"])
    sparse_vec = list(sparse_embedder.embed([state["query"]]))[0]

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=20),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=20,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=20,
    )

    # Convert Qdrant points into plain dicts — simpler to carry through state
    chunks = [
        {
            "arxiv_id": p.payload["arxiv_id"],
            "chunk_index": p.payload["chunk_index"],
            "text": p.payload["text"],
            "fusion_score": p.score,
        }
        for p in results.points
    ]

    print(f"[retrieve_node] Retrieved {len(chunks)} candidates.\n")

    # Only return the fields this node updates — LangGraph merges the rest
    return {"retrieved_chunks": chunks}


def rerank_node(state: RAGState) -> dict:
    """LangGraph node: cross-encoder rerank state['retrieved_chunks'] down to top FINAL_K."""
    print(f"[rerank_node] Reranking {len(state['retrieved_chunks'])} candidates...")

    pairs = [[state["query"], c["text"]] for c in state["retrieved_chunks"]]
    scores = reranker.predict(pairs)

    scored = list(zip(scores, state["retrieved_chunks"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:FINAL_K]

    reranked = [
        {**chunk, "rerank_score": float(score)}
        for score, chunk in top
    ]

    print(f"[rerank_node] Kept top {len(reranked)} after reranking.\n")

    return {"reranked_chunks": reranked}


def generate_node(state: RAGState) -> dict:
    """LangGraph node: generate an answer from state['reranked_chunks']."""
    print("[generate_node] Generating answer...")

    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{c['text']}" for i, c in enumerate(state["reranked_chunks"])
    )

    prompt = f"""Answer the question using ONLY the information in the sources below. \
If the sources don't contain enough information to answer, say so explicitly rather than guessing. \
Cite which source(s) support each claim using [Source N] notation.

Sources:
{context}

Question: {state['query']}

Answer:"""

    answer = call_llm(prompt)

    print(f"[generate_node] Answer generated ({len(answer)} chars).\n")

    return {"answer": answer}


def groundedness_check_node(state: RAGState) -> dict:
    """LangGraph node: verify state['answer'] is supported by state['reranked_chunks']."""
    print("[groundedness_check_node] Checking groundedness...")

    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{c['text']}" for i, c in enumerate(state["reranked_chunks"])
    )

    prompt = f"""You are a fact-checker. Given the sources and a generated answer, \
identify any claims in the answer that are NOT directly supported by the sources. \
Be strict — if a claim is a reasonable inference but not explicitly stated, flag it too.

Sources:
{context}

Answer to check:
{state['answer']}

For each claim in the answer, respond with:
- SUPPORTED: [claim] — [which source supports it]
- UNSUPPORTED: [claim] — [why it's not backed by the sources]

End your response with exactly one line: either "VERDICT: GROUNDED" or "VERDICT: NOT GROUNDED"."""

    report = call_llm(prompt)

    # Check the actual last line for the verdict, rather than searching the whole
    # report (since "UNSUPPORTED" claims could otherwise falsely trip a substring match)
    last_line = report.strip().splitlines()[-1].upper()
    is_grounded = "NOT GROUNDED" not in last_line and "GROUNDED" in last_line

    print(f"[groundedness_check_node] Verdict: {'GROUNDED' if is_grounded else 'NOT GROUNDED'}\n")

    return {
        "groundedness_report": report,
        "is_grounded": is_grounded,
        "retry_count": state["retry_count"] + 1,
    }


if __name__ == "__main__":
    # Manual test: call all four node functions directly in sequence, no graph yet
    state: RAGState = {
        "query": "How does hybrid retrieval combine dense and sparse search methods?",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "groundedness_report": "",
        "is_grounded": False,
        "retry_count": 0,
    }

    state.update(retrieve_node(state))
    state.update(rerank_node(state))
    state.update(generate_node(state))
    state.update(groundedness_check_node(state))

    print("=" * 60)
    print("ANSWER:")
    print(state["answer"])
    print("\nGROUNDEDNESS VERDICT:", "GROUNDED" if state["is_grounded"] else "NOT GROUNDED")
    print("\nFULL GROUNDEDNESS REPORT:")
    print(state["groundedness_report"])