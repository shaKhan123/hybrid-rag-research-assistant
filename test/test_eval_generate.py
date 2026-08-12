"""
Evaluate retrieval quality against benchmark.jsonl.

For each benchmark question, run hybrid retrieval and check whether the
known-correct source chunk shows up in the results — and at what rank.

Metrics:
- Hit Rate @ k: fraction of questions where the correct chunk appears
  anywhere in the top k results (simple yes/no per question).
- Mean Reciprocal Rank (MRR): average of 1/rank_of_correct_chunk across
  all questions (0 if not found). Rewards finding the right chunk EARLY,
  not just finding it somewhere in the list.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
import json
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "arxiv_rag_hybrid"
BENCHMARK_PATH = "benchmark.jsonl"
K = 10  # how many results to retrieve per question

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)

print("Loading embedders...")
dense_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
print("Embedders loaded.\n")


def hybrid_retrieve(query: str, k: int = K):
    dense_vec = dense_embedder.embed_query(query)
    sparse_vec = list(sparse_embedder.embed([query]))[0]

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=k),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
    )
    return results.points


def find_rank(results, target_arxiv_id: str, target_chunk_index: int):
    """Return the 1-based rank of the target chunk in results, or None if absent."""
    for rank, point in enumerate(results, start=1):
        if (point.payload["arxiv_id"] == target_arxiv_id
                and point.payload["chunk_index"] == target_chunk_index):
            return rank
    return None


# --- Load benchmark ---
benchmark = []
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            benchmark.append(json.loads(line))

print(f"Loaded {len(benchmark)} benchmark questions.\n")

# --- Run retrieval for every question, record the rank of the correct chunk ---
results_log = []
reciprocal_ranks = []
hits_at_k = 0

for i, item in enumerate(benchmark, start=1):
    query = item["question"]
    target_id = item["source_arxiv_id"]
    target_chunk = item["source_chunk_index"]

    retrieved = hybrid_retrieve(query, k=K)
    rank = find_rank(retrieved, target_id, target_chunk)

    if rank is not None:
        reciprocal_ranks.append(1.0 / rank)
        hits_at_k += 1
        status = f"FOUND at rank {rank}"
    else:
        reciprocal_ranks.append(0.0)
        status = f"NOT FOUND in top {K}"

    results_log.append({
        "question": query,
        "target": f"{target_id}_{target_chunk}",
        "rank": rank,
    })

    print(f"{i}. [{status}] {query[:80]}")

# --- Compute final metrics ---
hit_rate = hits_at_k / len(benchmark)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n" + "=" * 50)
print(f"Questions evaluated: {len(benchmark)}")
print(f"Hit Rate @ {K}: {hit_rate:.2%}  ({hits_at_k}/{len(benchmark)})")
print(f"MRR: {mrr:.4f}")
print("=" * 50)

# Save detailed results for later inspection
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "hit_rate_at_k": hit_rate,
        "mrr": mrr,
        "k": K,
        "num_questions": len(benchmark),
        "per_question": results_log,
    }, f, indent=2)

print("\nDetailed results saved to eval_results.json")