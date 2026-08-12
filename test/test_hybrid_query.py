from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "arxiv_rag_hybrid"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)

print("Loading embedders...")
dense_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

query = "How does hybrid retrieval combine dense and sparse search methods?"

# Embed the query both ways
dense_query_vector = dense_embedder.embed_query(query)
sparse_query_vector = list(sparse_embedder.embed([query]))[0]

results = client.query_points(
    collection_name=COLLECTION,
    prefetch=[
        Prefetch(
            query=dense_query_vector,
            using="dense",
            limit=20,
        ),
        Prefetch(
            query=SparseVector(
                indices=sparse_query_vector.indices.tolist(),
                values=sparse_query_vector.values.tolist(),
            ),
            using="sparse",
            limit=20,
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
)

for i, point in enumerate(results.points):
    print(f"--- Fused Result {i+1} (score: {point.score:.5f}) ---")
    print("Paper:", point.payload["arxiv_id"], "| Chunk:", point.payload["chunk_index"])
    print(point.payload["text"][:300], "...")
    print()