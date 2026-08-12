from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "arxiv_rag_chunks"

print("1. Env loaded. URL:", QDRANT_URL)

print("2. Loading embedding model...")
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("3. Embedding model loaded.")

print("4. Connecting to Qdrant client...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
print("5. Client created.")

query = "How does hybrid retrieval combine dense and sparse search methods?"

print("6. Embedding query...")
query_vector = embedder.embed_query(query)
print("7. Query embedded. Vector length:", len(query_vector))

print("8. Querying Qdrant...")
results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=5,
)
print("9. Got results.")

for i, point in enumerate(results.points):
    print(f"--- Result {i+1} (score: {point.score:.4f}) ---")
    print("Paper:", point.payload["arxiv_id"], "| Chunk:", point.payload["chunk_index"])
    print(point.payload["text"][:300], "...")
    print()