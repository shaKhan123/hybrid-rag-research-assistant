from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)

client.create_collection(
    collection_name="arxiv_rag_hybrid",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),
    },
)

print("Hybrid collection created.")
print(client.get_collections())