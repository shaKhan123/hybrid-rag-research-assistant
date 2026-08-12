from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
import json
import uuid
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "arxiv_rag_hybrid"
BATCH_SIZE = 64

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

print("Loading dense embedder...")
dense_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print("Loading sparse (BM25) embedder...")
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

def deterministic_id(arxiv_id: str, chunk_index: int) -> str:
    key = f"{arxiv_id}_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

chunks = []
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks. Embedding + uploading in batches of {BATCH_SIZE}...")

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    texts = [c["text"] for c in batch]

    dense_vectors = dense_embedder.embed_documents(texts)
    sparse_vectors = list(sparse_embedder.embed(texts))  # returns SparseEmbedding objects

    points = [
        PointStruct(
            id=deterministic_id(c["arxiv_id"], c["chunk_index"]),
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
            },
            payload={
                "arxiv_id": c["arxiv_id"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            },
        )
        for c, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"  Uploaded batch {i // BATCH_SIZE + 1} / {(len(chunks) - 1) // BATCH_SIZE + 1}")

print("Done.")
print(client.get_collection(COLLECTION))