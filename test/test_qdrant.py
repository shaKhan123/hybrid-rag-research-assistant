from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json
import uuid
import os

load_dotenv()  # reads .env and loads its values into environment variables

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "arxiv_rag_chunks"
BATCH_SIZE = 64

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY missing — check your .env file.")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

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
    vectors = embedder.embed_documents(texts)

    points = [
        PointStruct(
            id=deterministic_id(c["arxiv_id"], c["chunk_index"]),
            vector=vec,
            payload={
                "arxiv_id": c["arxiv_id"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            },
        )
        for c, vec in zip(batch, vectors)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"  Uploaded batch {i // BATCH_SIZE + 1} / {(len(chunks) - 1) // BATCH_SIZE + 1}")

print("Done.")
print(client.get_collection(COLLECTION))