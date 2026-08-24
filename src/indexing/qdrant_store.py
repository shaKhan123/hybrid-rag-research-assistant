"""
Qdrant collection management: creation and chunk upload.

Collection uses named vectors ("dense" + "sparse") on the same point, so
each chunk carries both representations side by side — this is what lets
hybrid retrieval fuse dense + sparse search server-side via Qdrant's
native FusionQuery, rather than fusing results in Python.

Point IDs are deterministic (derived from arxiv_id + chunk_index via
uuid5), not random — this makes re-running the upload idempotent: the
same chunk always maps to the same point ID, so re-uploading overwrites
rather than duplicates.
"""

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, SparseVector

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_TIMEOUT_SECONDS,
    DENSE_EMBEDDING_DIM,
    UPLOAD_BATCH_SIZE,
    CHUNKS_PATH,
)
from src.indexing.embed import embed_dense_documents, embed_sparse_documents

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT_SECONDS)
    return _client


def create_hybrid_collection(collection_name: str = QDRANT_COLLECTION):
    """Create a collection with both dense and sparse named vector slots.

    Qdrant doesn't support changing vector config on an existing
    collection, so this intentionally raises if the collection already
    exists rather than silently no-oping — a silent no-op could mask a
    real schema mismatch.
    """
    client = get_client()
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=DENSE_EMBEDDING_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    print(f"Collection '{collection_name}' created.")


def deterministic_id(arxiv_id: str, chunk_index: int) -> str:
    """Derive a stable UUID from (arxiv_id, chunk_index), so re-uploading
    the same chunk always maps to the same Qdrant point ID (upsert
    overwrites instead of duplicating)."""
    key = f"{arxiv_id}_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def upload_chunks(chunks_path: Path = Path(CHUNKS_PATH),
                   collection_name: str = QDRANT_COLLECTION,
                   batch_size: int = UPLOAD_BATCH_SIZE):
    """Embed (dense + sparse) and upload every chunk in chunks_path, in batches."""
    client = get_client()

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks. Embedding + uploading in batches of {batch_size}...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        dense_vectors = embed_dense_documents(texts)
        sparse_vectors = embed_sparse_documents(texts)

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

        client.upsert(collection_name=collection_name, points=points)
        print(f"  Uploaded batch {i // batch_size + 1} / {(len(chunks) - 1) // batch_size + 1}")

    print("Done.")
    print(client.get_collection(collection_name))


def main():
    parser = argparse.ArgumentParser(description="Create Qdrant collection and/or upload chunks.")
    parser.add_argument("--create-collection", action="store_true",
                         help="Create the hybrid collection before uploading.")
    parser.add_argument("--chunks-path", type=str, default=CHUNKS_PATH,
                         help="Path to chunks.jsonl.")
    parser.add_argument("--collection-name", type=str, default=QDRANT_COLLECTION,
                         help="Qdrant collection name.")
    args = parser.parse_args()

    if args.create_collection:
        create_hybrid_collection(collection_name=args.collection_name)

    upload_chunks(chunks_path=Path(args.chunks_path), collection_name=args.collection_name)


if __name__ == "__main__":
    main()
