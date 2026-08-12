import json
from rank_bm25 import BM25Okapi

# Load all chunks
chunks = []
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks.")

# BM25 needs tokenized text — simplest possible tokenizer: lowercase + split on whitespace
tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]

bm25 = BM25Okapi(tokenized_corpus)

query = "How does hybrid retrieval combine dense and sparse search methods?"
tokenized_query = query.lower().split()

scores = bm25.get_scores(tokenized_query)

# Get top 5 by BM25 score
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

for rank, idx in enumerate(top_indices, start=1):
    chunk = chunks[idx]
    print(f"--- Result {rank} (BM25 score: {scores[idx]:.4f}) ---")
    print("Paper:", chunk["arxiv_id"], "| Chunk:", chunk["chunk_index"])
    print(chunk["text"][:300], "...")
    print()