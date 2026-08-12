from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

sentences = [
    "Hybrid retrieval combines dense vector search with sparse keyword search like BM25.",
    "Combining semantic embeddings with keyword-based search improves retrieval accuracy.",
    "The chef added fresh basil to the tomato sauce before serving.",
]

vectors = embedder.embed_documents(sentences)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Similarity (sentence 0 vs 1 — both about hybrid retrieval):",
      cosine_similarity(vectors[0], vectors[1]))
print("Similarity (sentence 0 vs 2 — retrieval vs cooking):",
      cosine_similarity(vectors[0], vectors[2]))