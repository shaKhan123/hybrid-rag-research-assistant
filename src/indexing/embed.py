"""
Dense + sparse embedding functions.

Dense: BAAI/bge-small-en-v1.5 (384-dim), captures semantic meaning.
Sparse: Qdrant/bm25 via fastembed, captures exact keyword matches.

Both models are loaded lazily and cached at module level — loading them is
a real cost (a few seconds each), so every caller should share one instance
rather than constructing a fresh embedder per call.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding

from src.config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL

_dense_embedder = None
_sparse_embedder = None


def get_dense_embedder() -> HuggingFaceEmbeddings:
    global _dense_embedder
    if _dense_embedder is None:
        _dense_embedder = HuggingFaceEmbeddings(model_name=DENSE_EMBEDDING_MODEL)
    return _dense_embedder


def get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)
    return _sparse_embedder


def embed_dense_query(text: str) -> list[float]:
    """Embed a single query string (dense)."""
    return get_dense_embedder().embed_query(text)


def embed_dense_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (dense) — one model call for the whole batch."""
    return get_dense_embedder().embed_documents(texts)


def embed_sparse_query(text: str):
    """Embed a single query string (sparse). Returns a fastembed SparseEmbedding
    object with .indices and .values attributes."""
    return list(get_sparse_embedder().embed([text]))[0]


def embed_sparse_documents(texts: list[str]) -> list:
    """Embed a batch of documents (sparse). Returns a list of SparseEmbedding
    objects, one per input text."""
    return list(get_sparse_embedder().embed(texts))
