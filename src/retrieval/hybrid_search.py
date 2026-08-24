"""
Hybrid retrieval: dense + sparse search, fused server-side by Qdrant.

Uses Qdrant's native Prefetch + FusionQuery(fusion=Fusion.RRF) pattern
rather than a hand-rolled Python fusion loop — validated earlier in this
project to produce equivalent results to a manual Reciprocal Rank Fusion
implementation, at a fraction of the code and without pulling anything
into local process memory.
"""

from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

from src.config import QDRANT_COLLECTION, RETRIEVE_TOP_K
from src.indexing.embed import embed_dense_query, embed_sparse_query
from src.indexing.qdrant_store import get_client


def hybrid_retrieve(query: str, k: int = RETRIEVE_TOP_K, collection_name: str = QDRANT_COLLECTION) -> list[dict]:
    """
    Run hybrid (dense + sparse) retrieval for a query, fused via RRF.

    Returns a list of plain dicts (arxiv_id, chunk_index, text, fusion_score)
    rather than raw Qdrant point objects — keeps downstream code (reranking,
    generation, LangGraph nodes) decoupled from the Qdrant client's types.
    """
    client = get_client()

    dense_vec = embed_dense_query(query)
    sparse_vec = embed_sparse_query(query)

    results = client.query_points(
        collection_name=collection_name,
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

    return [
        {
            "arxiv_id": point.payload["arxiv_id"],
            "chunk_index": point.payload["chunk_index"],
            "text": point.payload["text"],
            "fusion_score": point.score,
        }
        for point in results.points
    ]
