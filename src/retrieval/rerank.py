"""
Cross-encoder reranking: a precise second pass over hybrid retrieval's
top-k candidates.

Unlike fusion (which only knows rank position, not content), a
cross-encoder reads the query and each candidate chunk TOGETHER and
outputs a genuine relevance score. Validated earlier in this project to
promote chunks fusion ranked poorly (~rank 5-6) up to rank 1 when they
were in fact the most relevant result — the two-stage retrieve-then-rerank
design earns its added latency over fusion alone.

Only run over the top-k from hybrid retrieval (not the whole corpus) —
cross-encoders score query+document jointly, so there's no way to
pre-compute anything; running this over thousands of chunks directly
would be far too slow.
"""

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, RERANK_TOP_K

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Rerank hybrid retrieval candidates against the query.

    candidates: list of dicts as returned by hybrid_retrieve() (must each
    have a "text" key).

    Returns the top_k candidates, sorted by rerank score descending, each
    with a "rerank_score" key added (original keys, e.g. fusion_score,
    are preserved).
    """
    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [{**candidate, "rerank_score": float(score)} for score, candidate in top]
