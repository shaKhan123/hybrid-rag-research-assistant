"""
In-process semantic cache for /query results: if a new question is close
enough (by dense-embedding cosine similarity) to one already answered, skip
the whole hybrid-retrieve -> rerank -> generate -> groundedness pipeline and
return the cached result instead.

Threshold is deliberately conservative (see SEMANTIC_CACHE_SIMILARITY_THRESHOLD
in config.py). A manual calibration against this project's embedding model
(BAAI/bge-small-en-v1.5) showed paraphrase and genuinely-distinct question
pairs don't separate cleanly at a low threshold — e.g. "What is RRF?" vs.
"How does RRF combine rankings?" scored 0.76 cosine similarity, while "RRF"
vs. "HyDE" (a different topic) scored 0.55. A loose threshold risks
confidently serving a wrong cached answer to a different question, which is
worse than a cache miss that just runs the pipeline normally. At the chosen
threshold this only catches near-duplicate rewordings, not general
paraphrases — a smaller win, but a safe one.

Only grounded results are cached — an ungrounded best-effort answer
shouldn't get "locked in" and repeatedly served to every close rewording of
the question.

In-memory only: resets on restart, not shared across multiple workers —
same scope/limitation as the per-IP rate limiter in api/main.py.
"""

import math
from collections import deque

from src.config import SEMANTIC_CACHE_SIMILARITY_THRESHOLD, SEMANTIC_CACHE_MAX_ENTRIES
from src.indexing.embed import embed_dense_query

_cache = deque(maxlen=SEMANTIC_CACHE_MAX_ENTRIES)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_cached_result(query: str, use_hyde: bool) -> dict | None:
    """Return a previously-cached pipeline result for a close-enough
    question, or None on a cache miss.

    Cached separately per use_hyde since it changes what's actually
    retrieved for a given question.
    """
    query_embedding = embed_dense_query(query)

    best_similarity = 0.0
    best_result = None
    for entry in _cache:
        if entry["use_hyde"] != use_hyde:
            continue
        similarity = _cosine_similarity(query_embedding, entry["embedding"])
        if similarity > best_similarity:
            best_similarity = similarity
            best_result = entry["result"]

    if best_similarity >= SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
        return best_result
    return None


def store_result(query: str, use_hyde: bool, result: dict) -> None:
    """Cache a pipeline result — but only if it was actually grounded."""
    if not result.get("is_grounded"):
        return

    _cache.append({
        "embedding": embed_dense_query(query),
        "use_hyde": use_hyde,
        "result": result,
    })
