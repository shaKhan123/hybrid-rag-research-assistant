"""
FastAPI backend for the RAG pipeline.

Design choices, deliberately scoped for a public portfolio demo rather
than a real production service (see README's "Path to production"
section for what's intentionally NOT built here):

- Per-IP in-memory rate limiting (slowapi) — protects the free-tier LLM
  quota from being drained by a bot or an overenthusiastic visitor.
  Not distributed (won't work correctly across multiple instances/workers)
  — fine for a single-container demo, not fine at real scale.
- No auth — this is a public read-only demo endpoint. Add an API key
  check here before exposing anything that costs real money at higher
  traffic.
- Errors are caught and returned as clean JSON, never raw tracebacks —
  a public demo shouldn't leak internals.
- Models (embedders, reranker) load once at process startup via the
  existing lazy-singleton pattern in src/indexing/embed.py and
  src/retrieval/rerank.py — the first request after a cold start will be
  slow (model loading), subsequent ones are fast.

Run locally:
    uvicorn api.main:app --reload --port 8000

Run in Docker: see Dockerfile.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.config import validate_config, QDRANT_COLLECTION
from src.graph.pipeline import run_query
from src.indexing.qdrant_store import get_client
from src.indexing.embed import get_dense_embedder, get_sparse_embedder
from src.retrieval.rerank import get_reranker
from src.generation.llm_client import LLMRateLimitedError
from src.generation.semantic_cache import find_cached_result, store_result
from api.schemas import QueryRequest, QueryResponse, SourceChunk, HealthResponse, ErrorResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Rate limiting: per-IP, in-memory ---
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail loudly and immediately on missing config, rather than on the
    first request. Current recommended FastAPI startup pattern —
    on_event("startup") is deprecated as of fastapi 0.115."""
    try:
        validate_config()
        logger.info("Config validated OK.")
    except ValueError as e:
        logger.error("Startup config validation failed: %s", e)
        raise

    logger.info("Warming up embedder/reranker models...")
    get_dense_embedder()
    get_sparse_embedder()
    get_reranker()
    logger.info("Models warmed up — ready to serve.")

    yield


app = FastAPI(
    title="Hybrid RAG Research Assistant API",
    description="Query a hybrid-retrieval RAG pipeline over arXiv papers on retrieval-augmented generation.",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: open for a public demo frontend. Tighten to specific origins if
# this is ever more than a portfolio piece.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(LLMRateLimitedError)
async def llm_rate_limited_handler(request: Request, exc: LLMRateLimitedError):
    """The LLM provider itself is rate-limited — distinct from slowapi's
    per-IP limiter above. call_llm() fails fast after one quick retry
    (see src/generation/llm_client.py) rather than hanging on repeated
    backoff, so this returns promptly instead of after several minutes."""
    logger.warning("LLM provider rate-limited on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="llm_rate_limited",
            detail="The AI model is temporarily rate-limited. Please try again in a minute.",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: never leak a raw traceback to a public client."""
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="Something went wrong processing this request.",
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Verify the service and its Qdrant dependency are reachable.
    Used by hosting platforms (Render/Fly/Railway) for uptime checks."""
    try:
        client = get_client()
        info = client.get_collection(QDRANT_COLLECTION)
        return HealthResponse(
            status="ok",
            qdrant_connected=True,
            qdrant_collection=QDRANT_COLLECTION,
            points_count=info.points_count,
        )
    except Exception as e:
        logger.warning("Health check: Qdrant unreachable: %s", e)
        return HealthResponse(
            status="degraded",
            qdrant_connected=False,
            detail=str(e),
        )


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute;100/day")
async def query(request: Request, body: QueryRequest):
    """
    Run a query through the full RAG pipeline: hybrid retrieve -> rerank
    -> generate -> groundedness check (with retry on ungrounded answers).

    A close-enough repeat of a previously-answered (and grounded) question
    is served from the semantic cache instead — see
    src/generation/semantic_cache.py for why this only catches near-duplicate
    rewordings rather than general paraphrases.
    """
    logger.info("Query received: %s (hyde=%s)", body.query[:100], body.use_hyde)

    cached = find_cached_result(body.query, body.use_hyde)
    if cached is not None:
        logger.info("Cache hit for: %s", body.query[:100])
        result = cached
        from_cache = True
    else:
        result = run_query(body.query, use_hyde=body.use_hyde)
        store_result(body.query, body.use_hyde, result)
        from_cache = False

    sources = [
        SourceChunk(
            arxiv_id=c["arxiv_id"],
            chunk_index=c["chunk_index"],
            rerank_score=c.get("rerank_score"),
        )
        for c in result["reranked_chunks"]
    ]

    return QueryResponse(
        query=body.query,
        answer=result["answer"],
        is_grounded=result["is_grounded"],
        retry_count=result["retry_count"],
        groundedness_report=None if result["is_grounded"] else result["groundedness_report"],
        sources=sources,
        from_cache=from_cache,
    )