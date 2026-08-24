# Single-container image for the RAG API. Qdrant and Groq are external
# dependencies reached over the network — this container holds only the
# FastAPI app and the local embedding/reranker models.

FROM python:3.11-slim

# curl is needed for the HEALTHCHECK below (not for the app itself).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
RUN chown -R appuser:appuser /app

USER appuser
ENV HF_HOME=/app/.cache/huggingface

# Pre-download the embedding/reranker weights at build time, as the same
# user that will run the container, so the runtime cache lookup is a hit
# instead of a silent re-download: bakes them into the image layer so
# `docker run` never depends on Hugging Face Hub being reachable, and a
# broken download fails the build instead of failing a live deploy.
RUN python -c "\
from src.indexing.embed import get_dense_embedder, get_sparse_embedder; \
from src.retrieval.rerank import get_reranker; \
get_dense_embedder(); get_sparse_embedder(); get_reranker()"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# $PORT is honored for platforms (Render, etc.) that inject it; defaults to
# 8000 for plain `docker run`.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-8000}/health || exit 1"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
