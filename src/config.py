"""
Central configuration for the RAG pipeline.

Every model name, collection name, and tunable constant lives here — nowhere
else in the codebase. When a model gets deprecated (it will — we've already
hit this twice with Gemini this project), fix it in ONE place, not by
grepping through a dozen scripts.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Credentials (loaded from .env, never hardcoded) ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- arXiv ingestion ---
ARXIV_DEFAULT_QUERY = "retrieval augmented generation"
ARXIV_DEFAULT_CATEGORIES = ["cs.CL", "cs.LG"]
ARXIV_MAX_RESULTS = 50
ARXIV_API_DELAY_SECONDS = 3.0  # politeness delay between paginated API requests

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CONTENT_CHUNK_CHARS = 500  # threshold used when sampling "real content" chunks for eval

# --- Embeddings ---
DENSE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_EMBEDDING_DIM = 384
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# --- Qdrant ---
QDRANT_COLLECTION = "arxiv_rag_hybrid"
QDRANT_TIMEOUT_SECONDS = 10
UPLOAD_BATCH_SIZE = 64

# --- Retrieval ---
RETRIEVE_TOP_K = 20      # candidates pulled by hybrid fusion, before reranking
RERANK_TOP_K = 5         # candidates kept after cross-encoder reranking
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- LLM (generation, HyDE, groundedness checking) ---
LLM_PROVIDER = "groq"             # kept explicit — this project has swapped providers before
GEMINI_MODEL = "gemini-3.6-flash"      # kept for reference / easy revert
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # kept for reference / easy revert
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was retired by Groq
GROQ_REASONING_EFFORT = "low"      # gpt-oss is a reasoning model; 'medium' (its default) can burn
                                    # the whole max_tokens budget on hidden reasoning and return
                                    # empty content for longer-output tasks like groundedness
LLM_MAX_RETRIES = 5
LLM_BASE_BACKOFF_SECONDS = 15     # matters most for Gemini free tier; Groq's limits are looser

# --- Groundedness / generation retry loop ---
MAX_GENERATION_RETRIES = 2        # how many times the graph will retry generation if ungrounded

# --- Evaluation ---
EVAL_BENCHMARK_SIZE = 30          # number of papers sampled for benchmark question generation
EVAL_RANDOM_SEED = 42             # reproducible sampling across runs

# --- Paths ---
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
CHUNKS_PATH = "data/processed/chunks.jsonl"
BENCHMARK_PATH = "data/processed/benchmark.jsonl"
EVAL_RESULTS_PATH = "data/processed/eval_results.json"


def validate_config():
    """Fail fast and clearly if required credentials are missing, rather than
    letting a cryptic connection error surface three layers deep later."""
    missing = []
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Check your .env file."
        )