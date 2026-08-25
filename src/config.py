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
# Groq and Gemini are both wired up as interchangeable providers (see
# src/generation/llm_client.py) — both free tiers are tight enough that
# hitting one's rate limit mid-session is a real scenario for this project,
# not a hypothetical one. Switch providers via LLM_PROVIDER env var (no
# code change, just restart) if one is exhausted and the other isn't.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")   # "groq" or "gemini"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # kept for reference / easy revert

GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was retired by Groq
GROQ_REASONING_EFFORT = "low"      # gpt-oss is a reasoning model; 'medium' (its default) can burn
                                    # the whole max_tokens budget on hidden reasoning and return
                                    # empty content for longer-output tasks like groundedness

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_THINKING_LEVEL = "MINIMAL"  # same failure mode as Groq's reasoning_effort: Gemini 3.6's
                                    # default thinking level can consume the whole max_output_tokens
                                    # budget internally and return a truncated/empty response

# On a rate limit: one quick retry after a short fixed delay, then give up
# with a friendly error — not exponential backoff. A live demo shouldn't
# leave a user's request hanging for minutes; failing fast (and letting a
# human flip LLM_PROVIDER if it keeps happening) is the better tradeoff here.
LLM_MAX_ATTEMPTS = 2
LLM_RATE_LIMIT_RETRY_DELAY_SECONDS = 5

# --- Semantic answer cache (in-process, resets on restart) ---
# Conservative on purpose: a manual calibration against bge-small-en-v1.5
# showed paraphrase and genuinely-distinct question pairs don't separate
# cleanly at a lower threshold (see src/generation/semantic_cache.py) — a
# loose threshold risks confidently serving a wrong cached answer to a
# different question, which is worse than a cache miss that just runs the
# pipeline normally. This only catches near-duplicate rewordings, not
# general paraphrases.
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.90
SEMANTIC_CACHE_MAX_ENTRIES = 200

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
    if LLM_PROVIDER not in ("groq", "gemini"):
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' — must be 'groq' or 'gemini'."
        )

    missing = []
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Check your .env file."
        )