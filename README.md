# Hybrid RAG Research Assistant

A retrieval-augmented generation system over arXiv papers, built to explore
and evaluate the techniques that separate a production-grade RAG pipeline
from a naive "embed and stuff into context" chatbot: hybrid (dense + sparse)
retrieval, cross-encoder reranking, query rewriting, automated groundedness
verification with a LangGraph retry loop, and quantitative evaluation.

Also includes a FastAPI backend and a Streamlit demo frontend that call
that pipeline over HTTP, a switchable dual LLM provider setup (Groq /
Gemini) for when one hits its free-tier rate limit, and a Docker image for
running the API as a single container.

Corpus: ~50 arXiv papers on **retrieval-augmented generation** itself
(`cs.CL` / `cs.LG`).

## Project structure

```
rag-project/
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile                 # single-container image for the FastAPI backend
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                  # downloaded PDFs + metadata.jsonl (gitignored)
│   └── processed/            # texts.jsonl, chunks.jsonl, benchmark.jsonl, eval_results.json
│
├── src/
│   ├── config.py              # every model name / constant / path - one place
│   │
│   ├── ingest/
│   │   ├── fetch_arxiv.py     # arXiv API search + PDF download, resumable
│   │   └── extract_text.py    # PyMuPDF text extraction
│   │
│   ├── chunking/
│   │   └── chunker.py         # LangChain splitter + reference cutoff + boilerplate filter
│   │
│   ├── indexing/
│   │   ├── embed.py           # dense + sparse embedding functions
│   │   └── qdrant_store.py    # collection creation + chunk upload
│   │
│   ├── retrieval/
│   │   ├── hybrid_search.py   # dense+sparse retrieval, native Qdrant RRF fusion
│   │   ├── rerank.py          # cross-encoder reranking
│   │   └── hyde.py            # HyDE query rewriting
│   │
│   ├── generation/
│   │   ├── llm_client.py      # shared call_llm(): dispatches to Groq or Gemini (LLM_PROVIDER),
│   │   │                      #   one quick retry then fail fast on a rate limit
│   │   ├── intent.py          # chitchat vs. real-question classification (front-door router)
│   │   ├── semantic_cache.py  # embedding-similarity cache for repeat/near-duplicate questions
│   │   ├── prompts.py         # shared prompt-formatting helpers
│   │   ├── answer.py          # source-grounded answer generation
│   │   └── groundedness.py    # per-claim groundedness verification
│   │
│   ├── graph/
│   │   ├── state.py           # RAGState TypedDict
│   │   ├── nodes.py           # LangGraph node functions (incl. classify_intent_node)
│   │   └── pipeline.py        # the compiled StateGraph + conditional retry/routing edges
│   │
│   └── evaluation/
│       ├── generate_benchmark.py  # auto-generate Q&A pairs from known chunks
│       └── metrics.py             # Hit Rate@k, MRR, groundedness pass rate
│
├── api/                       # FastAPI backend
│   ├── main.py                 # /query, /health; rate limiting, CORS, error handling
│   └── schemas.py               # request/response Pydantic models
│
├── streamlit_app/              # Streamlit demo frontend (thin client, calls the API over HTTP)
│   ├── app.py
│   └── requirements.txt        # kept separate from the API's — no torch/sentence-transformers needed
│
└── scripts/                  # thin CLI entrypoints
    ├── run_ingest.py          # fetch -> extract -> chunk -> index, end to end
    ├── run_query.py           # ask a single question through the graph
    └── run_eval.py            # generate benchmark / score retrieval or full pipeline
```

## Architecture

```
arXiv API
   |
   v
Fetch metadata + PDFs  ------------------------  src/ingest/fetch_arxiv.py
   |
   v
Extract text (PyMuPDF)  ---------------------  src/ingest/extract_text.py
   |
   v
Chunk (LangChain + reference cutoff + boilerplate filter)  --  src/chunking/chunker.py
   |
   v
Embed (dense: BAAI/bge-small-en-v1.5, sparse: Qdrant/bm25)  ---  src/indexing/embed.py
   |
   v
Qdrant Cloud (named vectors, server-side RRF fusion)  --------  src/indexing/qdrant_store.py
   |
   v
   +--------------------- src/graph/pipeline.py (compiled StateGraph) ---------------------+
   |                                                                                        |
   |  classify_intent --(chitchat)--> end (canned reply, skips everything below)            |
   |        |                                                                               |
   |        v (research question)                                                           |
   |  hyde -> retrieve -> rerank -> generate -> groundedness_check                          |
   |            (src/retrieval/)      (src/generation/)      |          |                   |
   |                                                          v          v                   |
   |                                                   retry (loop)   end (return answer)    |
   +----------------------------------------------------------------------------------------+
   |
   v
api/main.py (FastAPI)  --  semantic-cache check (src/generation/semantic_cache.py) short-circuits
   |                        a close-enough repeat question before it ever reaches the graph above
   v
streamlit_app/app.py  --  thin client, POSTs to /query over HTTP
```

`call_llm()` (`src/generation/llm_client.py`) sits underneath `hyde`, `generate`, and
`groundedness_check` — it dispatches to whichever provider `LLM_PROVIDER` names
(Groq or Gemini) and fails fast with a friendly error after one quick retry on a rate limit,
rather than hanging on repeated backoff.

## Setup

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then fill in your real credentials
```

Required in `.env`:
```
QDRANT_URL=...
QDRANT_API_KEY=...

# Both are wired up as interchangeable LLM providers (src/config.py's
# LLM_PROVIDER, default "groq") - set whichever one(s) you'll use. Setting
# both lets you flip providers with just an env var + restart if one's
# free-tier quota gets rate-limited.
GROQ_API_KEY=...
GOOGLE_API_KEY=...
# LLM_PROVIDER=groq   # or "gemini" - defaults to "groq" if unset
```

`ANTHROPIC_API_KEY` is unused by the current pipeline - kept in config only
for an easy revert.

## Running the pipeline

Everything runs as a module (`python -m ...`) from the project root, so
`from src... import ...` style imports resolve correctly.

```bash
# Full ingestion: fetch -> extract -> chunk -> create collection -> embed + upload
python -m scripts.run_ingest --max-results 50

# Ask a question through the full graph (hyde -> retrieve -> rerank -> generate -> check -> retry)
python -m scripts.run_query "How does hybrid retrieval combine dense and sparse search?"
python -m scripts.run_query "..." --hyde   # use HyDE query rewriting

# Build the evaluation benchmark (resumable - safe to interrupt and re-run)
python -m scripts.run_eval --generate

# Score retrieval only
python -m scripts.run_eval --score

# Score the full pipeline (retrieval + generation + groundedness)
python -m scripts.run_eval --score --full
```

## Running the API + frontend locally

Two long-running servers, so use two terminals. Both use the same virtual
environment - the frontend's dependencies (`streamlit`, `requests`) are
kept in `streamlit_app/requirements.txt`, separate from the main
`requirements.txt`, since the API doesn't need them and the frontend
doesn't need the API's heavier ML dependencies (torch, sentence-transformers).

**Terminal 1 - backend (FastAPI):**
```bash
pip install -r requirements.txt                 # if not already installed
uvicorn api.main:app --reload --port 8000
```
Wait for `Application startup complete` in the log - that's the embedder +
reranker models finishing their warmup (see `api/main.py`'s `lifespan`).
Verify with:
```bash
curl http://localhost:8000/health
```
should return `"qdrant_connected": true`.

**Terminal 2 - frontend (Streamlit):**
```bash
pip install -r streamlit_app/requirements.txt   # if not already installed
streamlit run streamlit_app/app.py
```
Opens `http://localhost:8501` automatically. `RAG_API_URL` defaults to
`http://localhost:8000` (see `streamlit_app/app.py`) so no extra config is
needed for local dev.

## Running the API in Docker

Single container, models baked into the image at build time (see
`Dockerfile` for why - avoids a live Hugging Face Hub dependency at
container start).

```bash
docker build -t rag-api .
docker run --env-file .env -p 8000:8000 rag-api
curl http://localhost:8000/health
```

The container reads `$PORT` if set (falls back to 8000), for platforms that
inject it dynamically. The Streamlit frontend isn't containerized here -
run it locally per above, pointed at the container via `RAG_API_URL`:
```bash
RAG_API_URL=http://localhost:8000 streamlit run streamlit_app/app.py
```

## Results so far

**Reranking demonstrably rescues chunks that fusion buries.** A chunk
fusion ranked near the bottom of the top-20 (fusion score 0.1667) was
correctly identified by the cross-encoder as the single most relevant
result (rerank score 0.9860, promoted to rank 1) - concrete evidence the
two-stage retrieve-then-rerank design earns its added latency over fusion
alone.

**Groundedness checking correctly flags fabricated claims.** Tested
against a deliberately hallucinated answer containing a fake method name,
a fabricated benchmark statistic, and an invented technical requirement -
the checker correctly flagged all four unsupported claims individually,
each with a specific reason.

**Full benchmark run (30 questions, k=20):** Hit Rate@20 = 100%, MRR =
0.861, groundedness rate = 86.7% (26/30 answers fully grounded). See
[`data/processed/eval_results.json`](data/processed/eval_results.json)
for the per-question breakdown; reproduce with
`python -m scripts.run_eval --score --full`.

## Known limitations

Documented deliberately rather than fixed silently:

- **Boilerplate filtering is not general.** Validated against one paper
  only; a positional heuristic (cut before the first real section heading)
  would generalize better than the current keyword match, but isn't
  implemented yet.
- **PDF math/algorithm notation extracts poorly**, since PyMuPDF reads
  spatial layout, not LaTeX semantics. Accepted as a cost/fidelity
  tradeoff.
- **HyDE is not a strict improvement.** With no visibility into corpus
  content, it can confidently resolve an ambiguous query term in the wrong
  direction and degrade retrieval quality.
- **Reference-section cutoff uses a simple heading match** - may miss
  papers with different formatting or column-layout extraction artifacts.
- **arXiv's `all:` search matches loosely on constituent words, not the
  intended phrase**, pulling in off-topic papers (~14% by a title audit
  of the fetched corpus). Left unfiltered to surface this as a real
  failure mode of keyword-based dataset construction rather than hide it.
- **The semantic cache only catches near-duplicate rewordings, not general
  paraphrases.** Calibration showed paraphrases and distinct questions
  don't separate cleanly at a low threshold, so it's set conservative
  (0.90) to avoid serving a wrong cached answer.
- **No automatic fallback between LLM providers.** A rate limit fails fast
  with a friendly error rather than silently retrying on the other
  provider - switching is a deliberate, manual env var change.

## Tech stack

- **Retrieval**: Qdrant Cloud (hybrid dense + sparse, native RRF fusion)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (dense, 384-dim), `Qdrant/bm25`
  via `fastembed` (sparse) - also reused for the semantic answer cache
- **Reranking**: `BAAI/bge-reranker-base` (cross-encoder)
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Orchestration**: LangGraph (`StateGraph` with conditional routing +
  retry edges)
- **LLM**: Groq (`openai/gpt-oss-120b`) and Gemini (`gemini-3.6-flash`),
  switchable via `LLM_PROVIDER`
- **PDF processing**: PyMuPDF
- **API**: FastAPI, with per-IP rate limiting (`slowapi`)
- **Frontend**: Streamlit (thin HTTP client over the API)
- **Containerization**: Docker (single container, models baked in at build
  time)
