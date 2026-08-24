# Hybrid RAG Research Assistant

A retrieval-augmented generation system over arXiv papers, built to explore
and evaluate the techniques that separate a production-grade RAG pipeline
from a naive "embed and stuff into context" chatbot: hybrid (dense + sparse)
retrieval, cross-encoder reranking, query rewriting, automated groundedness
verification with a LangGraph retry loop, and quantitative evaluation.

Corpus: ~50 arXiv papers on **retrieval-augmented generation** itself
(`cs.CL` / `cs.LG`).

## Project structure

```
rag-project/
├── .env.example
├── .gitignore
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
│   │   ├── llm_client.py      # shared call_llm() with retry/backoff
│   │   ├── prompts.py         # shared prompt-formatting helpers
│   │   ├── answer.py          # source-grounded answer generation
│   │   └── groundedness.py    # per-claim groundedness verification
│   │
│   ├── graph/
│   │   ├── state.py           # RAGState TypedDict
│   │   ├── nodes.py           # LangGraph node functions
│   │   └── pipeline.py        # the compiled StateGraph + conditional retry edge
│   │
│   └── evaluation/
│       ├── generate_benchmark.py  # auto-generate Q&A pairs from known chunks
│       └── metrics.py             # Hit Rate@k, MRR, groundedness pass rate
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
   |  hyde -> retrieve -> rerank -> generate -> groundedness_check                          |
   |            (src/retrieval/)      (src/generation/)      |          |                   |
   |                                                          v          v                   |
   |                                                   retry (loop)   end (return answer)    |
   +----------------------------------------------------------------------------------------+
```

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
ANTHROPIC_API_KEY=...
```

`GOOGLE_API_KEY` is optional — kept in config for easy revert to Gemini if
desired (see [Provider history](#provider-history) below).

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

*(Hit Rate@k / MRR / groundedness pass rate numbers from the full 30-question
benchmark: run `python -m scripts.run_eval --score --full` and update this
section with the output.)*

## Known limitations

Documented deliberately rather than fixed silently:

- **Boilerplate filtering is not general.** The chunking-stage filter for
  author-affiliation blocks was validated against one paper and only
  removed 1 chunk out of 2,609 in the full corpus - most papers' front
  matter noise likely isn't caught. A positional heuristic (cut everything
  before the first real section heading, e.g. "1 Introduction") would
  generalize far better than the current keyword-matching approach - noted
  as the next improvement, not yet implemented.
- **PDF math/algorithm notation extracts poorly**, since PyMuPDF reads
  spatial layout, not LaTeX semantics. Accepted as a cost/fidelity
  tradeoff - prose explaining a method still retrieves and answers
  correctly even when an adjacent equation is mangled.
- **HyDE is not a strict improvement.** On an ambiguous query ("how do
  people fix bad retrieval?"), the LLM interpreted "retrieval" as human
  cognitive memory retrieval rather than information retrieval, generating
  a fluent but off-topic hypothetical answer that actively degraded
  retrieval quality. HyDE has no visibility into corpus content, so
  ambiguous terms can be confidently resolved in the wrong direction.
- **Reference-section cutoff uses a simple heading match** - works
  reliably for a standalone "References" heading, may miss papers with
  different formatting or column-layout extraction artifacts.
- **arXiv's `all:` search matches loosely on constituent words, not the
  intended phrase.** A query for `all:retrieval augmented generation`
  also matches papers containing "augment(ed)" and "retrieval" separately,
  in unrelated senses — e.g. spatial *augmented reality*, *cognitive
  augmentation* in human-AI interaction research, or "retrieval" meaning
  human memory recall rather than information retrieval. Auditing the
  fetched corpus (57 papers) by title found ~8 papers (~14%) were
  off-topic by this pattern — e.g. "Reality Distortion Room: A Study of
  User Locomotion Responses to Spatial Augmented Reality Effects" and
  "Cognitive Dissonance Artificial Intelligence (CD-AI)" — despite
  matching the search query. This surfaced concretely when an
  auto-generated benchmark question ("How far does the virtual room move
  vertically during the Elevation Distortion treatment?") was clearly
  unrelated to retrieval-augmented generation. Left unfiltered rather than
  silently cleaned up, since it's a genuine, common failure mode of
  keyword-based dataset construction worth surfacing rather than hiding —
  the fix (quoting the exact phrase, `all:"retrieval-augmented
  generation"`, or restricting further by category) is straightforward
  but wasn't applied retroactively to avoid re-fetching an already-working
  corpus mid-evaluation.

## Provider history

This project has swapped LLM providers several times during development:
Gemini (free tier, hit rate-limit friction during a 30-call benchmark
generation run) -> Anthropic (hit a billing/insufficient-credits error,
API requires prepaid credits unlike Gemini's free tier) -> back to Gemini
(no credits required; rate limits are a pacing problem, not a volume
problem, given this project's realistic call count — see below — so the
existing retry/backoff logic was sufficient once re-applied).
`src/generation/llm_client.py` centralizes the provider behind one
`call_llm()` function so a further swap is a one-file change, not a
find-and-replace across the codebase.

**Call volume, for context:** a single query through the full pipeline
costs 2-3 LLM calls (generate + groundedness check, plus an optional HyDE
call); a failed groundedness check adds a retry (capped at
`MAX_GENERATION_RETRIES`). A full 30-question evaluation run costs
roughly 60-90 calls total. This is comfortably within any free tier's
total volume — the friction hit earlier in development was entirely about
requests-per-minute pacing (30 benchmark-generation calls fired
back-to-back), which the retry/backoff logic in `llm_client.py` resolves.

## Tech stack

- **Retrieval**: Qdrant Cloud (hybrid dense + sparse, native RRF fusion)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (dense, 384-dim), `Qdrant/bm25`
  via `fastembed` (sparse)
- **Reranking**: `BAAI/bge-reranker-base` (cross-encoder)
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Orchestration**: LangGraph (`StateGraph` with a conditional retry edge)
- **LLM**: Claude Haiku (Anthropic)
- **PDF processing**: PyMuPDF

## Roadmap

- [ ] Report real Hit Rate@k / MRR / groundedness-rate numbers from the
      full benchmark run
- [ ] Positional front-matter detection (better boilerplate filtering)
- [ ] FastAPI backend exposing `/query`
- [ ] Minimal frontend
- [ ] Containerize + deploy