# Hybrid RAG Research Assistant

A retrieval-augmented generation system over arXiv papers, built to explore
and evaluate the techniques that separate a production-grade RAG pipeline
from a naive "embed and stuff into context" chatbot: hybrid (dense + sparse)
retrieval, cross-encoder reranking, query rewriting, automated groundedness
verification, and quantitative evaluation.

Corpus: ~50 arXiv papers on **retrieval-augmented generation** itself
(`cs.CL` / `cs.LG`) — chosen deliberately so the system's own subject matter
doubles as a natural test bed for questions about its own techniques.

## Why this project

Most RAG tutorials stop at "chunk -> embed -> cosine similarity -> stuff
into prompt." This project exists to go past that baseline and actually
measure whether each additional technique (hybrid search, reranking, query
rewriting, groundedness checks) earns its complexity — including
documenting where a technique *didn't* help, which turned out to be one of
the more useful findings.

## Architecture

```
arXiv API
   |
   v
Fetch metadata + PDFs  ------------------------------  src/ingest/fetch_arxiv.py
   |
   v
Extract text from PDFs (PyMuPDF)  -------------------  src/ingest/extract_text.py
   |
   v
Chunk (LangChain RecursiveCharacterTextSplitter
        + reference-section cutoff + boilerplate filter)
   |
   v
Embed (dense: BAAI/bge-small-en-v1.5, 384-dim
        sparse: Qdrant/bm25 via fastembed)
   |
   v
Qdrant Cloud (named vectors: "dense" + "sparse",
              server-side RRF fusion)
   |
   v
Hybrid retrieval (top 20)  --->  Cross-encoder rerank
(BAAI/bge-reranker-base)         (top 5)
   |
   v
Answer generation (LLM, source-cited)
   |
   v
Groundedness check (second LLM pass, flags
                      unsupported claims per-sentence)
```

Query rewriting (HyDE) sits upstream of retrieval as an optional alternate
path — see [Known limitations](#known-limitations) for why it's not a
strict improvement in all cases.

## What's implemented

- [x] **Ingestion** — arXiv API search + PDF download, resumable (skips
      already-fetched papers), rate-limited
- [x] **Text extraction** — PyMuPDF, per-page text preserved for future
      citation-to-page-number support
- [x] **Chunking** — LangChain `RecursiveCharacterTextSplitter`
      (1000 chars, 150 overlap), with a references-section cutoff and a
      boilerplate/affiliation-block filter (see limitations — the filter
      is not fully general)
- [x] **Hybrid retrieval** — dense + sparse vectors stored natively in
      Qdrant, fused server-side via Reciprocal Rank Fusion
      (`FusionQuery(fusion=Fusion.RRF)`), not a hand-rolled fusion loop
- [x] **Reranking** — `BAAI/bge-reranker-base` cross-encoder, second-pass
      over the top 20 hybrid candidates, empirically shown to promote
      genuinely relevant chunks that fusion alone ranked poorly (see
      [Results](#results-so-far))
- [x] **Query rewriting (HyDE)** — implemented and evaluated; helps on some
      queries, actively hurts on others (documented, not glossed over)
- [x] **Answer generation** — source-grounded prompting with
      `[Source N]` citation instructions
- [x] **Groundedness checking** — second LLM pass verifies each claim in
      the generated answer against the retrieved sources, flags
      unsupported claims individually with a reason
- [x] **LangGraph nodes** — retrieval, reranking, generation, and
      groundedness checking are implemented as LangGraph-compatible node
      functions (`state in -> dict update out`)
- [ ] **LangGraph orchestration** — the actual `StateGraph` with a
      conditional edge (retry generation if not grounded) is not yet wired
      up; nodes are currently chained by hand for testing
- [ ] **Evaluation harness** — benchmark generation (auto-generating
      questions from known source chunks) and retrieval metrics
      (Hit Rate@k, MRR) are built and working; full-scale run (30 questions)
      in progress
- [ ] **FastAPI backend + frontend**
- [ ] **Deployment**

## Setup

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (never commit this — see `.gitignore`):

```
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
GOOGLE_API_KEY=your-gemini-api-key
```

## Pipeline stages (run in order)

```bash
# 1. Fetch papers from arXiv
python src/ingest/fetch_arxiv.py --query "retrieval augmented generation" \
  --categories cs.CL cs.LG --max-results 50 --out-dir data/raw

# 2. Extract text from PDFs
python src/ingest/extract_text.py --in-dir data/raw --out-dir data/processed

# 3. Chunk (LangChain-based)
python test_langchain_chunk.py

# 4. Create Qdrant collection with dense + sparse vector slots
python create_hybrid_collection.py

# 5. Embed and upload all chunks
python upload_hybrid.py

# 6. Query end-to-end (hybrid retrieve -> rerank -> generate -> groundedness check)
python test_langgraph.py

# 7. Build evaluation benchmark (auto-generates Q&A pairs from your own chunks)
python test_eval_generate.py

# 8. Score retrieval quality against the benchmark
python test_eval_retrieval.py
```

## Results so far

**Reranking demonstrably rescues chunks that fusion buries.** Example from a
real query (`"How does hybrid retrieval combine dense and sparse search
methods?"`):

| Rank (fusion) | Rank (after rerank) | Fusion score | Rerank score |
|---|---|---|---|
| ~5-6 | **1** | 0.1667 | **0.9860** |

A chunk fusion ranked near the bottom of the top-20 was correctly identified
by the cross-encoder as the single most relevant result — concrete evidence
the two-stage retrieve-then-rerank design earns its added latency/complexity
over fusion alone.

**Groundedness checking correctly flags fabricated claims.** Tested against
a deliberately hallucinated answer containing a fake method name, a
fabricated benchmark statistic, and an invented technical requirement — the
checker correctly flagged all four unsupported claims individually, with a
specific reason for each (not just a single "this seems wrong" verdict).

*(Full retrieval Hit Rate@k / MRR numbers pending — evaluation harness run
in progress against a 30-question benchmark.)*

## Known limitations

Documented deliberately rather than fixed silently, in the interest of
being upfront about real engineering tradeoffs:

- **Boilerplate filtering is not general.** The chunking-stage filter for
  author-affiliation blocks (university names, phone/fax/email lines) was
  validated against one paper and only removed 1 chunk out of 2,609 total —
  meaning most papers' front-matter noise likely isn't being caught. A more
  robust approach (e.g., detecting front matter positionally, before the
  abstract) would be a natural next iteration.
- **PDF math/algorithm notation extracts poorly.** Equations and pseudocode
  blocks get garbled by position-based text extraction (PyMuPDF reads
  spatial layout, not LaTeX semantics). Accepted as a cost/fidelity
  tradeoff for this project rather than integrating a vision-based
  extractor (e.g. Nougat) — prose explaining a method still retrieves and
  answers correctly even when the adjacent equation is mangled.
- **HyDE is not a strict improvement.** On a well-specified query, HyDE and
  raw-query retrieval returned different but comparably relevant results.
  On an ambiguous query ("how do people fix bad retrieval?"), the LLM
  interpreted "retrieval" as human cognitive memory retrieval rather than
  information retrieval — generating a fluent but off-topic hypothetical
  answer that then *actively degraded* retrieval quality. HyDE has no
  visibility into what's actually in the corpus, so ambiguous terms can be
  confidently resolved in the wrong direction. A corpus-anchored prompt
  (e.g. "...as if it were an excerpt from a computer science paper about
  information retrieval") would likely mitigate this — noted as a next
  step, not yet implemented.
- **Reference-section cutoff uses a simple heading match.** Works reliably
  when a paper has a standalone "References" heading; may miss papers using
  different formatting (all-caps headings, "Bibliography," or headings that
  merge with body text due to column-layout extraction artifacts).

## Tech stack

- **Retrieval**: Qdrant Cloud (hybrid dense + sparse, native RRF fusion)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (dense, 384-dim),
  `Qdrant/bm25` via `fastembed` (sparse)
- **Reranking**: `BAAI/bge-reranker-base` (cross-encoder)
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Orchestration**: LangGraph (nodes implemented; graph wiring in
  progress)
- **LLM**: Google Gemini (`gemini-3.6-flash`)
- **PDF processing**: PyMuPDF

## Roadmap

- [ ] Wire nodes into an actual LangGraph `StateGraph` with a conditional
      retry edge (regenerate if groundedness check fails, up to N retries)
- [ ] Complete the 30-question evaluation benchmark and report Hit Rate@k /
      MRR numbers
- [ ] Add answer-level evaluation (RAGAS or similar) alongside retrieval
      metrics
- [ ] FastAPI backend exposing `/query`
- [ ] Minimal frontend
- [ ] Containerize + deploy
