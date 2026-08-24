"""
Generate an evaluation benchmark: Q&A pairs auto-generated from known
source chunks, sampled across different papers.

Since each question is generated FROM a specific chunk, we automatically
know the "correct" source for that question — no manual labeling needed.

Resumable — tracks which (arxiv_id, chunk_index) pairs already have a
benchmark entry and skips them on re-run, writing with append + flush
after each one. This matters a lot given free-tier LLM rate limits can
interrupt a 30-question run partway through (happened twice during this
project's development) — nothing is lost on interruption.
"""

import argparse
import json
import random
import time
from pathlib import Path

from src.config import (
    CHUNKS_PATH,
    BENCHMARK_PATH,
    MIN_CONTENT_CHUNK_CHARS,
    EVAL_BENCHMARK_SIZE,
    EVAL_RANDOM_SEED,
)
from src.generation.llm_client import call_llm

# Pause between successive generation calls, independent of the retry/backoff
# logic in llm_client.py — free tier rate limits are tight enough that a
# burst of calls can exhaust the quota faster than backoff alone recovers
# from it. Pacing proactively avoids hitting the limit in the first place.
_PACING_DELAY_SECONDS = 4.0

_QUESTION_PROMPT_TEMPLATE = """Read this excerpt from an academic paper. Write ONE specific question \
that this excerpt directly answers. The question should be answerable using only this \
excerpt, phrased the way a curious researcher would naturally ask it — not overly \
formal or textbook-like.

Excerpt:
{chunk_text}

Question:"""


def generate_question_from_chunk(chunk_text: str) -> str:
    prompt = _QUESTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    return call_llm(prompt)


def load_completed_keys(benchmark_path: Path) -> set:
    """Return (arxiv_id, chunk_index) pairs already in the benchmark file."""
    completed = set()
    if benchmark_path.exists():
        with open(benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    completed.add((item["source_arxiv_id"], item["source_chunk_index"]))
    return completed


def sample_chunks(chunks_path: Path, n_papers: int, seed: int) -> list[dict]:
    """Sample one chunk from each of n_papers different papers, preferring
    chunks with real content (above MIN_CONTENT_CHUNK_CHARS) over stray
    short fragments."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    by_paper = {}
    for c in chunks:
        by_paper.setdefault(c["arxiv_id"], []).append(c)

    random.seed(seed)
    paper_ids = list(by_paper.keys())
    random.shuffle(paper_ids)
    sample_paper_ids = paper_ids[:n_papers]

    selected = []
    for pid in sample_paper_ids:
        candidates = by_paper[pid]
        good_candidates = [c for c in candidates if c["char_count"] > MIN_CONTENT_CHUNK_CHARS]
        chosen = random.choice(good_candidates if good_candidates else candidates)
        selected.append(chosen)

    return selected


def generate_benchmark(chunks_path: Path = Path(CHUNKS_PATH),
                        benchmark_path: Path = Path(BENCHMARK_PATH),
                        n_papers: int = EVAL_BENCHMARK_SIZE,
                        seed: int = EVAL_RANDOM_SEED):
    completed_keys = load_completed_keys(benchmark_path)
    if completed_keys:
        print(f"Found {len(completed_keys)} already-generated benchmark questions — will skip those.")

    selected_chunks = sample_chunks(chunks_path, n_papers, seed)
    print(f"Selected {len(selected_chunks)} chunks from {len(selected_chunks)} different papers.\n")

    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    with open(benchmark_path, "a", encoding="utf-8") as out_f:
        for i, chunk in enumerate(selected_chunks):
            key = (chunk["arxiv_id"], chunk["chunk_index"])
            if key in completed_keys:
                print(f"{i + 1}. Skipping (already done): {chunk['arxiv_id']} chunk {chunk['chunk_index']}")
                continue

            question = generate_question_from_chunk(chunk["text"])

            record = {
                "question": question,
                "source_arxiv_id": chunk["arxiv_id"],
                "source_chunk_index": chunk["chunk_index"],
                "source_text": chunk["text"],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            completed_keys.add(key)

            print(f"{i + 1}. Q: {question}")
            print(f"   (ground truth: {chunk['arxiv_id']} chunk {chunk['chunk_index']})")

            time.sleep(_PACING_DELAY_SECONDS)

    print(f"\nDone. Total benchmark entries: {len(completed_keys)}")
    print(f"Saved to {benchmark_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate an evaluation benchmark from chunks.jsonl.")
    parser.add_argument("--chunks-path", type=str, default=CHUNKS_PATH)
    parser.add_argument("--benchmark-path", type=str, default=BENCHMARK_PATH)
    parser.add_argument("--n-papers", type=int, default=EVAL_BENCHMARK_SIZE)
    parser.add_argument("--seed", type=int, default=EVAL_RANDOM_SEED)
    args = parser.parse_args()

    generate_benchmark(
        chunks_path=Path(args.chunks_path),
        benchmark_path=Path(args.benchmark_path),
        n_papers=args.n_papers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()