"""
Evaluation metrics against the benchmark.

Two evaluation modes:
1. Retrieval-only: for each benchmark question, run hybrid retrieval and
   check whether the known-correct source chunk appears in the results,
   and at what rank. Reports Hit Rate@k and Mean Reciprocal Rank (MRR).
2. Full pipeline: additionally runs generation + groundedness checking
   for each question, reporting the fraction of answers that pass
   groundedness verification alongside the retrieval metrics.

Both modes are resumable: per-question results are written to a .jsonl
progress file as they complete (append + flush), and a re-run skips
questions already scored. This matters because full-pipeline evaluation
makes 2-3 LLM calls per question (60-90 total for a 30-question
benchmark) and has hit real rate-limit interruptions partway through
during this project's development — losing 25/30 completed results to a
crash on question 26 would otherwise mean redoing everything.
"""

import argparse
import json
from pathlib import Path

from src.config import BENCHMARK_PATH, EVAL_RESULTS_PATH, RETRIEVE_TOP_K
from src.retrieval.hybrid_search import hybrid_retrieve
from src.retrieval.rerank import rerank
from src.generation.answer import generate_answer
from src.generation.groundedness import check_groundedness


def find_rank(results: list[dict], target_arxiv_id: str, target_chunk_index: int):
    """Return the 1-based rank of the target chunk in results, or None if absent."""
    for rank, chunk in enumerate(results, start=1):
        if chunk["arxiv_id"] == target_arxiv_id and chunk["chunk_index"] == target_chunk_index:
            return rank
    return None


def load_benchmark(benchmark_path: Path) -> list[dict]:
    benchmark = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                benchmark.append(json.loads(line))
    return benchmark


def _progress_path(out_path: Path) -> Path:
    """Per-question progress file, kept alongside the final aggregate JSON."""
    return out_path.with_suffix(".progress.jsonl")


def _load_progress(progress_path: Path) -> dict:
    """Return {question_key: result_dict} for questions already scored."""
    completed = {}
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed[record["question"]] = record
    return completed


def _aggregate(results_log: list[dict], has_groundedness: bool, k: int) -> dict:
    n = len(results_log)
    reciprocal_ranks = [1.0 / r["rank"] if r["rank"] else 0.0 for r in results_log]
    hits = sum(1 for r in results_log if r["rank"] is not None)

    hit_rate = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    result = {
        "hit_rate_at_k": hit_rate,
        "mrr": mrr,
        "k": k,
        "num_questions": n,
        "per_question": results_log,
    }

    if has_groundedness:
        grounded_count = sum(1 for r in results_log if r.get("is_grounded"))
        result["groundedness_rate"] = grounded_count / n if n else 0.0

    return result


def evaluate_retrieval(benchmark_path: Path = Path(BENCHMARK_PATH),
                        k: int = RETRIEVE_TOP_K,
                        out_path: Path = Path(EVAL_RESULTS_PATH)) -> dict:
    """Evaluate retrieval quality only: Hit Rate@k and MRR. Resumable."""
    benchmark = load_benchmark(benchmark_path)
    print(f"Loaded {len(benchmark)} benchmark questions.\n")

    progress_path = _progress_path(out_path)
    completed = _load_progress(progress_path)
    if completed:
        print(f"Found {len(completed)} already-scored questions — will skip those.\n")

    progress_path.parent.mkdir(parents=True, exist_ok=True)

    with open(progress_path, "a", encoding="utf-8") as prog_f:
        for i, item in enumerate(benchmark, start=1):
            query = item["question"]
            if query in completed:
                print(f"{i}. Skipping (already scored): {query[:70]}")
                continue

            target_id = item["source_arxiv_id"]
            target_chunk = item["source_chunk_index"]

            retrieved = hybrid_retrieve(query, k=k)
            rank = find_rank(retrieved, target_id, target_chunk)

            record = {"question": query, "target": f"{target_id}_{target_chunk}", "rank": rank}
            prog_f.write(json.dumps(record) + "\n")
            prog_f.flush()
            completed[query] = record

            status = f"FOUND at rank {rank}" if rank is not None else f"NOT FOUND in top {k}"
            print(f"{i}. [{status}] {query[:80]}")

    results_log = [completed[item["question"]] for item in benchmark]
    result = _aggregate(results_log, has_groundedness=False, k=k)

    print("\n" + "=" * 50)
    print(f"Questions evaluated: {result['num_questions']}")
    print(f"Hit Rate @ {k}: {result['hit_rate_at_k']:.2%}")
    print(f"MRR: {result['mrr']:.4f}")
    print("=" * 50)

    return result


def evaluate_full_pipeline(benchmark_path: Path = Path(BENCHMARK_PATH),
                            k: int = RETRIEVE_TOP_K,
                            out_path: Path = Path(EVAL_RESULTS_PATH)) -> dict:
    """Evaluate retrieval AND generation/groundedness quality end to end.
    Resumable — see module docstring.

    For each question: retrieve -> rerank -> generate -> groundedness check.
    Reports retrieval metrics (Hit Rate@k, MRR) alongside the fraction of
    generated answers that passed groundedness verification.
    """
    benchmark = load_benchmark(benchmark_path)
    print(f"Loaded {len(benchmark)} benchmark questions.\n")

    progress_path = _progress_path(out_path)
    completed = _load_progress(progress_path)
    if completed:
        print(f"Found {len(completed)} already-scored questions — will skip those.\n")

    progress_path.parent.mkdir(parents=True, exist_ok=True)

    with open(progress_path, "a", encoding="utf-8") as prog_f:
        for i, item in enumerate(benchmark, start=1):
            query = item["question"]
            if query in completed:
                print(f"{i}. Skipping (already scored): {query[:70]}")
                continue

            target_id = item["source_arxiv_id"]
            target_chunk = item["source_chunk_index"]

            retrieved = hybrid_retrieve(query, k=k)
            rank = find_rank(retrieved, target_id, target_chunk)

            reranked = rerank(query, retrieved)
            answer = generate_answer(query, reranked)
            groundedness = check_groundedness(answer, reranked)

            record = {
                "question": query,
                "target": f"{target_id}_{target_chunk}",
                "rank": rank,
                "is_grounded": groundedness["is_grounded"],
            }
            prog_f.write(json.dumps(record) + "\n")
            prog_f.flush()
            completed[query] = record

            print(f"{i}. [rank={rank}] [grounded={groundedness['is_grounded']}] {query[:70]}")

    results_log = [completed[item["question"]] for item in benchmark]
    result = _aggregate(results_log, has_groundedness=True, k=k)

    print("\n" + "=" * 50)
    print(f"Questions evaluated: {result['num_questions']}")
    print(f"Hit Rate @ {k}: {result['hit_rate_at_k']:.2%}")
    print(f"MRR: {result['mrr']:.4f}")
    print(f"Groundedness pass rate: {result['groundedness_rate']:.2%}")
    print("=" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally full pipeline) quality.")
    parser.add_argument("--benchmark-path", type=str, default=BENCHMARK_PATH)
    parser.add_argument("--out-path", type=str, default=EVAL_RESULTS_PATH)
    parser.add_argument("--k", type=int, default=RETRIEVE_TOP_K)
    parser.add_argument("--full", action="store_true",
                         help="Also evaluate generation + groundedness (slower, more LLM calls).")
    args = parser.parse_args()

    out_path = Path(args.out_path)

    if args.full:
        results = evaluate_full_pipeline(Path(args.benchmark_path), k=args.k, out_path=out_path)
    else:
        results = evaluate_retrieval(Path(args.benchmark_path), k=args.k, out_path=out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()