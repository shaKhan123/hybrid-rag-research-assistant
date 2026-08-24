"""
CLI entrypoint: build the benchmark and/or score against it.

Usage:
    python -m scripts.run_eval --generate            # build/resume benchmark
    python -m scripts.run_eval --score                # retrieval-only metrics
    python -m scripts.run_eval --score --full          # + generation/groundedness
"""

import argparse
import sys

# LLM-generated question text can contain Unicode punctuation (smart quotes,
# en-dashes) that the default Windows console codepage can't encode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.generate_benchmark import generate_benchmark
from src.evaluation.metrics import evaluate_retrieval, evaluate_full_pipeline
from src.config import EVAL_RESULTS_PATH
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description="Build and/or score the evaluation benchmark.")
    parser.add_argument("--generate", action="store_true", help="Generate/resume the benchmark.")
    parser.add_argument("--score", action="store_true", help="Score retrieval (or full pipeline) against the benchmark.")
    parser.add_argument("--full", action="store_true", help="With --score: also evaluate generation + groundedness.")
    args = parser.parse_args()

    if not args.generate and not args.score:
        parser.error("Specify at least one of --generate or --score.")

    if args.generate:
        generate_benchmark()

    if args.score:
        results = evaluate_full_pipeline() if args.full else evaluate_retrieval()

        out_path = Path(EVAL_RESULTS_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
