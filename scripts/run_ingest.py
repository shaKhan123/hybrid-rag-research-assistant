"""
CLI entrypoint: run the full ingestion pipeline in order.

    fetch arXiv papers -> extract text -> chunk -> create Qdrant collection
    -> embed + upload

Usage:
    python -m scripts.run_ingest --max-results 50
    python -m scripts.run_ingest --max-results 50 --skip-collection-create
"""

import argparse
from pathlib import Path

from src.config import (
    ARXIV_DEFAULT_QUERY,
    ARXIV_DEFAULT_CATEGORIES,
    ARXIV_MAX_RESULTS,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    CHUNKS_PATH,
    validate_config,
)
from src.ingest.fetch_arxiv import fetch
from src.ingest.extract_text import extract_all
from src.chunking.chunker import chunk_all
from src.indexing.qdrant_store import create_hybrid_collection, upload_chunks


def main():
    validate_config()

    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline.")
    parser.add_argument("--query", type=str, default=ARXIV_DEFAULT_QUERY)
    parser.add_argument("--categories", nargs="*", default=ARXIV_DEFAULT_CATEGORIES)
    parser.add_argument("--max-results", type=int, default=ARXIV_MAX_RESULTS)
    parser.add_argument("--skip-collection-create", action="store_true",
                         help="Skip Qdrant collection creation (use if it already exists).")
    args = parser.parse_args()

    print("### Step 1/4: Fetching papers from arXiv ###")
    fetch(query=args.query, categories=args.categories, max_results=args.max_results,
          out_dir=Path(DATA_RAW_DIR))

    print("\n### Step 2/4: Extracting text from PDFs ###")
    extract_all(in_dir=Path(DATA_RAW_DIR), out_dir=Path(DATA_PROCESSED_DIR))

    print("\n### Step 3/4: Chunking ###")
    chunk_all(in_path=Path(DATA_PROCESSED_DIR) / "texts.jsonl", out_path=Path(CHUNKS_PATH))

    print("\n### Step 4/4: Indexing (Qdrant) ###")
    if not args.skip_collection_create:
        create_hybrid_collection()
    upload_chunks(chunks_path=Path(CHUNKS_PATH))

    print("\nIngestion pipeline complete.")


if __name__ == "__main__":
    main()
