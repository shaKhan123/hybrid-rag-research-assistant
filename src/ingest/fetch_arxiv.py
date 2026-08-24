"""
Fetch papers from the arXiv API: metadata + PDFs.

Resumable — skips papers already present in metadata.jsonl / already on
disk, so a re-run after an interruption (rate limit, crash, network drop)
picks up where it left off instead of starting over.
"""

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import arxiv
from tqdm import tqdm

from src.config import (
    ARXIV_DEFAULT_QUERY,
    ARXIV_DEFAULT_CATEGORIES,
    ARXIV_MAX_RESULTS,
    ARXIV_API_DELAY_SECONDS,
    DATA_RAW_DIR,
)


def load_existing_ids(metadata_path: Path) -> set:
    """Return arxiv_ids already recorded, so re-runs don't re-download."""
    if not metadata_path.exists():
        return set()
    ids = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["arxiv_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def build_query(query: str, categories: list[str]) -> str:
    """Combine a free-text query with arXiv category filters."""
    cat_filter = " OR ".join(f"cat:{c}" for c in categories) if categories else ""
    text_filter = f"all:{query}" if query else ""
    parts = [p for p in [text_filter, f"({cat_filter})" if cat_filter else ""] if p]
    return " AND ".join(parts)


def fetch(query: str, categories: list[str], max_results: int, out_dir: Path,
          delay_seconds: float = ARXIV_API_DELAY_SECONDS):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = out_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"

    existing_ids = load_existing_ids(metadata_path)
    if existing_ids:
        print(f"Found {len(existing_ids)} papers already fetched — will skip those.")

    search_query = build_query(query, categories)
    print(f"arXiv query: {search_query}")

    client = arxiv.Client(page_size=100, delay_seconds=delay_seconds, num_retries=5)
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    new_count = 0
    with open(metadata_path, "a", encoding="utf-8") as meta_f:
        for result in tqdm(client.results(search), total=max_results, desc="Fetching"):
            arxiv_id = result.get_short_id().split("v")[0]  # strip version suffix

            if arxiv_id in existing_ids:
                continue

            record = {
                "arxiv_id": arxiv_id,
                "title": result.title.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "abstract": result.summary.strip().replace("\n", " "),
                "categories": result.categories,
                "primary_category": result.primary_category,
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "pdf_path": str(pdf_dir / f"{arxiv_id}.pdf"),
            }

            pdf_path = pdf_dir / f"{arxiv_id}.pdf"
            if not pdf_path.exists():
                try:
                    # Result.download_pdf() was removed in arxiv package v4.0.0 —
                    # current recommended approach is a direct download via pdf_url.
                    urlretrieve(result.pdf_url, pdf_path)
                except Exception as e:
                    print(f"  [warn] failed to download {arxiv_id}: {e}")
                    continue

            meta_f.write(json.dumps(record) + "\n")
            meta_f.flush()
            existing_ids.add(arxiv_id)
            new_count += 1

    print(f"Done. {new_count} new papers fetched (total now: {len(existing_ids)}).")
    print(f"Metadata: {metadata_path}")
    print(f"PDFs:     {pdf_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv for a RAG corpus.")
    parser.add_argument("--query", type=str, default=ARXIV_DEFAULT_QUERY,
                         help="Free-text search query.")
    parser.add_argument("--categories", nargs="*", default=ARXIV_DEFAULT_CATEGORIES,
                         help="arXiv category codes to filter on (e.g. cs.CL cs.LG).")
    parser.add_argument("--max-results", type=int, default=ARXIV_MAX_RESULTS,
                         help="Max number of papers to fetch.")
    parser.add_argument("--out-dir", type=str, default=DATA_RAW_DIR,
                         help="Output directory for metadata.jsonl and pdfs/.")
    parser.add_argument("--delay-seconds", type=float, default=ARXIV_API_DELAY_SECONDS,
                         help="Delay between arXiv API page requests (be polite to the API).")
    args = parser.parse_args()

    fetch(
        query=args.query,
        categories=args.categories,
        max_results=args.max_results,
        out_dir=Path(args.out_dir),
        delay_seconds=args.delay_seconds,
    )


if __name__ == "__main__":
    main()
