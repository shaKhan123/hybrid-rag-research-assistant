"""
Quick diagnostic: print every paper's title in metadata.jsonl, so you can
eyeball how many are genuinely about retrieval-augmented generation vs.
off-topic results that slipped in from a loosely-matching arXiv query.

Usage:
    python check_corpus_titles.py
"""

import json
from pathlib import Path

metadata_path = Path("data/raw/metadata.jsonl")

with open(metadata_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total papers: {len(records)}\n")

for r in records:
    print(f"{r['arxiv_id']:>14}  {r['title']}")