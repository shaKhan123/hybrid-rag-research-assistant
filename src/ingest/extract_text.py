"""
Stage 2: Extract clean text from downloaded PDFs.

Uses PyMuPDF (fitz) rather than pypdf because it preserves reading order
much better on two-column academic paper layouts, and gives us per-page
text so downstream chunking can still cite page numbers.

Output: one JSON record per paper in data/processed/texts.jsonl, containing
the full text plus a list of (page_number, page_text) tuples so the chunker
can attribute chunks back to a page.

Usage:
    python extract_text.py --in-dir data/raw --out-dir data/processed
"""

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Light cleanup: collapse whitespace, drop obvious header/footer noise."""
    # Collapse runs of whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop common arXiv footer/header artifacts (arXiv id stamp, page numbers alone on a line)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[.*?\]\s*\d{1,2}\s\w+\s\d{4}", "", text)
    return text.strip()


def extract_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        pages.append({"page": page_num, "text": clean_text(raw)})
    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages if p["text"])
    return {"pages": pages, "full_text": full_text, "num_pages": len(pages)}


def main():
    parser = argparse.ArgumentParser(description="Extract text from downloaded arXiv PDFs.")
    parser.add_argument("--in-dir", type=str, default="data/raw",
                         help="Directory containing metadata.jsonl and pdfs/.")
    parser.add_argument("--out-dir", type=str, default="data/processed",
                         help="Where to write texts.jsonl.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = in_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found — run fetch_arxiv.py first."
        )

    out_path = out_dir / "texts.jsonl"
    n_ok, n_fail = 0, 0

    with open(metadata_path, "r", encoding="utf-8") as meta_f, \
         open(out_path, "w", encoding="utf-8") as out_f:

        records = [json.loads(line) for line in meta_f if line.strip()]

        for record in tqdm(records, desc="Extracting text"):
            pdf_path = Path(record["pdf_path"])
            if not pdf_path.exists():
                n_fail += 1
                continue
            try:
                extracted = extract_pdf(pdf_path)
            except Exception as e:
                print(f"  [warn] failed to extract {pdf_path.name}: {e}")
                n_fail += 1
                continue

            out_record = {
                "arxiv_id": record["arxiv_id"],
                "title": record["title"],
                "authors": record["authors"],
                "abstract": record["abstract"],
                "categories": record["categories"],
                "published": record["published"],
                **extracted,
            }
            out_f.write(json.dumps(out_record) + "\n")
            n_ok += 1

    print(f"Done. {n_ok} papers extracted, {n_fail} failed/skipped.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
