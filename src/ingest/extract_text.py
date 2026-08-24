"""
Extract clean text from downloaded PDFs.

Uses PyMuPDF (fitz) rather than pypdf because it preserves reading order
much better on two-column academic paper layouts, and gives us per-page
text so downstream chunking/citations can still reference page numbers.

Known limitation (documented, not fixed): math/algorithm notation extracts
poorly, since PyMuPDF reads spatial layout, not LaTeX semantics. Equations
and pseudocode blocks come out garbled. Accepted as a cost/fidelity
tradeoff — prose explaining a method still retrieves and answers correctly
even when an adjacent equation is mangled. A vision-based extractor (e.g.
Nougat) would be the fix, not currently integrated.
"""

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def clean_text(text: str) -> str:
    """Light cleanup: collapse whitespace, drop obvious header/footer noise."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop lines that are just a page number
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Drop arXiv's sideways watermark stamp, e.g. "arXiv:2312.10997v1 [cs.CL] 18 Dec 2023"
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


def extract_all(in_dir: Path, out_dir: Path):
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
            # Some records may predate the pdf_path field (older metadata.jsonl
            # from before this project's restructuring) — reconstruct it from
            # arxiv_id + in_dir rather than crashing the whole batch.
            pdf_path_str = record.get("pdf_path")
            if pdf_path_str:
                pdf_path = Path(pdf_path_str)
            else:
                pdf_path = in_dir / "pdfs" / f"{record['arxiv_id']}.pdf"

            if not pdf_path.exists():
                print(f"  [warn] PDF not found for {record.get('arxiv_id', '?')}: {pdf_path}")
                n_fail += 1
                continue
            try:
                extracted = extract_pdf(pdf_path)
            except Exception as e:
                # MuPDF sometimes warns internally (e.g. "cannot create appearance
                # stream for Screen annotations") without raising — that's harmless
                # noise from unrendered interactive PDF elements, not a real failure.
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


def main():
    parser = argparse.ArgumentParser(description="Extract text from downloaded arXiv PDFs.")
    parser.add_argument("--in-dir", type=str, default=DATA_RAW_DIR,
                         help="Directory containing metadata.jsonl and pdfs/.")
    parser.add_argument("--out-dir", type=str, default=DATA_PROCESSED_DIR,
                         help="Where to write texts.jsonl.")
    args = parser.parse_args()

    extract_all(in_dir=Path(args.in_dir), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()