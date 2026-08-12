import fitz
import re
import json
import os

pdf_dir = "pdfs"
out_path = "texts.jsonl"

def clean_text(text):
    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ blank lines into just 2 (keep paragraph breaks, drop excess)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are JUST a number (likely a page number)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove the arXiv watermark stamp, e.g. "arXiv:2312.10997v1 [cs.CL] 18 Dec 2023"
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[.*?\]\s*\d{1,2}\s\w+\s\d{4}", "", text)
    return text.strip()

pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs to process.")

with open(out_path, "w", encoding="utf-8") as out_f:
    for filename in pdf_files:
        arxiv_id = filename.replace(".pdf", "")
        pdf_path = os.path.join(pdf_dir, filename)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  [skip] couldn't open {filename}: {e}")
            continue

        pages = []
        for page_num, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            cleaned = clean_text(raw)
            pages.append({"page": page_num, "text": cleaned})
        doc.close()

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])

        record = {
            "arxiv_id": arxiv_id,
            "num_pages": len(pages),
            "pages": pages,
            "full_text": full_text,
        }

        out_f.write(json.dumps(record) + "\n")
        print(f"Extracted: {arxiv_id} ({len(pages)} pages, {len(full_text)} chars)")

print("Done. Output:", out_path)