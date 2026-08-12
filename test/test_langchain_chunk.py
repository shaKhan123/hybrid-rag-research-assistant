import json
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

in_path = "texts.jsonl"
out_path = "chunks.jsonl"

def cut_references(text: str) -> str:
    ref_match = re.search(r"\n\s*References\s*\n", text, flags=re.IGNORECASE)
    if ref_match:
        return text[: ref_match.start()]
    return text

def is_boilerplate(text: str) -> bool:
    signals = ["University", "Institute", "Tel.:", "Tel:", "Fax:", "E-mail:", "Email:"]
    hits = sum(1 for s in signals if s in text)
    return len(text) < 400 and hits >= 2

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

total_chunks = 0
total_removed = 0
papers_processed = 0

with open(in_path, "r", encoding="utf-8") as in_f, \
     open(out_path, "w", encoding="utf-8") as out_f:

    for line in in_f:
        if not line.strip():
            continue
        paper = json.loads(line)

        full_text = cut_references(paper["full_text"])

        doc = Document(
            page_content=full_text,
            metadata={"arxiv_id": paper["arxiv_id"], "num_pages": paper["num_pages"]},
        )

        chunks = splitter.split_documents([doc])
        before = len(chunks)
        chunks = [c for c in chunks if not is_boilerplate(c.page_content)]
        removed = before - len(chunks)

        for i, chunk in enumerate(chunks):
            record = {
                "arxiv_id": paper["arxiv_id"],
                "chunk_index": i,
                "text": chunk.page_content,
                "char_count": len(chunk.page_content),
            }
            out_f.write(json.dumps(record) + "\n")

        total_chunks += len(chunks)
        total_removed += removed
        papers_processed += 1
        print(f"{paper['arxiv_id']}: {len(chunks)} chunks kept, {removed} removed")

print(f"\nDone. {papers_processed} papers processed.")
print(f"Total chunks written: {total_chunks}")
print(f"Total boilerplate chunks removed: {total_removed}")
print(f"Output: {out_path}")