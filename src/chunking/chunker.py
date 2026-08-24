"""
Chunk extracted paper texts into retrieval-sized pieces.

Uses LangChain's RecursiveCharacterTextSplitter (industry-standard default —
tries to split on paragraph breaks first, then line breaks, then spaces,
only falling back to raw character splitting when it has to).

Two cleanup passes on top of the raw splitter, based on real noise patterns
found during development:

1. Reference-section cutoff — text after a "References" heading is dropped
   before chunking, since it's citation-list noise with no retrievable
   content value.
2. Boilerplate filter — chunks that look like author-affiliation blocks
   (university names, phone/fax/email lines, few real sentences) are
   dropped after chunking.

KNOWN LIMITATION (see README): the boilerplate filter was validated against
one paper and only caught 1 chunk out of 2,609 in the full corpus run —
meaning most papers' front-matter noise likely isn't caught by this
heuristic. Documented, not silently fixed, pending a more robust approach
(e.g. positional detection of front matter before the abstract).
"""

import argparse
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_PROCESSED_DIR,
    CHUNKS_PATH,
)

# Signals used by the boilerplate heuristic — affiliation/contact blocks
# tend to pack several of these into a short chunk.
_BOILERPLATE_SIGNALS = ["University", "Institute", "Tel.:", "Tel:", "Fax:", "E-mail:", "Email:"]
_BOILERPLATE_MAX_CHARS = 400
_BOILERPLATE_MIN_SIGNAL_HITS = 2


def cut_references(text: str) -> str:
    """Drop everything from a standalone 'References' heading onward.

    Known limitation: only matches a literal 'References' heading on its
    own line. Papers using 'REFERENCES' (all caps handled via IGNORECASE),
    'Bibliography', or headings that merge with body text due to
    column-layout extraction artifacts may not be caught.
    """
    ref_match = re.search(r"\n\s*References\s*\n", text, flags=re.IGNORECASE)
    if ref_match:
        return text[: ref_match.start()]
    return text


def is_boilerplate(text: str) -> bool:
    """Heuristic: short chunk + multiple affiliation/contact signals ==
    likely author-affiliation block rather than real content."""
    hits = sum(1 for s in _BOILERPLATE_SIGNALS if s in text)
    return len(text) < _BOILERPLATE_MAX_CHARS and hits >= _BOILERPLATE_MIN_SIGNAL_HITS


def chunk_paper(arxiv_id: str, full_text: str, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Chunk a single paper's text, applying both cleanup passes.

    Returns a list of plain dicts (not LangChain Document objects) — simpler
    to serialize to JSONL and pass through the rest of the pipeline.
    """
    cleaned_text = cut_references(full_text)

    doc = Document(page_content=cleaned_text, metadata={"arxiv_id": arxiv_id})
    raw_chunks = splitter.split_documents([doc])

    kept_chunks = [c for c in raw_chunks if not is_boilerplate(c.page_content)]

    return [
        {
            "arxiv_id": arxiv_id,
            "chunk_index": i,
            "text": chunk.page_content,
            "char_count": len(chunk.page_content),
        }
        for i, chunk in enumerate(kept_chunks)
    ]


def chunk_all(in_path: Path, out_path: Path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    total_chunks = 0
    papers_processed = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as in_f, \
         open(out_path, "w", encoding="utf-8") as out_f:

        for line in in_f:
            if not line.strip():
                continue
            paper = json.loads(line)

            chunks = chunk_paper(paper["arxiv_id"], paper["full_text"], splitter)

            for record in chunks:
                out_f.write(json.dumps(record) + "\n")

            total_chunks += len(chunks)
            papers_processed += 1
            print(f"{paper['arxiv_id']}: {len(chunks)} chunks kept")

    print(f"\nDone. {papers_processed} papers processed.")
    print(f"Total chunks written: {total_chunks}")
    print(f"Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted paper texts for retrieval.")
    parser.add_argument("--in-path", type=str,
                         default=str(Path(DATA_PROCESSED_DIR) / "texts.jsonl"),
                         help="Path to texts.jsonl (output of extract_text.py).")
    parser.add_argument("--out-path", type=str, default=CHUNKS_PATH,
                         help="Where to write chunks.jsonl.")
    args = parser.parse_args()

    chunk_all(in_path=Path(args.in_path), out_path=Path(args.out_path))


if __name__ == "__main__":
    main()
