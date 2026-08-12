import arxiv
import json
import os
import time
from urllib.request import urlretrieve

os.makedirs("pdfs", exist_ok=True)

existing_ids = set()
if os.path.exists("metadata.jsonl"):
    with open("metadata.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_ids.add(json.loads(line)["arxiv_id"])

print(f"Already have {len(existing_ids)} papers.")

search = arxiv.Search(
    query="all:retrieval augmented generation",
    max_results=50,   # bumped up from 5
    sort_by=arxiv.SortCriterion.Relevance,
)

# Configure the client to be polite: wait 3 sec between paginated requests,
# retry up to 5 times if arXiv's API hiccups
client = arxiv.Client(
    page_size=100,
    delay_seconds=3.0,
    num_retries=5,
)

with open("metadata.jsonl", "a", encoding="utf-8") as f:
    for result in client.results(search):
        arxiv_id = result.get_short_id().split("v")[0]

        if arxiv_id in existing_ids:
            print("Skipping (already have):", arxiv_id)
            continue

        record = {
            "arxiv_id": arxiv_id,
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "categories": result.categories,
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
            "abstract": result.summary.strip(),
        }

        pdf_path = os.path.join("pdfs", f"{arxiv_id}.pdf")
        urlretrieve(result.pdf_url, pdf_path)

        f.write(json.dumps(record) + "\n")
        f.flush()
        existing_ids.add(arxiv_id)
        print("Downloaded:", arxiv_id, "-", record["title"])

        time.sleep(1)  # small pause between PDF downloads specifically