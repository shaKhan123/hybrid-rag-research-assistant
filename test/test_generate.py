from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_answer(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(chunks))

    prompt = f"""Answer the question using ONLY the information in the sources below. \
If the sources don't contain enough information to answer, say so explicitly rather than guessing. \
Cite which source(s) support each claim using [Source N] notation.

Sources:
{context}

Question: {query}

Answer:"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )
    return interaction.output_text.strip()


# Use real chunk texts from your earlier hybrid retrieval output
chunks = [
    "sparse methods excel in precise keyword matching, whilst dense methods effectively capture semantic similarity. 5.4.1 BGE-M3 and Unified Retrieval Models...",
    "original question. 3.3.2 Hybrid Retrieval and Reranking For each sub-query, we employ a hybrid retrieval strategy to maximize recall...",
]

query = "How does hybrid retrieval combine dense and sparse search methods?"

answer = generate_answer(query, chunks)
print(answer)

def check_groundedness(answer: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(chunks))

    prompt = f"""You are a fact-checker. Given the sources and a generated answer, \
identify any claims in the answer that are NOT directly supported by the sources. \
Be strict — if a claim is a reasonable inference but not explicitly stated, flag it too.

Sources:
{context}

Answer to check:
{answer}

For each claim in the answer, respond with:
- SUPPORTED: [claim] — [which source supports it]
- UNSUPPORTED: [claim] — [why it's not backed by the sources]

If everything is fully supported, just say "All claims are grounded in the sources."
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )
    return interaction.output_text.strip()


groundedness_report = check_groundedness(answer, chunks)
print(groundedness_report)

fake_answer = """Hybrid retrieval combines dense and sparse search by first running BM25 \
to get an initial candidate set, then re-ranking those candidates using a fine-tuned \
cross-encoder called ColBERT-v3. This approach was shown to improve NDCG@10 by 12% on \
the MS MARCO benchmark, and it requires GPU inference at query time to work effectively."""

groundedness_report = check_groundedness(fake_answer, chunks)
print(groundedness_report)