from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "arxiv_rag_hybrid"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
dense_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")


def generate_hyde_answer(query: str) -> str:
    prompt = f"""Write a short, plausible-sounding paragraph that could answer this question, \
as if it were an excerpt from an academic paper. It's okay if the details aren't factually \
verified — the goal is just to match the style and vocabulary of real research writing.

Question: {query}

Paragraph:"""
    interaction = gemini.interactions.create(model="gemini-3.6-flash", input=prompt)
    return interaction.output_text.strip()


def hybrid_retrieve(text: str, k: int = 5):
    dense_vec = dense_embedder.embed_query(text)
    sparse_vec = list(sparse_embedder.embed([text]))[0]

    results = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=20),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=20,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
    )
    return results.points


def print_results(label, points):
    print(f"\n=== {label} ===")
    for i, p in enumerate(points, start=1):
        print(f"{i}. [{p.payload['arxiv_id']} chunk {p.payload['chunk_index']}] "
              f"(score: {p.score:.5f})")
        print("   ", p.payload["text"][:150], "...")


query = "how do people fix bad retrieval?"

print("Retrieving using RAW QUERY...")
raw_results = hybrid_retrieve(query)
print_results("Results from raw query", raw_results)

print("\nGenerating HyDE answer...")
hyde_answer = generate_hyde_answer(query)
print("HyDE paragraph:", hyde_answer[:200], "...\n")

print("Retrieving using HYDE ANSWER...")
hyde_results = hybrid_retrieve(hyde_answer)
print_results("Results from HyDE-expanded query", hyde_results)