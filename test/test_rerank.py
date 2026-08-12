from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

query = "How does hybrid retrieval combine dense and sparse search methods?"

# Paste in the actual chunk texts from your last hybrid query output
candidates = [
    "to the capability of our hybrid queries to effectively utilize available metadata to source the most pertinent results Implementing dense vector-based (KNN) semantic search results in a marked improvement over keyword-based search approaches.• Employing semantic search-based hybrid queries reali ...",
    "sparse methods excel in precise keyword matching, whilst dense methods effectively capture semantic similarity. 5.4.1 BGE-M3 and Unified Retrieval Models BGE-M3 is a premier unified model that enables the retrieval of dense, sparse, and multi-vector data across over 100 languages, with a maximum of  ...",
    "hybrid retrieval combines dense vector search with sparse keyword search (like BM25) to improve information retrieval. Dense methods capture semantic meaning, while sparse methods excel in exact keyword matching. By fusing both approaches, hybrid retrieval enhances accuracy and relevance of results.",
    "lenses across various fields.• Best Fields: Pursues the aggregation of words within a singular field. • Phrase Prefix: Operates similarly to Best Fields but prioritizes phrases over keywords. After initial match queries, we incorporate dense vector (KNN) and sparse encoder indices, each with their  ... ",
    "79.6 Trec Covid NDCG@10 80.4 HotpotQA F1 , EM 0.85 While most of prior efforts in improving RAG accuracy is on G part, by tweaking LLM prompts, tuning etc.,[9] they have limited impact on the overall accuracy of the RAG system, since if R part is feeding irreverent context then answer would be inacc ..."]

pairs = [[query, c] for c in candidates]
scores = reranker.predict(pairs)

for score, text in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True):
    print(f"Score: {score:.4f}")
    print(text[:200], "...")
    print()