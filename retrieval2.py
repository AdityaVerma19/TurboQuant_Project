from embeddings import EmbeddingGenerator
from retrieval import VectorRetriever

documents = [
    "Machine learning is transforming industries.",
    "Deep learning uses neural networks.",
    "Python is popular for AI development.",
    "Vector databases store embeddings efficiently.",
    "FAISS enables fast similarity search."
]

embedder = EmbeddingGenerator()

print("\n[INFO] Generating Embeddings...")

embeddings = embedder.encode(documents)

print(f"[INFO] Embedding Shape: {embeddings.shape}")

retriever = VectorRetriever(
    embeddings.shape[1]
)

retriever.build(embeddings)

query = embedder.encode(
    ["What is vector search?"]
)

scores, ids = retriever.search(
    query,
    k=3
)

print("\n========== SEARCH RESULTS ==========\n")

for rank, idx in enumerate(ids[0]):

    print(f"Rank {rank + 1}")
    print(f"Document : {documents[idx]}")
    print(f"Score    : {scores[0][rank]:.4f}")
    print("-" * 50)