import faiss
import numpy as np


class VectorRetriever:

    def __init__(self, dimension):

        print("\n[INFO] Initializing FAISS Vector Retriever...")
        print(f"[INFO] Embedding Dimension: {dimension}")

        self.index = faiss.IndexFlatIP(dimension)

        print("[INFO] Using IndexFlatIP (Inner Product Search)")
        print("[INFO] If vectors are normalized, Inner Product = Cosine Similarity")

    def build(self, vectors):

        print("\n[INFO] Building Vector Index...")

        print(f"[INFO] Number of vectors: {vectors.shape[0]}")
        print(f"[INFO] Vector dimension: {vectors.shape[1]}")

        faiss.normalize_L2(vectors)

        print("[INFO] Vectors normalized successfully.")

        self.index.add(
            vectors.astype(np.float32)
        )

        print(
            f"[INFO] Added {self.index.ntotal} vectors to FAISS index."
        )

    def search(self, query_vector, k=3):

        print("\n[INFO] Performing Semantic Search...")

        query = query_vector.copy()

        faiss.normalize_L2(query)

        print("[INFO] Query normalized.")

        scores, ids = self.index.search(
            query,
            k
        )

        print(f"[INFO] Top-{k} results retrieved.")

        return scores, ids