from sentence_transformers import SentenceTransformer
import numpy as np
import os


class EmbeddingGenerator:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def encode(self, texts):

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings


if __name__ == "__main__":

    texts = [
        "Machine learning is transforming industries.",
        "Deep learning uses neural networks.",
        "Python is popular for AI development."
    ]

    embedder = EmbeddingGenerator()

    embeddings = embedder.encode(texts)

    print("\n===== EMBEDDING INFORMATION =====")
    print(f"Shape: {embeddings.shape}")
    print(f"Datatype: {embeddings.dtype}")

    print("\n===== FIRST EMBEDDING VECTOR =====")
    print(embeddings[0])

    print("\n===== VECTOR DIMENSION =====")
    print(len(embeddings[0]))

    os.makedirs("embeddings", exist_ok=True)

    np.save(
        "embeddings/document_embeddings.npy",
        embeddings
    )

    print(
        "\nEmbeddings saved to:"
        " embeddings/document_embeddings.npy"
    )