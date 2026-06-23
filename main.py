from embeddings import EmbeddingGenerator
from retrieval import VectorRetriever

with open(
    "data/document.txt",
    "r",
    encoding="utf-8"
) as f:

    docs = [
        line.strip()
        for line in f.readlines()
    ]

embedder = EmbeddingGenerator()

embeddings = embedder.encode(docs)

dimension = embeddings.shape[1]

retriever = VectorRetriever(
    dimension
)

retriever.build(
    embeddings
)

query = embedder.encode(
    ["What is vector search?"]
)

scores, ids = retriever.search(
    query,
    k=3
)

for rank, idx in enumerate(ids[0]):

    print(
        rank + 1,
        docs[idx]
    )

from turboquant import TurboQuant
from evaluate import Evaluator