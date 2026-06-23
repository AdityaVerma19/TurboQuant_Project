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

from turboquant import TurboQuant
from evaluate import Evaluator

# Initialize TurboQuant
quantizer = TurboQuant(dimension)

# Compress the embeddings
q_vectors, scale, sign_bits = quantizer.compress(embeddings)

# Reconstruct the embeddings
reconstructed = quantizer.reconstruct(q_vectors, scale, sign_bits)

# Build a retriever for reconstructed embeddings
retriever_reconstructed = VectorRetriever(dimension)
retriever_reconstructed.build(reconstructed)

# Search with the same query
scores_rec, ids_rec = retriever_reconstructed.search(query, k=3)

# Calculate metrics
mse = Evaluator.mse(embeddings, reconstructed)
cosine_err = Evaluator.cosine_error(embeddings, reconstructed)
ratio = Evaluator.compression_ratio(embeddings, q_vectors)

# Build the complete report
report_lines = []

report_lines.append("========== ORIGINAL SEARCH RESULTS ==========")
for rank, idx in enumerate(ids[0]):
    report_lines.append(f"Rank {rank + 1}: {docs[idx]} (Score: {scores[0][rank]:.4f})")

report_lines.append("\n========== RECONSTRUCTED SEARCH RESULTS ==========")
for rank, idx in enumerate(ids_rec[0]):
    report_lines.append(f"Rank {rank + 1}: {docs[idx]} (Score: {scores_rec[0][rank]:.4f})")

report_lines.append("\n========== SEARCH COMPARISON ==========")
report_lines.append(f"{'Rank':<5} | {'Original (Uncompressed)':<45} | {'Reconstructed (Compressed)':<45}")
report_lines.append("-" * 105)
for rank in range(3):
    orig_doc = docs[ids[0][rank]]
    rec_doc = docs[ids_rec[0][rank]]
    # truncate strings for nice display
    orig_display = orig_doc[:42] + "..." if len(orig_doc) > 42 else orig_doc
    rec_display = rec_doc[:42] + "..." if len(rec_doc) > 42 else rec_doc
    report_lines.append(f"{rank + 1:<5} | {orig_display:<45} | {rec_display:<45}")

report_lines.append("\n============================================================")
report_lines.append("TURBOQUANT REPORT")
report_lines.append("============================================================")
report_lines.append(f"MSE: {mse:.6f}")
report_lines.append(f"Cosine Error: {cosine_err:.6f}")
report_lines.append(f"Compression Ratio: {ratio:.2f}x")
report_lines.append("============================================================")

report_content = "\n".join(report_lines)

# Print the complete report to the console
print("\n" + report_content)

# Save the complete report to evaluation_report.txt
with open("evaluation_report.txt", "w", encoding="utf-8") as f_report:
    f_report.write(report_content)

print("\n[INFO] Evaluation report saved to evaluation_report.txt")

# Save compressed embeddings (quantized vectors, scale, and sign bits) to the embeddings folder
import os
import numpy as np
os.makedirs("embeddings", exist_ok=True)
np.savez(
    "embeddings/compressed_embeddings.npz",
    q_vectors=q_vectors,
    scale=scale,
    sign_bits=sign_bits
)
print("[INFO] Compressed embeddings saved to embeddings/compressed_embeddings.npz")