import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:

    @staticmethod
    def mse(
        original,
        reconstructed
    ):

        return np.mean(
            (
                original -
                reconstructed
            ) ** 2
        )

    @staticmethod
    def cosine_error(
        original,
        reconstructed
    ):

        original_sim = cosine_similarity(
            original
        )

        reconstructed_sim = cosine_similarity(
            reconstructed
        )

        return np.mean(
            np.abs(
                original_sim -
                reconstructed_sim
            )
        )

    @staticmethod
    def compression_ratio(
        original_vectors,
        quantized_vectors
    ):

        original_size = (
            original_vectors.size *
            original_vectors.itemsize
        )

        compressed_size = (
            quantized_vectors.size
        )

        return (
            original_size /
            compressed_size
        )

    @staticmethod
    def print_report(
        original,
        reconstructed,
        quantized
    ):

        mse = Evaluator.mse(
            original,
            reconstructed
        )

        cosine_err = (
            Evaluator.cosine_error(
                original,
                reconstructed
            )
        )

        ratio = (
            Evaluator.compression_ratio(
                original,
                quantized
            )
        )

        print("\n")
        print("=" * 60)
        print("TURBOQUANT REPORT")
        print("=" * 60)

        print(
            f"MSE: {mse:.6f}"
        )

        print(
            f"Cosine Error: {cosine_err:.6f}"
        )

        print(
            f"Compression Ratio: {ratio:.2f}x"
        )

        print("=" * 60)