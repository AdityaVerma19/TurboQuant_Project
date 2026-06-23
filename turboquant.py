import numpy as np


class TurboQuant:

    def __init__(self, dimension):

        self.dimension = dimension

        print("\n[TurboQuant] Initializing...")

        self.rotation = self._create_rotation_matrix()

        print(
            f"[TurboQuant] Rotation Matrix Shape: "
            f"{self.rotation.shape}"
        )

    def _create_rotation_matrix(self):

        """
        Generate orthogonal matrix using QR decomposition.
        """

        A = np.random.randn(
            self.dimension,
            self.dimension
        )

        Q, _ = np.linalg.qr(A)

        return Q.astype(np.float32)

    def rotate(self, vectors):

        print(
            "\n[TurboQuant] Applying Orthogonal Rotation..."
        )

        return vectors @ self.rotation

    def quantize_4bit(self, vectors):

        print(
            "\n[TurboQuant] Quantizing to 4-bit..."
        )

        xmax = np.max(np.abs(vectors))

        scale = xmax / 7.0

        q = np.round(vectors / scale)

        q = np.clip(q, -8, 7)

        return q.astype(np.int8), scale

    def dequantize(self, q_vectors, scale):

        print(
            "\n[TurboQuant] Dequantizing..."
        )

        return q_vectors.astype(
            np.float32
        ) * scale

    def residual_correction(
        self,
        rotated_vectors,
        reconstructed_vectors
    ):

        print(
            "\n[TurboQuant] Applying TurboVec-style Residual Correction..."
        )

        residual = (
            rotated_vectors -
            reconstructed_vectors
        )

        sign_bits = np.sign(
            residual
        )

        alpha = np.mean(
            np.abs(residual)
        )

        corrected = (
            reconstructed_vectors +
            alpha * sign_bits
        )

        return corrected

    def inverse_rotate(
        self,
        vectors
    ):

        print(
            "\n[TurboQuant] Inverse Rotation..."
        )

        return vectors @ self.rotation.T

    def compress(self, vectors):

        rotated = self.rotate(vectors)

        q_vectors, scale = (
            self.quantize_4bit(rotated)
        )

        deq = self.dequantize(
            q_vectors,
            scale
        )

        residual = rotated - deq

        sign_bits = np.sign(
            residual
        ).astype(np.int8)

        return (
            q_vectors,
            scale,
            sign_bits
        )

    def reconstruct(
        self,
        q_vectors,
        scale,
        sign_bits
    ):

        deq = self.dequantize(
            q_vectors,
            scale
        )

        alpha = scale / 2

        corrected = (
            deq +
            alpha * sign_bits
        )

        recovered = (
            self.inverse_rotate(
                corrected
            )
        )

        return recovered


print("TurboQuant")
print("Aditya")