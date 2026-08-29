"""Dense vector route: TF-IDF -> truncated SVD (LSA) -> cosine similarity.

Pillar I requires combining "keyword, category, and vector similarity", and the
constraints require the whole system to run in memory for light execution.

A transformer encoder (BGE-M3 and similar) would mean ~2GB of dependencies, a
model download at build time, and CPU inference per query -- against rules that
warn scoring may run with network disabled under CPU and timeout limits. Latent
semantic indexing gives real dense vectors with genuine synonymy generalization,
builds from the catalog itself in seconds, needs nothing but numpy, and cannot
fail at scoring time for lack of a network. The encoder is swappable: anything
implementing `encode(list[str]) -> ndarray` can replace this.
"""
from __future__ import annotations

import math

import numpy as np

from pipeline.textutil import normalize, terms

DIMENSIONS = 192
OVERSAMPLE = 12          # extra columns for randomized range-finding accuracy
POWER_ITERATIONS = 2     # sharpens the spectrum; TF-IDF decays slowly
MAX_TERMS_PER_DOC = 320
MIN_DOC_FREQUENCY = 2
MAX_DOC_FRACTION = 0.4   # drop terms in >40% of the catalog; they carry no signal
CHUNK = 2048
SEED = 20260829


class DenseIndex:
    """In-memory LSA index. Deterministic given the catalog and SEED."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.projection: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    # -- sparse construction -------------------------------------------------

    def _vectorize(self, documents: list[list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a row-normalized TF-IDF matrix in CSR form."""
        indptr = np.zeros(len(documents) + 1, dtype=np.int64)
        indices: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for row, tokens in enumerate(documents):
            if tokens:
                columns, counts = np.unique(
                    np.fromiter(
                        (self.vocabulary[t] for t in tokens if t in self.vocabulary),
                        dtype=np.int64,
                    ),
                    return_counts=True,
                )
            else:
                columns, counts = np.zeros(0, np.int64), np.zeros(0, np.int64)
            if columns.size:
                # sublinear tf damps long product descriptions
                weight = (1.0 + np.log(counts.astype(np.float32))) * self.idf[columns]
                norm = float(np.sqrt((weight * weight).sum()))
                if norm > 0:
                    weight /= norm
            else:
                weight = np.zeros(0, np.float32)
            indices.append(columns)
            values.append(weight.astype(np.float32))
            indptr[row + 1] = indptr[row] + columns.size
        flat_indices = np.concatenate(indices) if indices else np.zeros(0, np.int64)
        flat_values = np.concatenate(values) if values else np.zeros(0, np.float32)
        return indptr, flat_indices, flat_values

    def _matmul(self, indptr, indices, values, dense: np.ndarray) -> np.ndarray:
        """A @ dense, streamed in row blocks so nnz x k never materializes."""
        out = np.zeros((len(indptr) - 1, dense.shape[1]), dtype=np.float32)
        for start in range(0, len(indptr) - 1, CHUNK):
            stop = min(start + CHUNK, len(indptr) - 1)
            lo, hi = indptr[start], indptr[stop]
            if hi == lo:
                continue
            block = dense[indices[lo:hi]] * values[lo:hi, None]
            segments = indptr[start:stop + 1] - lo
            # reduceat needs non-empty segments; empty rows stay zero
            nonempty = np.nonzero(np.diff(segments))[0]
            if nonempty.size:
                sums = np.add.reduceat(block, segments[nonempty], axis=0)
                out[start + nonempty] = sums
        return out

    def _rmatmul(self, indptr, indices, values, dense: np.ndarray) -> np.ndarray:
        """A.T @ dense, same streaming strategy."""
        out = np.zeros((self.idf.size, dense.shape[1]), dtype=np.float32)
        for start in range(0, len(indptr) - 1, CHUNK):
            stop = min(start + CHUNK, len(indptr) - 1)
            lo, hi = indptr[start], indptr[stop]
            if hi == lo:
                continue
            rows = np.repeat(np.arange(start, stop), np.diff(indptr[start:stop + 1]))
            np.add.at(out, indices[lo:hi], dense[rows] * values[lo:hi, None])
        return out

    # -- build ---------------------------------------------------------------

    def build(self, texts: list[str]) -> None:
        documents = [terms(normalize(text))[:MAX_TERMS_PER_DOC] for text in texts]
        frequency: dict[str, int] = {}
        for tokens in documents:
            for token in set(tokens):
                frequency[token] = frequency.get(token, 0) + 1
        ceiling = max(2, int(len(documents) * MAX_DOC_FRACTION))
        kept = sorted(t for t, c in frequency.items() if MIN_DOC_FREQUENCY <= c <= ceiling)
        self.vocabulary = {token: i for i, token in enumerate(kept)}
        total = max(1, len(documents))
        self.idf = np.array(
            [math.log(1.0 + total / (1.0 + frequency[t])) for t in kept], dtype=np.float32
        )

        indptr, indices, values = self._vectorize(documents)
        width = min(self.dimensions + OVERSAMPLE, len(self.vocabulary), len(documents))
        rng = np.random.default_rng(SEED)
        omega = rng.standard_normal((len(self.vocabulary), width), dtype=np.float32)

        # Randomized range finder with power iterations.
        sample = self._matmul(indptr, indices, values, omega)
        for _ in range(POWER_ITERATIONS):
            sample, _ = np.linalg.qr(sample)
            sample = self._rmatmul(indptr, indices, values, sample)
            sample, _ = np.linalg.qr(sample)
            sample = self._matmul(indptr, indices, values, sample)
        basis, _ = np.linalg.qr(sample)

        projected = self._rmatmul(indptr, indices, values, basis).T   # width x vocabulary
        u, singular, vt = np.linalg.svd(projected, full_matrices=False)
        rank = min(self.dimensions, vt.shape[0])
        # Terms -> latent space. Queries use the same map, so they land in the
        # same space as documents without re-running the decomposition.
        self.projection = np.ascontiguousarray(vt[:rank].T.astype(np.float32))
        self.vectors = self._l2(self._matmul(indptr, indices, values, self.projection))

    @staticmethod
    def _l2(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.maximum(norms, 1e-8, out=norms)
        return (matrix / norms).astype(np.float32)

    # -- query ---------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        tokens = [t for t in terms(normalize(text)) if t in self.vocabulary]
        vector = np.zeros(self.projection.shape[1], dtype=np.float32)
        if not tokens:
            return vector
        columns, counts = np.unique(
            np.fromiter((self.vocabulary[t] for t in tokens), dtype=np.int64),
            return_counts=True,
        )
        weight = (1.0 + np.log(counts.astype(np.float32))) * self.idf[columns]
        norm = float(np.sqrt((weight * weight).sum()))
        if norm > 0:
            weight /= norm
        vector = weight @ self.projection[columns]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else vector

    def similarity(self, text: str, candidates: np.ndarray | None = None) -> np.ndarray:
        """Cosine similarity against all documents, or a candidate subset."""
        query = self.encode(text)
        if not query.any():
            return np.zeros(len(candidates) if candidates is not None else len(self.vectors),
                            dtype=np.float32)
        matrix = self.vectors if candidates is None else self.vectors[candidates]
        return matrix @ query
