from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EmbeddingModelInfo:
    model_name: str
    backend: str
    device: str
    version: str


class EmbeddingLayer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._backend = "hashing"
        self._model: Any | None = None
        self._load_model()

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_name=self.model_name,
            backend=self._backend,
            device=self.device,
            version="1",
        )

    def _load_model(self) -> None:
        if self.model_name.lower() in {"hashing", "hashing://default"}:
            self._model = None
            self._backend = "hashing"
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sentence_transformer"
        except (ImportError, OSError, RuntimeError, ValueError):
            self._model = None
            self._backend = "hashing"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._backend == "sentence_transformer" and self._model is not None:
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vectors.astype(np.float32)
        return self._hashing_encode(texts)

    def _hashing_encode(self, texts: list[str], dims: int = 384) -> np.ndarray:
        matrix = np.zeros((len(texts), dims), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in text.lower().split():
                col = hash(token) % dims
                matrix[row_idx, col] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def cosine_similarity_batch(query_vectors: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
    return np.matmul(query_vectors, reference_vectors.T)


def load_embeddings_npz(path: str | Path) -> np.ndarray:
    blob = np.load(Path(path), allow_pickle=False)
    return blob["embeddings"]
