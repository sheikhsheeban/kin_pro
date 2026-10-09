"""Text embeddings with EmbeddingGemma 300M (on-device, via moonshine-voice's native runtime).

EmbeddingGemma is Matryoshka-trained, so the 768-d output is truncated to the
first `dim` values and re-normalised (256-d by default; architecture §3).
It expects task prompts: queries and documents are embedded differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_query(self, text: str) -> np.ndarray: ...
    def embed_document(self, text: str) -> np.ndarray: ...
    def close(self) -> None: ...


def find_embedding_model(models_dir: Path) -> Path:
    hits = sorted(p.parent for p in Path(models_dir).glob("**/embeddinggemma-300m/tokenizer.bin"))
    if not hits:
        raise FileNotFoundError(f"EmbeddingGemma not found under {models_dir}. Run: uv run kin download-models")
    return hits[-1]


def truncate_normalise(vec: np.ndarray | list[float], dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)[:dim]
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("zero embedding")
    return v / norm


class GemmaEmbedder:
    def __init__(self, model_dir: Path, variant: str = "q4", dim: int = 256) -> None:
        from moonshine_voice.embedding_model import EmbeddingModel

        self._model = EmbeddingModel(str(model_dir), model_variant=variant)
        self.dim = dim
        self.name = f"embeddinggemma-300m-{variant}@{dim}"

    def embed_query(self, text: str) -> np.ndarray:
        return truncate_normalise(self._model.calculate_embedding(QUERY_PROMPT + text), self.dim)

    def embed_document(self, text: str) -> np.ndarray:
        return truncate_normalise(self._model.calculate_embedding(DOCUMENT_PROMPT + text), self.dim)

    def close(self) -> None:
        self._model.close()
