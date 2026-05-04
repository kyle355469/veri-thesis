from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+'[bdh][0-9a-fA-F_xzXZ]+|\d+")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        ...


@dataclass
class HashingEmbedder:
    """Deterministic dependency-light embedder for local prototyping."""

    dim: int = 768

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[bucket] += sign
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def make_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashingEmbedder()
    if name.startswith("sentence-transformers/") or name.startswith("BAAI/"):
        return SentenceTransformerEmbedder(name)
    raise ValueError(f"Unknown embedder '{name}'. Use 'hash' or a sentence-transformers model name.")
