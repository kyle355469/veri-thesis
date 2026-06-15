from __future__ import annotations

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

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

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self.model.get_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def _chunk_texts(texts: Sequence[str], jobs: int) -> List[List[str]]:
    worker_count = min(jobs, len(texts))
    chunk_size = (len(texts) + worker_count - 1) // worker_count
    return [list(texts[index : index + chunk_size]) for index in range(0, len(texts), chunk_size)]


def _encode_hash_chunk(dim: int, texts: List[str]) -> np.ndarray:
    return HashingEmbedder(dim=dim).encode(texts)


def encode_texts(embedder: Embedder, texts: Iterable[str], jobs: int = 1) -> np.ndarray:
    text_list = list(texts)
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if jobs == 1 or len(text_list) <= 1:
        return embedder.encode(text_list)

    chunks = _chunk_texts(text_list, jobs)
    if isinstance(embedder, HashingEmbedder):
        with ProcessPoolExecutor(max_workers=min(jobs, len(chunks))) as executor:
            vectors = list(executor.map(_encode_hash_chunk, [embedder.dim] * len(chunks), chunks))
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, len(chunks))) as executor:
            vectors = list(executor.map(embedder.encode, chunks))
    return np.vstack(vectors) if vectors else np.zeros((0, embedder.dim), dtype=np.float32)


def make_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashingEmbedder()
    if name.startswith("sentence-transformers/") or name.startswith("BAAI/"):
        return SentenceTransformerEmbedder(name)
    raise ValueError(f"Unknown embedder '{name}'. Use 'hash' or a sentence-transformers model name.")


DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_embedder_with_fallback(name: str = "auto", warn=print) -> tuple[Embedder, str]:
    """Resolve an embedder by name, falling back to HashingEmbedder when
    sentence-transformers (or its model download) is unavailable.

    Returns (embedder, resolved_name) where resolved_name is what should be
    recorded in index metadata and run reports ("hash" or the ST model name).
    """
    if name == "hash":
        return HashingEmbedder(), "hash"
    model_name = DEFAULT_ST_MODEL if name == "auto" else name
    try:
        return SentenceTransformerEmbedder(model_name), model_name
    except (ImportError, OSError) as exc:
        warn(
            f"[rag] sentence-transformers embedder '{model_name}' unavailable ({exc}); "
            "falling back to HashingEmbedder — semantic scores will be coarser, "
            "consider lowering --planner-retrieval-min-score"
        )
        return HashingEmbedder(), "hash"
