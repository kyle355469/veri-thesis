from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .types import RtlDocument, RetrievalHit


class VectorStore:
    def __init__(self, documents: List[RtlDocument], vectors: np.ndarray):
        self.documents = documents
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if len(self.documents) != len(self.vectors):
            raise ValueError("Document/vector count mismatch")

    def search(self, query_vector: np.ndarray, top_k: int = 8) -> List[RetrievalHit]:
        if not self.documents:
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        scores = self.vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalHit(document=self.documents[int(idx)], score=float(scores[int(idx)]))
            for idx in top_indices
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        print(f"Saving vector store to {directory}...")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "documents.jsonl").open("w", encoding="utf-8") as handle:
            for document in self.documents:
                handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")
        documents: List[RtlDocument] = []
        with (directory / "documents.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                documents.append(RtlDocument(**payload))
        return cls(documents, vectors)


def build_vector_store(documents: Iterable[RtlDocument], vectors: np.ndarray) -> VectorStore:
    return VectorStore(list(documents), vectors)
