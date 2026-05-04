from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import Embedder


@dataclass
class CacheEntry:
    query: str
    result: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    hit_count: int = 0


class HistorySemanticCache:
    def __init__(self, embedder: Embedder, path: str | Path, threshold: float = 0.88, max_size: int = 1000):
        self.embedder = embedder
        self.path = Path(path)
        self.threshold = threshold
        self.max_size = max_size
        self.entries: List[CacheEntry] = []
        if self.path.exists():
            self.load()

    def get(self, query: str) -> Optional[CacheEntry]:
        if not self.entries:
            return None
        query_vector = self.embedder.encode([query])[0]
        best_score = -1.0
        best_entry: Optional[CacheEntry] = None
        for entry in self.entries:
            score = float(np.dot(query_vector, np.asarray(entry.embedding, dtype=np.float32)))
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry and best_score >= self.threshold:
            best_entry.hit_count += 1
            best_entry.last_access_time = time.time()
            best_entry.metadata["last_score"] = best_score
            self.save()
            return best_entry
        return None

    def put(self, query: str, result: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        embedding = self.embedder.encode([query])[0].astype(float).tolist()
        self.entries.append(CacheEntry(query=query, result=result, embedding=embedding, metadata=metadata or {}))
        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda entry: entry.last_access_time)
            self.entries = self.entries[-self.max_size :]
        self.save()

    def load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.entries = [CacheEntry(**item) for item in payload]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(entry) for entry in self.entries], handle, ensure_ascii=False, indent=2)
