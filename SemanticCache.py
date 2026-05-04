import time
import json
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple


class SemanticCache:
    """
    A simple prototype semantic cache.

    It stores:
        query text
        query embedding
        cached result
        metadata
        timestamps

    Lookup:
        new query -> embedding -> cosine similarity search
        if max similarity >= threshold -> cache hit
        else -> cache miss
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], np.ndarray],
        threshold: float = 0.85,
        max_size: int = 1000,
    ):
        """
        Args:
            embedding_fn:
                Function that converts text into a numpy vector.

            threshold:
                Cosine similarity threshold for cache hit.
                Higher = safer but fewer hits.
                Lower = more hits but higher false-hit risk.

            max_size:
                Maximum number of cache entries.
        """
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.max_size = max_size

        self.entries: List[Dict[str, Any]] = []

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)

        if norm == 0:
            raise ValueError("Embedding vector has zero norm.")

        return vec / norm

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def get(
        self,
        query: str,
        return_score: bool = False,
    ) -> Optional[Any]:
        """
        Search cache by semantic similarity.

        Args:
            query:
                User query.

            return_score:
                If True, return (result, score, matched_query).

        Returns:
            cache result if hit, otherwise None.
        """
        if not self.entries:
            return None

        query_emb = self._normalize(self.embedding_fn(query))

        best_score = -1.0
        best_entry = None

        for entry in self.entries:
            score = self._cosine_similarity(query_emb, entry["embedding"])

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.threshold:
            best_entry["last_access_time"] = time.time()
            best_entry["hit_count"] += 1

            if return_score:
                return best_entry["result"], best_score, best_entry["query"]

            return best_entry["result"]

        return None

    def get_highest_sim_pair(
        self,
        query: str,
        k:int = 1,
    ) -> Optional[Any]:
        if not self.entries:
            return None

        query_emb = self._normalize(self.embedding_fn(query))

        best_score = [-1.0] * k
        best_entry = [None] * k

        for entry in self.entries:
            score = self._cosine_similarity(query_emb, entry["embedding"])

            if score > best_score[-1]:
                # a new entry with higher score than the lowest in the top-k list
                best_score[-1] = score
                best_entry[-1] = entry
                # sort the top-k list in descending order of score
                sorted_indices = np.argsort(best_score)[::-1]
                best_score = [best_score[i] for i in sorted_indices]
                best_entry = [best_entry[i] for i in sorted_indices]

        return best_score, best_entry

    
    def put(
        self,
        query: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert a new query-result pair into cache.
        """
        query_emb = self._normalize(self.embedding_fn(query))

        entry = {
            "query": query,
            "embedding": query_emb,
            "result": result,
            "metadata": metadata or {},
            "created_time": time.time(),
            "last_access_time": time.time(),
            "hit_count": 0,
        }

        self.entries.append(entry)

        if len(self.entries) > self.max_size:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """
        Remove the least recently used entry.
        """
        self.entries.sort(key=lambda x: x["last_access_time"])
        self.entries.pop(0)

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def stats(self) -> Dict[str, Any]:
        total_hits = sum(entry["hit_count"] for entry in self.entries)

        return {
            "num_entries": len(self.entries),
            "total_hits": total_hits,
            "threshold": self.threshold,
            "max_size": self.max_size,
        }
    
    def load_cache_from_json(self, path: str)-> None:
        with open(path, "r") as f:
            data = json.load(f)
            
        for entry in data:
            self.put(
                query=entry["query"],
                result=entry["result"]
            )
    
    def load_cache_from_jsonl(self, path: str)-> None:
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.put(
                    query=entry["query"],
                    result=entry["result"]
                )