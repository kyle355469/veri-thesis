from __future__ import annotations

import re
from typing import List

from .embeddings import Embedder
from .types import RetrievalHit
from .vector_store import VectorStore

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 8) -> List[RetrievalHit]:
        query_vector = self.embedder.encode([query])[0]
        return self.store.search(query_vector, top_k=top_k)


class LexicalReranker:
    """Small reranker that rewards shared RTL/interface terms."""

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int = 4) -> List[RetrievalHit]:
        query_terms = set(TOKEN_RE.findall(query.lower()))
        reranked: List[RetrievalHit] = []
        for hit in hits:
            doc_terms = set(TOKEN_RE.findall(hit.document.retrieval_text.lower()))
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            tag_bonus = 0.03 * len(set(hit.document.tags) & query_terms)
            hit.rerank_score = (0.70 * hit.score) + (0.30 * overlap) + tag_bonus
            reranked.append(hit)
        return sorted(reranked, key=lambda item: item.rerank_score or item.score, reverse=True)[:top_k]
