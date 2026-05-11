from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .embeddings import Embedder
from .retrieval import LexicalReranker, Retriever
from .summarizer import ContextSummarizer
from .tool_calling import RtlToolExecutor
from .types import RetrievalHit
from .vector_store import VectorStore
from .verifier import RtlVerifier


@dataclass
class RetrievalContext:
    """Reusable retrieval stack for any pipeline stage."""

    retriever: Retriever
    reranker: LexicalReranker
    summarizer: ContextSummarizer

    @classmethod
    def from_store(cls, store: VectorStore, embedder: Embedder) -> "RetrievalContext":
        return cls(
            retriever=Retriever(store, embedder),
            reranker=LexicalReranker(),
            summarizer=ContextSummarizer(),
        )

    def prepare(self, query: str, retrieve_k: int, context_k: int) -> list[RetrievalHit]:
        hits = self.retriever.retrieve(query, top_k=retrieve_k)
        hits = self.reranker.rerank(query, hits, top_k=context_k)
        return self.summarizer.maybe_summarize(hits)

    def tool_executor(
        self,
        verifier: RtlVerifier,
        default_top_module: Optional[str] = None,
    ) -> RtlToolExecutor:
        return RtlToolExecutor(
            retriever=self.retriever,
            reranker=self.reranker,
            summarizer=self.summarizer,
            verifier=verifier,
            default_top_module=default_top_module,
        )
