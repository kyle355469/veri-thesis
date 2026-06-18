from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .repository import JsonIpRepository, matches_filters
from .types import IpAssessment, IpCandidate, IpDescription


@dataclass
class SearchTrace:
    query: str
    mode: str
    top_score: Optional[float]
    returned: int
    filtered_below_threshold: int
    min_score: float
    low_confidence: bool
    latency_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticIpRepository:
    """IpRepository implementation backed by embedding retrieval over the task catalog.

    The retriever/reranker are duck-typed (rag_rtl.retrieval.Retriever /
    LexicalReranker) and injected by the caller, so this package keeps no
    hard dependency on rag_rtl. Retrieved doc_ids must be ip_ids of the
    wrapped JsonIpRepository.
    """

    inner: JsonIpRepository
    retriever: Any
    reranker: Any = None
    mode: str = "semantic"  # semantic | hybrid
    min_score: float = 0.30
    below_threshold: str = "flag"  # flag | drop
    traces: List[SearchTrace] = field(default_factory=list)

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[IpCandidate]:
        filters = filters or {}
        top_k = max(1, min(int(top_k), 50))
        started = time.monotonic()
        hits = self.retriever.retrieve(query, top_k=3 * top_k)
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, top_k=3 * top_k)
        candidates: List[IpCandidate] = []
        filtered = 0
        for hit in hits:
            score = hit.rerank_score if getattr(hit, "rerank_score", None) is not None else hit.score
            try:
                description = self.inner.inspect(hit.document.doc_id)
            except KeyError:
                continue
            candidate = IpCandidate(**asdict(description.candidate))
            if not matches_filters(candidate, filters):
                continue
            candidate.score = round(float(score), 4)
            if candidate.score < self.min_score:
                filtered += 1
                if self.below_threshold == "drop":
                    continue
                candidate.criteria = dict(candidate.criteria)
                candidate.criteria["retrieval_confidence"] = "low"
            candidates.append(candidate)
        if self.mode == "hybrid":
            seen = {candidate.ip_id for candidate in candidates}
            for candidate in self.inner.search(query, filters=filters, top_k=top_k):
                if candidate.ip_id not in seen:
                    candidate.criteria = dict(candidate.criteria)
                    candidate.criteria["retrieval_source"] = "token"
                    candidates.append(candidate)
                    seen.add(candidate.ip_id)
        candidates = candidates[:top_k]
        confident = [c for c in candidates if c.criteria.get("retrieval_confidence") != "low"]
        self.traces.append(
            SearchTrace(
                query=query,
                mode=self.mode,
                top_score=candidates[0].score if candidates else None,
                returned=len(candidates),
                filtered_below_threshold=filtered,
                min_score=self.min_score,
                low_confidence=not confident,
                latency_s=round(time.monotonic() - started, 4),
            )
        )
        return candidates

    def list_candidates(self) -> List[IpCandidate]:
        return self.inner.list_candidates()

    def pop_last_trace(self) -> Optional[Dict[str, Any]]:
        return self.traces[-1].to_dict() if self.traces else None

    def inspect(self, ip_id: str) -> IpDescription:
        return self.inner.inspect(ip_id)

    def score(self, candidate: IpCandidate, module_requirements: Dict[str, Any]) -> IpAssessment:
        return self.inner.score(candidate, module_requirements)
