from __future__ import annotations

from typing import Any, Dict

from rag_rtl.types import RetrievalHit

from .constants import CRITERIA, METADATA_ALIASES
from .serialization import metadata_json
from .types import IpCandidate


def candidate_from_hit(hit: RetrievalHit) -> IpCandidate:
    metadata = dict(hit.document.metadata or {})
    return IpCandidate(
        doc_id=hit.document.doc_id,
        score=hit.score,
        rerank_score=hit.rerank_score,
        tags=list(hit.document.tags),
        problem=hit.document.problem,
        solution=hit.document.solution,
        metadata=metadata,
        criteria={criterion: metadata_value(metadata, criterion) for criterion in CRITERIA},
    )


def metadata_value(metadata: Dict[str, Any], criterion: str) -> str:
    for key in METADATA_ALIASES[criterion]:
        if key in metadata and metadata[key] not in (None, ""):
            value = metadata[key]
            if isinstance(value, (dict, list)):
                return metadata_json(value)
            return str(value)
    return "unknown"
