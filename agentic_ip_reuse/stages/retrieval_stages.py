from __future__ import annotations

from typing import Any, Dict, List

from ..planning import decision_from_payload as _decision_from_payload
from ..prompts import build_candidate_evaluation_prompt
from ..retrieval import candidate_from_hit
from ..types import IpCandidate, LlmTrace, ModuleReuseDecision, ModuleSpec


class RetrievalStagesMixin:
    def _build_decisions(
        self,
        modules: List[ModuleSpec],
        llm_traces: List[LlmTrace],
        retrieval_traces: List[Dict[str, Any]],
    ) -> List[ModuleReuseDecision]:
        decisions: List[ModuleReuseDecision] = []
        for module in modules:
            query = module.reuse_query or f"{module.category} {module.name} {module.purpose}"
            self._stage("ip_search", "running", module=module.name, category=module.category, query=query)
            hits = self.retrieval_context.prepare(
                query=query,
                retrieve_k=self.config.retrieve_k,
                context_k=self.config.context_k,
            )
            # Merge hits from the live store (previously generated leaves in this session).
            if self._live_store is not None:
                embedder = self.retrieval_context.retriever.embedder
                query_vector = embedder.encode([query])[0]
                live_hits = self._live_store.search(query_vector, top_k=self.config.retrieve_k)
                seen_ids = {hit.document.doc_id for hit in hits}
                hits = hits + [h for h in live_hits if h.document.doc_id not in seen_ids]
            candidates = [candidate_from_hit(hit) for hit in hits]
            self._stage(
                "ip_search",
                "complete",
                module=module.name,
                candidate_count=len(candidates),
                doc_ids=[candidate.doc_id for candidate in candidates],
            )
            retrieval_traces.append(
                {
                    "module": module.name,
                    "category": module.category,
                    "query": query,
                    "doc_ids": [candidate.doc_id for candidate in candidates],
                }
            )
            decisions.append(self._evaluate_module(module, candidates, llm_traces))
        return decisions

    def _evaluate_module(
        self,
        module: ModuleSpec,
        candidates: List[IpCandidate],
        llm_traces: List[LlmTrace],
    ) -> ModuleReuseDecision:
        if not candidates:
            self._stage("ip_evaluation", "complete", module=module.name, action="new", reason="no_candidates")
            return ModuleReuseDecision(
                module=module,
                candidates=[],
                action="new",
                rationale="No candidates were retrieved from the IP index.",
            )
        self._stage(
            "ip_evaluation",
            "running",
            module=module.name,
            candidate_count=len(candidates),
            doc_ids=[candidate.doc_id for candidate in candidates],
        )
        payload = self._complete_json(
            f"ip_evaluation:{module.name}",
            build_candidate_evaluation_prompt(module, candidates),
            llm_traces,
        )
        decision = _decision_from_payload(module, candidates, payload)
        self._stage(
            "ip_evaluation",
            "complete",
            module=module.name,
            selected_doc_id=decision.selected_doc_id,
            action=decision.action,
        )
        return decision

