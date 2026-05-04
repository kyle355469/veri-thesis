from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import Embedder
from .history_cache import HistorySemanticCache
from .llm import VllmClient, extract_code
from .monitor import Monitor
from .prompting import build_generation_prompt
from .retrieval import LexicalReranker, Retriever
from .summarizer import ContextSummarizer
from .types import Diagnostic, PipelineResponse, RtlTask, VerificationReport
from .vector_store import VectorStore
from .verifier import RtlVerifier


class RagRtlPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm_client: Optional[Any] = None,
        verifier: Optional[RtlVerifier] = None,
        cache_path: str | Path = "data/history_cache.json",
        monitor_path: str | Path = "runs/monitor.jsonl",
        cache_threshold: float = 0.90,
    ):
        self.retriever = Retriever(store, embedder)
        self.reranker = LexicalReranker()
        self.llm = llm_client or VllmClient.from_env()
        self.verifier = verifier or RtlVerifier()
        self.summarizer = ContextSummarizer()
        self.cache = HistorySemanticCache(embedder, cache_path, threshold=cache_threshold)
        self.monitor = Monitor(monitor_path)

    def run(self, task: RtlTask, retrieve_k: int = 8, context_k: int = 4) -> PipelineResponse:
        timings: Dict[str, float] = {}
        start = time.perf_counter()

        cache_entry = self.cache.get(task.prompt)
        timings["cache_s"] = time.perf_counter() - start
        if cache_entry:
            verification = self.verifier.verify(cache_entry.result)
            response = PipelineResponse(
                rtl=cache_entry.result,
                verification=verification,
                retrieved_doc_ids=[],
                cache_source="history",
                repair_attempts=0,
                timings=timings,
                metadata={"cache_score": cache_entry.metadata.get("last_score")},
            )
            self.monitor.log("pipeline_response", {"response": response})
            return response

        t0 = time.perf_counter()
        hits = self.retriever.retrieve(task.prompt, top_k=retrieve_k)
        hits = self.reranker.rerank(task.prompt, hits, top_k=context_k)
        hits = self.summarizer.maybe_summarize(hits)
        timings["retrieve_rerank_s"] = time.perf_counter() - t0

        diagnostics: List[Diagnostic] = []
        rtl = ""
        verification = VerificationReport(False, False, [])
        repair_attempts = 0
        for attempt in range(task.max_repair_attempts + 1):
            t0 = time.perf_counter()
            prompt = build_generation_prompt(task, hits, diagnostics if attempt else None)
            model_text = self.llm.complete(prompt)
            rtl = extract_code(model_text)
            timings[f"llm_attempt_{attempt}_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            verification = self.verifier.verify(rtl)
            timings[f"verify_attempt_{attempt}_s"] = time.perf_counter() - t0
            diagnostics = verification.diagnostics
            if verification.passed:
                break
            repair_attempts = attempt

        cache_metadata = {
            "verified": verification.passed,
            "retrieved_doc_ids": [hit.document.doc_id for hit in hits],
            "repair_attempts": repair_attempts,
        }
        if verification.passed or repair_attempts >= task.max_repair_attempts:
            self.cache.put(task.prompt, rtl, cache_metadata)

        timings["total_s"] = time.perf_counter() - start
        response = PipelineResponse(
            rtl=rtl,
            verification=verification,
            retrieved_doc_ids=[hit.document.doc_id for hit in hits],
            cache_source="miss",
            repair_attempts=repair_attempts,
            timings=timings,
            metadata=cache_metadata,
        )
        self.monitor.log("pipeline_response", {"response": response})
        return response
