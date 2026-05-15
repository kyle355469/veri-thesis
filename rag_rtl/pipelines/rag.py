from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import CacheConfig, RuntimeConfig, ToolCallingConfig
from ..embeddings import Embedder
from ..generation import FIRST_STAGE_ACTIONS, RtlGenerationStage
from ..history_cache import CacheLookup, HistorySemanticCache, LlmKeywordExtractor
from ..json_utils import append_jsonl, dumps_json
from ..llm import VllmClient
from ..monitor import Monitor
from ..prompting import build_emergency_generation_prompt, build_generation_prompt
from ..retrieval_context import RetrievalContext
from ..types import PipelineResponse, RetrievalHit, RtlTask, VerificationReport
from ..vector_store import VectorStore
from ..verifier import RtlVerifier


class RagRtlPipeline:
    _failed_log_lock = threading.Lock()

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm_client: Optional[Any] = None,
        verifier: Optional[RtlVerifier] = None,
        cache_path: str | Path = "data/history_cache.json",
        monitor_path: str | Path = "runs/monitor.jsonl",
        cache_threshold: Optional[float] = None,
        cache_reuse_threshold: float = 0.95,
        cache_evidence_threshold: float = 0.88,
        cache_mode: str = "keywords",
        failed_log_path: str | Path = "runs/failed_attempts.jsonl",
        verbose_generation: bool = False,
        generation_temperature: float = 0.4,
        max_tokens: int = 2048,
        enable_tool_calling: bool = False,
        tool_choice: Any = "auto",
        max_tool_rounds: int = 4,
        cache_config: Optional[CacheConfig] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        tool_config: Optional[ToolCallingConfig] = None,
    ):
        cache_config = cache_config or CacheConfig(
            path=cache_path,
            threshold=cache_threshold,
            reuse_threshold=cache_reuse_threshold,
            evidence_threshold=cache_evidence_threshold,
            mode=cache_mode,
        )
        runtime_config = runtime_config or RuntimeConfig(
            monitor_path=monitor_path,
            failed_log_path=failed_log_path,
            verbose_generation=verbose_generation,
            generation_temperature=generation_temperature,
            max_tokens=max_tokens,
        )
        tool_config = tool_config or ToolCallingConfig(
            enabled=enable_tool_calling,
            choice=tool_choice,
            max_rounds=max_tool_rounds,
        )

        self.context = RetrievalContext.from_store(store, embedder)
        self.retriever = self.context.retriever
        self.reranker = self.context.reranker
        self.summarizer = self.context.summarizer
        self.llm = llm_client or VllmClient.from_env()
        self.verifier = verifier or RtlVerifier()
        keyword_extractor = LlmKeywordExtractor(self.llm) if cache_config.mode == "keywords" else None
        self.cache = HistorySemanticCache(
            embedder,
            cache_config.path,
            threshold=cache_config.threshold,
            reuse_threshold=cache_config.reuse_threshold,
            evidence_threshold=cache_config.evidence_threshold,
            mode=cache_config.mode,
            keyword_extractor=keyword_extractor,
        )
        self.monitor = Monitor(runtime_config.monitor_path)
        self.failed_log_path = Path(runtime_config.failed_log_path)
        self.runtime_config = runtime_config
        self.tool_config = tool_config
        self.verbose_generation = runtime_config.verbose_generation
        self.generation_temperature = runtime_config.generation_temperature
        self.max_tokens = runtime_config.max_tokens
        self.enable_tool_calling = tool_config.enabled
        self.tool_choice = tool_config.choice
        self.max_tool_rounds = tool_config.max_rounds

    def run(self, task: RtlTask, retrieve_k: int = 8, context_k: int = 4) -> PipelineResponse:
        timings: Dict[str, float] = {}
        llm_actions: List[Dict[str, Any]] = []
        start = time.perf_counter()

        cache_lookup = self.cache.lookup(task.prompt)
        timings["cache_s"] = time.perf_counter() - start
        self._verbose("cache_lookup", cache_lookup.to_metadata())
        cached_response = self._response_from_reusable_history(task, cache_lookup, timings)
        if cached_response:
            return cached_response

        t0 = time.perf_counter()
        hits = self._prepare_context(task.prompt, retrieve_k=retrieve_k, context_k=context_k)
        timings["retrieve_rerank_s"] = time.perf_counter() - t0
        retrieved_doc_ids = [hit.document.doc_id for hit in hits]
        llm_actions.append(
            {
                "action": "retrieval_context_prepared",
                "description": "Pipeline prepared context before asking the LLM to generate RTL.",
                "doc_ids": retrieved_doc_ids,
            }
        )

        stage_result = self._generation_stage().run(
            task=task,
            max_attempts=task.max_repair_attempts,
            build_prompt=lambda feedback, attempt: build_generation_prompt(
                task,
                hits,
                feedback,
                cache_lookup if cache_lookup.evidence_entry and not attempt else None,
            ),
            llm_actions=llm_actions,
            timings=timings,
            action_metadata=lambda attempt, _feedback: {
                "with_history_evidence": bool(cache_lookup.evidence_entry and not attempt),
                "tool_calling_enabled": self.tool_config.enabled,
            },
            build_emergency_prompt=lambda model_text, _attempt, _feedback: build_emergency_generation_prompt(
                task,
                model_text,
            ),
            on_failed_attempt=lambda rtl, verification, attempt, final_attempt: self._log_failed_attempt(
                task=task,
                rtl=rtl,
                verification=verification,
                attempt=attempt,
                retrieved_doc_ids=retrieved_doc_ids,
                cache_lookup=cache_lookup,
                final_attempt=final_attempt,
            ),
        )

        cache_metadata = {
            "verified": stage_result.verification.passed,
            "retrieved_doc_ids": retrieved_doc_ids,
            "repair_attempts": stage_result.repair_attempts,
            "cache_decision": cache_lookup.to_metadata(),
        }
        if stage_result.verification.passed:
            self.cache.put(task.prompt, stage_result.rtl, cache_metadata)

        timings["total_s"] = time.perf_counter() - start
        response = PipelineResponse(
            rtl=stage_result.rtl,
            verification=stage_result.verification,
            retrieved_doc_ids=retrieved_doc_ids,
            cache_source="history_evidence" if cache_lookup.evidence_entry else "miss",
            repair_attempts=stage_result.repair_attempts,
            llm_actions=llm_actions,
            prompt=task.prompt,
            timings=timings,
            metadata={
                **cache_metadata,
                "best_history_match": cache_lookup.best_history_match,
            },
        )
        self.monitor.log("pipeline_response", {"response": response})
        return response

    def _response_from_reusable_history(
        self,
        task: RtlTask,
        cache_lookup: CacheLookup,
        timings: Dict[str, float],
    ) -> Optional[PipelineResponse]:
        cache_entry = cache_lookup.reusable_entry
        if not cache_entry:
            return None
        verification = self.verifier.verify(cache_entry.result, top_module=task.top_module)
        response = PipelineResponse(
            rtl=cache_entry.result,
            verification=verification,
            retrieved_doc_ids=[],
            cache_source="history",
            repair_attempts=0,
            llm_actions=[
                {
                    "action": "history_cache_reuse",
                    "description": "No LLM generation was needed because a verified cache entry was reused.",
                    "matched_query": cache_entry.query,
                    "score": cache_lookup.score,
                }
            ],
            prompt=task.prompt,
            timings=timings,
            metadata={
                "cache_decision": cache_lookup.to_metadata(),
                "best_history_match": cache_lookup.best_history_match,
            },
        )
        self.monitor.log("pipeline_response", {"response": response})
        return response

    def _prepare_context(self, query: str, retrieve_k: int, context_k: int) -> List[RetrievalHit]:
        return self.context.prepare(query, retrieve_k=retrieve_k, context_k=context_k)

    def _generation_stage(self) -> RtlGenerationStage:
        return RtlGenerationStage(
            llm_client=self.llm,
            verifier=self.verifier,
            retrieval_context=self.context,
            runtime_config=self.runtime_config,
            tool_config=self.tool_config,
            verbose=self._verbose,
            actions=FIRST_STAGE_ACTIONS,
        )

    def _log_failed_attempt(
        self,
        task: RtlTask,
        rtl: str,
        verification: VerificationReport,
        attempt: int,
        retrieved_doc_ids: List[str],
        cache_lookup: CacheLookup,
        final_attempt: bool,
    ) -> None:
        record = {
            "time": time.time(),
            "prompt": task.prompt,
            "attempt": attempt,
            "final_attempt": final_attempt,
            "generated_rtl": rtl,
            "verification": verification,
            "retrieved_doc_ids": retrieved_doc_ids,
            "cache_decision": cache_lookup.to_metadata(),
        }
        with self._failed_log_lock:
            append_jsonl(self.failed_log_path, record)

    def _verbose(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.verbose_generation:
            return
        self.monitor.log(f"verbose_{event}", payload)
        print(f"[verbose:{event}] {dumps_json(payload)}")
