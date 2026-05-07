from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CacheConfig, FixedPipeConfig, RuntimeConfig, ToolCallingConfig
from .datapath import YosysDatapathExtractor
from .embeddings import Embedder
from .history_cache import CacheLookup, HistorySemanticCache, LlmKeywordExtractor
from .json_utils import append_jsonl, dumps_json, preview_text
from .llm import VllmClient, extract_code
from .monitor import Monitor
from .prompting import build_generation_prompt, build_second_edition_prompt
from .retrieval import LexicalReranker, Retriever
from .summarizer import ContextSummarizer
from .tool_calling import RTL_TOOL_SCHEMAS, RtlToolExecutor
from .types import Diagnostic, PipelineResponse, RetrievalHit, RtlDocument, RtlTask, VerificationReport
from .vector_store import VectorStore
from .verifier import RtlVerifier


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
        )
        tool_config = tool_config or ToolCallingConfig(
            enabled=enable_tool_calling,
            choice=tool_choice,
            max_rounds=max_tool_rounds,
        )
        self.retriever = Retriever(store, embedder)
        self.reranker = LexicalReranker()
        self.llm = llm_client or VllmClient.from_env()
        self.verifier = verifier or RtlVerifier()
        self.summarizer = ContextSummarizer()
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
        self.verbose_generation = runtime_config.verbose_generation
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
        llm_actions.append(
            {
                "action": "retrieval_context_prepared",
                "description": "Pipeline prepared context before asking the LLM to generate RTL.",
                "doc_ids": [hit.document.doc_id for hit in hits],
            }
        )

        diagnostics: List[Diagnostic] = []
        rtl = ""
        verification = VerificationReport(False, False, [])
        repair_attempts = 0
        for attempt in range(task.max_repair_attempts + 1):
            t0 = time.perf_counter()
            prompt = build_generation_prompt(
                task,
                hits,
                diagnostics if attempt else None,
                cache_lookup if cache_lookup.evidence_entry and not attempt else None,
            )
            llm_actions.append(
                {
                    "action": "llm_generation_attempt",
                    "attempt": attempt,
                    "description": "Asked the LLM to produce final RTL in a <final_rtl> block.",
                    "with_repair_diagnostics": bool(diagnostics if attempt else None),
                    "with_history_evidence": bool(cache_lookup.evidence_entry and not attempt),
                    "tool_calling_enabled": self.enable_tool_calling,
                }
            )
            self._verbose("generation_prompt", {"attempt": attempt, "prompt": prompt})
            model_text = self._complete_generation(prompt, task, llm_actions, attempt)
            self._verbose("raw_model_text", {"attempt": attempt, "text": model_text})
            rtl = extract_code(model_text)
            llm_actions.append(
                {
                    "action": "rtl_extracted",
                    "attempt": attempt,
                    "description": "Extracted RTL from the LLM response, preferring the <final_rtl> block.",
                    "rtl": rtl,
                    "rtl_preview": preview_text(rtl),
                }
            )
            self._verbose("extracted_rtl", {"attempt": attempt, "rtl": rtl})
            timings[f"llm_attempt_{attempt}_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            verification = self.verifier.verify(rtl, top_module=task.top_module)
            timings[f"verify_attempt_{attempt}_s"] = time.perf_counter() - t0
            repair_attempts = attempt
            diagnostics = verification.diagnostics
            llm_actions.append(
                {
                    "action": "verification_result",
                    "attempt": attempt,
                    "description": "Pipeline verified the RTL produced by this LLM attempt.",
                    "passed": verification.passed,
                    "syntax_passed": verification.syntax_passed,
                    "lint_passed": verification.lint_passed,
                    "failed_tools": [diagnostic.tool for diagnostic in verification.diagnostics if not diagnostic.passed],
                }
            )
            self._verbose("verification", {"attempt": attempt, "verification": verification})
            if verification.passed:
                break
            self._log_failed_attempt(
                task=task,
                rtl=rtl,
                verification=verification,
                attempt=attempt,
                retrieved_doc_ids=[hit.document.doc_id for hit in hits],
                cache_lookup=cache_lookup,
                final_attempt=attempt >= task.max_repair_attempts,
            )

        cache_metadata = {
            "verified": verification.passed,
            "retrieved_doc_ids": [hit.document.doc_id for hit in hits],
            "repair_attempts": repair_attempts,
            "cache_decision": cache_lookup.to_metadata(),
        }
        if verification.passed:
            self.cache.put(task.prompt, rtl, cache_metadata)

        timings["total_s"] = time.perf_counter() - start
        response = PipelineResponse(
            rtl=rtl,
            verification=verification,
            retrieved_doc_ids=[hit.document.doc_id for hit in hits],
            cache_source="history_evidence" if cache_lookup.evidence_entry else "miss",
            repair_attempts=repair_attempts,
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
        hits = self.retriever.retrieve(query, top_k=retrieve_k)
        hits = self.reranker.rerank(query, hits, top_k=context_k)
        return self.summarizer.maybe_summarize(hits)

    def _complete_generation(
        self,
        prompt: str,
        task: RtlTask,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
    ) -> str:
        if not self.enable_tool_calling or not hasattr(self.llm, "complete_with_tools"):
            model_text = self.llm.complete(prompt)
            llm_actions.append(
                {
                    "action": "llm_final_response",
                    "attempt": attempt,
                    "used_tools": False,
                    "content": model_text,
                    "content_preview": preview_text(model_text),
                }
            )
            return model_text
        executor = RtlToolExecutor(
            retriever=self.retriever,
            reranker=self.reranker,
            summarizer=self.summarizer,
            verifier=self.verifier,
            default_top_module=task.top_module,
        )
        return self.llm.complete_with_tools(
            prompt,
            tools=RTL_TOOL_SCHEMAS,
            tool_executor=executor.execute,
            tool_choice=self.tool_choice,
            max_tool_rounds=self.max_tool_rounds,
            action_recorder=lambda action: llm_actions.append({"attempt": attempt, **action}),
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


class FixedPipeRtlPipeline:
    """Pipeline matching thesis-fixed-pipe-flow.png.

    Stage 1 uses the spec VectorDB and verification fallback loop from
    RagRtlPipeline. If stage 1 passes, Yosys builds a datapath graph from the
    first edition, the graph text retrieves similar code structures, and the
    LLM produces a second edition that is verified again.
    """

    def __init__(
        self,
        spec_store: VectorStore,
        code_structure_store: VectorStore,
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
        enable_tool_calling: bool = False,
        tool_choice: Any = "auto",
        max_tool_rounds: int = 4,
        yosys_bin: str = "yosys",
        yosys_timeout_s: int = 30,
        second_edition_repair_attempts: int = 1,
        cache_config: Optional[CacheConfig] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        tool_config: Optional[ToolCallingConfig] = None,
        fixed_pipe_config: Optional[FixedPipeConfig] = None,
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
        )
        tool_config = tool_config or ToolCallingConfig(
            enabled=enable_tool_calling,
            choice=tool_choice,
            max_rounds=max_tool_rounds,
        )
        fixed_pipe_config = fixed_pipe_config or FixedPipeConfig(
            yosys_bin=yosys_bin,
            yosys_timeout_s=yosys_timeout_s,
            second_edition_repair_attempts=second_edition_repair_attempts,
        )
        shared_llm = llm_client or VllmClient.from_env()
        shared_verifier = verifier or RtlVerifier()
        self.first_stage = RagRtlPipeline(
            store=spec_store,
            embedder=embedder,
            llm_client=shared_llm,
            verifier=shared_verifier,
            cache_config=cache_config,
            runtime_config=runtime_config,
            tool_config=tool_config,
        )
        self.structure_retriever = Retriever(code_structure_store, embedder)
        self.structure_reranker = LexicalReranker()
        self.summarizer = ContextSummarizer()
        self.llm = shared_llm
        self.verifier = shared_verifier
        self.monitor = Monitor(runtime_config.monitor_path)
        self.datapath_extractor = YosysDatapathExtractor(
            yosys_bin=fixed_pipe_config.yosys_bin,
            timeout_s=fixed_pipe_config.yosys_timeout_s,
        )
        self.second_edition_repair_attempts = fixed_pipe_config.second_edition_repair_attempts
        self.verbose_generation = runtime_config.verbose_generation
        self.enable_tool_calling = tool_config.enabled
        self.tool_choice = tool_config.choice
        self.max_tool_rounds = tool_config.max_rounds

    def run(
        self,
        task: RtlTask,
        retrieve_k: int = 8,
        context_k: int = 4,
        structure_retrieve_k: int = 8,
        structure_context_k: int = 4,
    ) -> PipelineResponse:
        start = time.perf_counter()
        first_response = self.first_stage.run(task, retrieve_k=retrieve_k, context_k=context_k)
        llm_actions = list(first_response.llm_actions)
        timings = {f"first_stage_{key}": value for key, value in first_response.timings.items()}
        timings["first_stage_wall_s"] = time.perf_counter() - start
        metadata = {
            **first_response.metadata,
            "pipeline": "fixed_pipe",
            "first_edition": {
                "passed": first_response.verification.passed,
                "rtl": first_response.rtl,
                "retrieved_doc_ids": first_response.retrieved_doc_ids,
                "verification": first_response.verification,
            },
        }

        if not first_response.verification.passed:
            llm_actions.append(
                {
                    "action": "fixed_pipe_stop_after_first_edition",
                    "description": "First-edition RTL did not pass verification, so graph build and second-edition generation were skipped.",
                }
            )
            response = PipelineResponse(
                rtl=first_response.rtl,
                verification=first_response.verification,
                retrieved_doc_ids=first_response.retrieved_doc_ids,
                cache_source=first_response.cache_source,
                repair_attempts=first_response.repair_attempts,
                llm_actions=llm_actions,
                prompt=task.prompt,
                timings={**timings, "total_s": time.perf_counter() - start},
                metadata=metadata,
            )
            self.monitor.log("fixed_pipe_response", {"response": response})
            return response

        t0 = time.perf_counter()
        try:
            graphs = self.datapath_extractor.extract_document(
                _as_document(task.prompt, first_response.rtl, doc_id="first_edition")
            )
            graph_text = "\n\n".join(graph.retrieval_text() for graph in graphs)
            graph_error = None
        except Exception as exc:  # noqa: BLE001 - keep the pipeline reportable.
            graphs = []
            graph_text = ""
            graph_error = str(exc)
        timings["graph_build_s"] = time.perf_counter() - t0
        metadata["first_edition_datapath"] = {
            "graph_count": len(graphs),
            "error": graph_error,
            "summary": graph_text,
        }
        llm_actions.append(
            {
                "action": "yosys_graph_build",
                "description": "Built a datapath graph from first-edition RTL before code-structure retrieval.",
                "graph_count": len(graphs),
                "error": graph_error,
            }
        )
        if graph_error:
            response = PipelineResponse(
                rtl=first_response.rtl,
                verification=first_response.verification,
                retrieved_doc_ids=first_response.retrieved_doc_ids,
                cache_source=first_response.cache_source,
                repair_attempts=first_response.repair_attempts,
                llm_actions=llm_actions,
                prompt=task.prompt,
                timings={**timings, "total_s": time.perf_counter() - start},
                metadata=metadata,
            )
            self.monitor.log("fixed_pipe_response", {"response": response})
            return response

        t0 = time.perf_counter()
        structure_query = f"{task.prompt}\n\n{graph_text}".strip()
        structure_hits = self._prepare_structure_context(
            structure_query,
            retrieve_k=structure_retrieve_k,
            context_k=structure_context_k,
        )
        timings["code_structure_retrieve_rerank_s"] = time.perf_counter() - t0
        structure_doc_ids = [hit.document.doc_id for hit in structure_hits]
        metadata["code_structure_retrieval"] = {
            "doc_ids": structure_doc_ids,
            "query": structure_query,
        }
        llm_actions.append(
            {
                "action": "code_structure_context_prepared",
                "description": "Retrieved graph-wise code-structure context for second-edition generation.",
                "doc_ids": structure_doc_ids,
            }
        )

        diagnostics: List[Diagnostic] = []
        second_rtl = first_response.rtl
        second_verification = first_response.verification
        second_attempt = 0
        for attempt in range(self.second_edition_repair_attempts + 1):
            t0 = time.perf_counter()
            prompt = build_second_edition_prompt(
                task=task,
                first_edition_rtl=first_response.rtl,
                first_edition_datapath=graph_text,
                structure_hits=structure_hits,
                diagnostics=diagnostics if attempt else None,
            )
            self._verbose("second_edition_prompt", {"attempt": attempt, "prompt": prompt})
            llm_actions.append(
                {
                    "action": "second_edition_generation_attempt",
                    "attempt": attempt,
                    "description": "Asked the LLM for second-edition RTL using first-edition code, Yosys graph, and code-structure VectorDB context.",
                    "with_repair_diagnostics": bool(diagnostics if attempt else None),
                }
            )
            model_text = self._complete_second_edition_generation(prompt, task, llm_actions, attempt)
            self._verbose("second_edition_raw_model_text", {"attempt": attempt, "text": model_text})
            llm_actions.append(
                {
                    "action": "second_edition_raw_model_text",
                    "attempt": attempt,
                    "description": "Received raw model text for this second-edition generation attempt.",
                    "content": model_text,
                    "content_preview": preview_text(model_text),
                }
            )
            second_rtl = extract_code(model_text)
            self._verbose("second_edition_extracted_rtl", {"attempt": attempt, "rtl": second_rtl})
            llm_actions.append(
                {
                    "action": "second_edition_rtl_extracted",
                    "attempt": attempt,
                    "description": "Extracted second-edition RTL from the model response.",
                    "rtl": second_rtl,
                    "rtl_preview": preview_text(second_rtl),
                }
            )
            timings[f"second_edition_llm_attempt_{attempt}_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            second_verification = self.verifier.verify(second_rtl, top_module=task.top_module)
            timings[f"second_edition_verify_attempt_{attempt}_s"] = time.perf_counter() - t0
            diagnostics = second_verification.diagnostics
            second_attempt = attempt
            llm_actions.append(
                {
                    "action": "second_edition_verification_result",
                    "attempt": attempt,
                    "description": "Verified the second-edition RTL.",
                    "passed": second_verification.passed,
                    "syntax_passed": second_verification.syntax_passed,
                    "lint_passed": second_verification.lint_passed,
                    "failed_tools": [
                        diagnostic.tool for diagnostic in second_verification.diagnostics if not diagnostic.passed
                    ],
                }
            )
            if second_verification.passed:
                break

        metadata["second_edition"] = {
            "passed": second_verification.passed,
            "repair_attempts": second_attempt,
            "retrieved_doc_ids": structure_doc_ids,
        }
        response = PipelineResponse(
            rtl=second_rtl,
            verification=second_verification,
            retrieved_doc_ids=first_response.retrieved_doc_ids + structure_doc_ids,
            cache_source=first_response.cache_source,
            repair_attempts=first_response.repair_attempts + second_attempt,
            llm_actions=llm_actions,
            prompt=task.prompt,
            timings={**timings, "total_s": time.perf_counter() - start},
            metadata=metadata,
        )
        self.monitor.log("fixed_pipe_response", {"response": response})
        return response

    def _verbose(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.verbose_generation:
            return
        self.monitor.log(f"verbose_{event}", payload)
        print(f"[verbose:{event}] {dumps_json(payload)}")

    def _prepare_structure_context(self, query: str, retrieve_k: int, context_k: int) -> List[RetrievalHit]:
        hits = self.structure_retriever.retrieve(query, top_k=retrieve_k)
        hits = self.structure_reranker.rerank(query, hits, top_k=context_k)
        return self.summarizer.maybe_summarize(hits)

    def _complete_second_edition_generation(
        self,
        prompt: str,
        task: RtlTask,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
    ) -> str:
        if not self.enable_tool_calling or not hasattr(self.llm, "complete_with_tools"):
            return self.llm.complete(prompt)
        executor = RtlToolExecutor(
            retriever=self.structure_retriever,
            reranker=self.structure_reranker,
            summarizer=self.summarizer,
            verifier=self.verifier,
            default_top_module=task.top_module,
        )
        return self.llm.complete_with_tools(
            prompt,
            tools=RTL_TOOL_SCHEMAS,
            tool_executor=executor.execute,
            tool_choice=self.tool_choice,
            max_tool_rounds=self.max_tool_rounds,
            action_recorder=lambda action: llm_actions.append(
                {"attempt": attempt, "stage": "second_edition", **action}
            ),
        )


def _as_document(prompt: str, rtl: str, doc_id: str) -> RtlDocument:
    return RtlDocument(doc_id=doc_id, problem=prompt, solution=rtl, metadata={"source": "fixed_pipe_first_edition"})
