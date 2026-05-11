from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import CacheConfig, FixedPipeConfig, RuntimeConfig, ToolCallingConfig
from ..datapath import YosysDatapathExtractor
from ..embeddings import Embedder
from ..generation import SECOND_STAGE_ACTIONS, RtlGenerationStage
from ..json_utils import dumps_json
from ..llm import VllmClient
from ..monitor import Monitor
from ..prompting import build_second_edition_prompt
from ..retrieval_context import RetrievalContext
from ..types import PipelineResponse, RetrievalHit, RtlDocument, RtlTask
from ..vector_store import VectorStore
from ..verifier import RtlVerifier
from .rag import RagRtlPipeline


class FixedPipeRtlPipeline:
    """Spec RAG, first-edition verification, graph RAG, second-edition verification."""

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
        generation_temperature: float = 0.4,
        max_tokens: int = 2048,
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
            generation_temperature=generation_temperature,
            max_tokens=max_tokens,
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
        self.structure_context = RetrievalContext.from_store(code_structure_store, embedder)
        self.structure_retriever = self.structure_context.retriever
        self.structure_reranker = self.structure_context.reranker
        self.summarizer = self.structure_context.summarizer
        self.llm = shared_llm
        self.verifier = shared_verifier
        self.monitor = Monitor(runtime_config.monitor_path)
        self.runtime_config = runtime_config
        self.tool_config = tool_config
        self.datapath_extractor = YosysDatapathExtractor(
            yosys_bin=fixed_pipe_config.yosys_bin,
            timeout_s=fixed_pipe_config.yosys_timeout_s,
        )
        self.second_edition_repair_attempts = fixed_pipe_config.second_edition_repair_attempts
        self.verbose_generation = runtime_config.verbose_generation
        self.generation_temperature = runtime_config.generation_temperature
        self.max_tokens = runtime_config.max_tokens
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
                    "description": (
                        "First-edition RTL did not pass verification, so graph build "
                        "and second-edition generation were skipped."
                    ),
                }
            )
            return self._finish_with_first_response(first_response, task, timings, metadata, llm_actions, start)

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
            return self._finish_with_first_response(first_response, task, timings, metadata, llm_actions, start)

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

        stage_result = self._second_edition_stage().run(
            task=task,
            max_attempts=self.second_edition_repair_attempts,
            build_prompt=lambda diagnostics, _attempt: build_second_edition_prompt(
                task=task,
                first_edition_rtl=first_response.rtl,
                first_edition_datapath=graph_text,
                structure_hits=structure_hits,
                diagnostics=diagnostics,
            ),
            llm_actions=llm_actions,
            timings=timings,
        )

        metadata["second_edition"] = {
            "passed": stage_result.verification.passed,
            "repair_attempts": stage_result.repair_attempts,
            "retrieved_doc_ids": structure_doc_ids,
        }
        response = PipelineResponse(
            rtl=stage_result.rtl,
            verification=stage_result.verification,
            retrieved_doc_ids=first_response.retrieved_doc_ids + structure_doc_ids,
            cache_source=first_response.cache_source,
            repair_attempts=first_response.repair_attempts + stage_result.repair_attempts,
            llm_actions=llm_actions,
            prompt=task.prompt,
            timings={**timings, "total_s": time.perf_counter() - start},
            metadata=metadata,
        )
        self.monitor.log("fixed_pipe_response", {"response": response})
        return response

    def _finish_with_first_response(
        self,
        first_response: PipelineResponse,
        task: RtlTask,
        timings: Dict[str, float],
        metadata: Dict[str, Any],
        llm_actions: List[Dict[str, Any]],
        start: float,
    ) -> PipelineResponse:
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

    def _prepare_structure_context(self, query: str, retrieve_k: int, context_k: int) -> List[RetrievalHit]:
        return self.structure_context.prepare(query, retrieve_k=retrieve_k, context_k=context_k)

    def _second_edition_stage(self) -> RtlGenerationStage:
        return RtlGenerationStage(
            llm_client=self.llm,
            verifier=self.verifier,
            retrieval_context=self.structure_context,
            runtime_config=self.runtime_config,
            tool_config=self.tool_config,
            verbose=self._verbose,
            actions=SECOND_STAGE_ACTIONS,
        )

    def _verbose(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.verbose_generation:
            return
        self.monitor.log(f"verbose_{event}", payload)
        print(f"[verbose:{event}] {dumps_json(payload)}")


def _as_document(prompt: str, rtl: str, doc_id: str) -> RtlDocument:
    return RtlDocument(doc_id=doc_id, problem=prompt, solution=rtl, metadata={"source": "fixed_pipe_first_edition"})
