from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from rag_rtl.llm import extract_code
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

from .config import AgenticIpReuseConfig
from .manifest import (
    dependency_order as _dependency_order,
    manifest_validation_errors as _manifest_validation_errors,
    recursive_decomposition_validation_errors as _recursive_decomposition_validation_errors,
)
from .planning import (
    modules_from_payload as _modules_from_payload,
    requirements_from_payload as _requirements_from_payload,
)
from .prompts import (
    build_decomposition_prompt,
    build_repair_prompt,
    build_requirements_prompt,
    build_rtl_generation_prompt,
)
from .retrieval import candidate_from_hit
from .serialization import dumps_result
from .stages import (
    LargeSpecStagesMixin,
    LlmStagesMixin,
    PartitionStagesMixin,
    RecursiveStagesMixin,
    RetrievalStagesMixin,
    VerificationCoreMixin,
    VerificationStagesMixin,
)
from .types import (
    AgenticIpReuseResult,
    IpReusePlan,
    LlmTrace,
)


class AgenticIpReuseAgent(
    LargeSpecStagesMixin,
    RecursiveStagesMixin,
    PartitionStagesMixin,
    VerificationStagesMixin,
    RetrievalStagesMixin,
    LlmStagesMixin,
    VerificationCoreMixin,
):
    def __init__(
        self,
        llm_client: Any,
        retrieval_context: RetrievalContext,
        verifier: RtlVerifier,
        config: Optional[AgenticIpReuseConfig] = None,
        stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.llm = llm_client
        self.retrieval_context = retrieval_context
        self.verifier = verifier
        self.config = config or AgenticIpReuseConfig()
        self.stage_callback = stage_callback
        self._live_store: Optional[VectorStore] = None

    def run(
        self,
        prompt: str,
        *,
        target_hdl: Optional[str] = None,
        top_module: Optional[str] = None,
        constraints: Optional[Iterable[str]] = None,
        workspace_dir: Optional[str | Path] = None,
    ) -> AgenticIpReuseResult:
        target = target_hdl or self.config.target_hdl
        constraint_list = list(constraints or [])
        llm_traces: List[LlmTrace] = []
        retrieval_traces: List[Dict[str, Any]] = []

        is_large_spec = len(prompt) > self.config.large_spec_threshold_chars
        use_large_spec_path = is_large_spec or self.config.decomposition_mode == "chunking"
        self._stage(
            "agent_start",
            "running",
            top_module=top_module,
            target_hdl=target,
            prompt_chars=len(prompt),
            large_spec=is_large_spec,
            decomposition_mode=self.config.decomposition_mode,
        )
        self._stage(
            "large_spec_precheck",
            "complete",
            prompt_chars=len(prompt),
            threshold_chars=self.config.large_spec_threshold_chars,
            activated=use_large_spec_path,
            large_spec=is_large_spec,
            decomposition_mode=self.config.decomposition_mode,
        )
        if use_large_spec_path:
            return self._run_large_spec(
                prompt,
                target=target,
                top_module=top_module,
                constraints=constraint_list,
                llm_traces=llm_traces,
                retrieval_traces=retrieval_traces,
                workspace_dir=workspace_dir,
            )

        requirements_payload = self._complete_json(
            "requirements",
            build_requirements_prompt(prompt, target, constraint_list),
            llm_traces,
        )
        requirements = _requirements_from_payload(requirements_payload, prompt)
        self._stage("requirements", "complete", functionality=requirements.functionality)

        decomposition_payload = self._complete_json(
            "decomposition",
            build_decomposition_prompt(requirements),
            llm_traces,
        )
        modules = _modules_from_payload(decomposition_payload, requirements)
        self._stage("decomposition", "complete", module_count=len(modules), modules=[module.name for module in modules])

        decisions = self._build_decisions(modules, llm_traces, retrieval_traces)

        plan = IpReusePlan(requirements=requirements, modules=modules, decisions=decisions)
        self._stage("rtl_generation", "running", top_module=top_module)
        final_text = self._complete_text(
            "rtl_generation",
            build_rtl_generation_prompt(plan, target, top_module),
            llm_traces,
        )
        rtl = extract_code(final_text)
        self._stage("rtl_generation", "complete", rtl_chars=len(rtl))
        self._stage("verification", "running", top_module=top_module)
        verification = self._verify_or_empty(rtl, top_module)
        self._stage("verification", "complete", passed=verification.passed)

        repair_attempts = 0
        while not verification.passed and repair_attempts < self.config.max_repair_attempts:
            repair_attempts += 1
            diagnostics = [asdict(item) for item in verification.diagnostics]
            self._stage("repair", "running", attempt=repair_attempts)
            final_text = self._complete_text(
                f"repair_{repair_attempts}",
                build_repair_prompt(plan, rtl, diagnostics, target, top_module),
                llm_traces,
            )
            rtl = extract_code(final_text)
            self._stage("repair", "generated", attempt=repair_attempts, rtl_chars=len(rtl))
            self._stage("verification", "running", top_module=top_module, repair_attempt=repair_attempts)
            verification = self._verify_or_empty(rtl, top_module)
            self._stage("verification", "complete", passed=verification.passed, repair_attempt=repair_attempts)

        result = AgenticIpReuseResult(
            plan=plan,
            rtl=rtl,
            final_text=final_text,
            verification=verification,
            repair_attempts=repair_attempts,
            llm_traces=llm_traces,
            retrieval_traces=retrieval_traces,
        )
        self._stage("agent_complete", "complete", passed=verification.passed, repair_attempts=repair_attempts)
        return result
