from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from rag_rtl.llm import extract_code
from rag_rtl.repair_cache import normalize_diagnostics as _normalize_diagnostics
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

from .config import AgenticIpReuseConfig
from .manifest import (
    condensation_fidelity_report as _condensation_fidelity_report,
    dependency_order as _dependency_order,
    manifest_validation_errors as _manifest_validation_errors,
    prepare_workspace as _prepare_workspace,
    recursive_decomposition_validation_errors as _recursive_decomposition_validation_errors,
    render_condensed_spec as _render_condensed_spec,
    render_manifest_index as _render_manifest_index,
    render_module_spec as _render_module_spec,
    verbatim_interface_excerpts as _verbatim_interface_excerpts,
    write_text as _write_text,
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
        repair_cache: Optional[Any] = None,
    ) -> None:
        self.llm = llm_client
        self.retrieval_context = retrieval_context
        self.verifier = verifier
        self.config = config or AgenticIpReuseConfig()
        self.stage_callback = stage_callback
        # Duck-typed rag_rtl.repair_cache.RepairFixCache: lookup_hint/record_fix.
        self.repair_cache = repair_cache
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
        return self._run_rtl_generation_from_plan(
            plan,
            target=target,
            top_module=top_module,
            llm_traces=llm_traces,
            retrieval_traces=retrieval_traces,
            original_spec=prompt,
        )

    def run_from_plan(
        self,
        plan: IpReusePlan,
        *,
        target_hdl: Optional[str] = None,
        top_module: Optional[str] = None,
        workspace_dir: Optional[str | Path] = None,
        original_spec: Optional[str] = None,
        reuse_modules: Optional[Dict[str, str]] = None,
        environment_notes: Optional[List[str]] = None,
    ) -> AgenticIpReuseResult:
        target = target_hdl or self.config.target_hdl
        llm_traces: List[LlmTrace] = []
        self._stage(
            "agent_start",
            "running",
            top_module=top_module,
            target_hdl=target,
            source="plan",
            module_count=len(plan.modules),
        )
        self._stage(
            "plan_input",
            "complete",
            module_count=len(plan.modules),
            modules=[module.name for module in plan.modules],
        )
        return self._run_rtl_generation_from_plan(
            plan,
            target=target,
            top_module=top_module,
            llm_traces=llm_traces,
            retrieval_traces=[],
            workspace_dir=workspace_dir,
            original_spec=original_spec,
            reuse_modules=reuse_modules,
            environment_notes=environment_notes,
        )

    def condense_spec(
        self,
        prompt: str,
        *,
        target_hdl: Optional[str] = None,
        top_module: Optional[str] = None,
        constraints: Optional[Iterable[str]] = None,
        workspace_dir: Optional[str | Path] = None,
        llm_traces: Optional[List[LlmTrace]] = None,
    ) -> str:
        """Chunk & merge an oversized spec into one bounded structured summary.

        Thin wrapper over condense_spec_views that returns the generation view.
        """
        return self.condense_spec_views(
            prompt,
            target_hdl=target_hdl,
            top_module=top_module,
            constraints=constraints,
            workspace_dir=workspace_dir,
            llm_traces=llm_traces,
        )["generation"]

    def condense_spec_views(
        self,
        prompt: str,
        *,
        target_hdl: Optional[str] = None,
        top_module: Optional[str] = None,
        constraints: Optional[Iterable[str]] = None,
        workspace_dir: Optional[str | Path] = None,
        provided_modules: Optional[Iterable[str]] = None,
        excerpt_max_chars: int = 24000,
        llm_traces: Optional[List[LlmTrace]] = None,
    ) -> Dict[str, str]:
        """Chunk & merge an oversized spec, then render two condensed views.

        Reuses the large-spec partition pipeline (markdown chunking, per-chunk
        extraction, manifest merge). From the single merged manifest two texts are
        rendered: a "planner" view (interfaces + reuse queries + short behavioral
        digest) and a "generation" view (interfaces + full behavioral requirements
        for modules that must be generated; provided_modules are interface-only).
        Both carry verbatim port tables / module headers extracted deterministically
        from the raw spec, and a fidelity report of identifiers that did not survive
        condensation is written to the workspace.
        """
        workspace = _prepare_workspace(workspace_dir)
        traces: List[LlmTrace] = llm_traces if llm_traces is not None else []
        artifacts: Dict[str, str] = {}
        manifest = self._partition_large_spec(
            prompt,
            target=target_hdl or self.config.target_hdl,
            top_module=top_module,
            constraints=list(constraints or []),
            llm_traces=traces,
            workspace=workspace,
            artifacts=artifacts,
        )
        _write_text(
            workspace / "spec_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        excerpts = _verbatim_interface_excerpts(prompt, excerpt_max_chars)
        provided = list(provided_modules or [])
        views = {
            view: _render_condensed_spec(
                manifest,
                view=view,
                provided_modules=provided,
                interface_excerpts=excerpts,
            )
            for view in ("planner", "generation")
        }
        fidelity = _condensation_fidelity_report(prompt, views["generation"])
        _write_text(
            workspace / "condense_fidelity_report.json",
            json.dumps(fidelity, ensure_ascii=False, indent=2) + "\n",
        )
        _write_text(workspace / "condensed_spec.txt", views["generation"])
        _write_text(workspace / "condensed_spec.planner.txt", views["planner"])
        self._stage(
            "spec_condensation",
            "complete",
            original_chars=len(prompt),
            condensed_chars=len(views["generation"]),
            planner_chars=len(views["planner"]),
            excerpt_chars=len(excerpts),
            module_count=len(manifest["modules"]),
            macros_missing=len(fidelity["macros_missing"]),
            identifiers_missing=len(fidelity["identifiers_missing"]),
        )
        return views

    def _run_rtl_generation_from_plan(
        self,
        plan: IpReusePlan,
        *,
        target: str,
        top_module: Optional[str],
        llm_traces: List[LlmTrace],
        retrieval_traces: List[Dict[str, Any]],
        workspace_dir: Optional[str | Path] = None,
        original_spec: Optional[str] = None,
        reuse_modules: Optional[Dict[str, str]] = None,
        environment_notes: Optional[List[str]] = None,
    ) -> AgenticIpReuseResult:
        self._stage("rtl_generation", "running", top_module=top_module)
        final_text = self._complete_text(
            "rtl_generation",
            build_rtl_generation_prompt(
                plan,
                target,
                top_module,
                original_spec=original_spec,
                reuse_modules=reuse_modules,
                environment_notes=environment_notes,
            ),
            llm_traces,
        )
        rtl = extract_code(final_text)
        self._stage("rtl_generation", "complete", rtl_chars=len(rtl))
        self._stage("verification", "running", top_module=top_module)
        verification = self._verify_or_empty(rtl, top_module)
        self._stage("verification", "complete", passed=verification.passed)

        repair_attempts = 0
        repair_cache_events: List[Dict[str, Any]] = []
        while not verification.passed and repair_attempts < self.config.max_repair_attempts:
            repair_attempts += 1
            diagnostics = [asdict(item) for item in verification.diagnostics]
            self._stage("repair", "running", attempt=repair_attempts)
            signature = None
            repair_hints: Optional[List[str]] = None
            if self.repair_cache is not None:
                signature = _normalize_diagnostics(diagnostics)
                hint = self.repair_cache.lookup_hint(signature)
                lookup_event = {
                    "event": "lookup",
                    "attempt": repair_attempts,
                    "decision": hint.decision if hint else "miss",
                    "score": hint.score if hint else None,
                    "error_codes": signature.error_codes if signature else [],
                    "injected": hint is not None,
                }
                repair_cache_events.append(lookup_event)
                self._stage("repair_cache", "lookup", **lookup_event)
                if hint is not None:
                    repair_hints = [hint.text]
            prev_rtl = rtl
            final_text = self._complete_text(
                f"repair_{repair_attempts}",
                build_repair_prompt(
                    plan,
                    rtl,
                    diagnostics,
                    target,
                    top_module,
                    original_spec=original_spec,
                    reuse_modules=reuse_modules,
                    environment_notes=environment_notes,
                    repair_hints=repair_hints,
                ),
                llm_traces,
            )
            rtl = extract_code(final_text)
            self._stage("repair", "generated", attempt=repair_attempts, rtl_chars=len(rtl))
            self._stage("verification", "running", top_module=top_module, repair_attempt=repair_attempts)
            verification = self._verify_or_empty(rtl, top_module)
            self._stage("verification", "complete", passed=verification.passed, repair_attempt=repair_attempts)
            # Only verified fixes enter the cache, so a bad repair can never poison it.
            if self.repair_cache is not None and signature is not None and verification.passed:
                self.repair_cache.record_fix(
                    signature,
                    prev_rtl,
                    rtl,
                    task_id=top_module or target,
                    attempt=repair_attempts,
                )
                record_event = {
                    "event": "record",
                    "attempt": repair_attempts,
                    "error_codes": signature.error_codes,
                }
                repair_cache_events.append(record_event)
                self._stage("repair_cache", "record", **record_event)

        result = AgenticIpReuseResult(
            plan=plan,
            rtl=rtl,
            final_text=final_text,
            verification=verification,
            repair_attempts=repair_attempts,
            llm_traces=llm_traces,
            retrieval_traces=retrieval_traces,
            repair_cache_events=repair_cache_events,
        )
        if workspace_dir is not None:
            result.workspace_dir = str(Path(workspace_dir).resolve())
        self._stage("agent_complete", "complete", passed=verification.passed, repair_attempts=repair_attempts)
        return result
