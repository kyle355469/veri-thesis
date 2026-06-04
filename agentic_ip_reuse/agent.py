from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from rag_rtl.json_utils import dumps_json, json_default, preview_text
from rag_rtl.llm import extract_code
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import Diagnostic, RetrievalHit, VerificationReport
from rag_rtl.verifier import RtlVerifier

from .prompts import (
    CRITERIA,
    MODULE_CATEGORIES,
    build_candidate_evaluation_prompt,
    build_decomposition_prompt,
    build_repair_prompt,
    build_requirements_prompt,
    build_rtl_generation_prompt,
)
from .types import (
    AgenticIpReuseResult,
    IpCandidate,
    IpReusePlan,
    LlmTrace,
    ModuleReuseDecision,
    ModuleSpec,
    SystemRequirements,
)


ACTION_VALUES = {"reuse", "configure", "adapt", "new"}
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

METADATA_ALIASES = {
    "function_match": ("function_match", "function", "behavior"),
    "interface_compatibility": ("interface_compatibility", "interface", "bus", "protocol"),
    "configurability": ("configurability", "parameters", "parameterized", "configurable"),
    "verification_status": ("verification_status", "verification", "verified", "testbench", "formal"),
    "license": ("license", "licence"),
    "synthesis_support": ("synthesis_support", "synthesis", "synthesizable", "timing"),
    "documentation_quality": ("documentation_quality", "documentation", "docs", "readme"),
}


@dataclass(frozen=True)
class AgenticIpReuseConfig:
    target_hdl: str = "verilog"
    retrieve_k: int = 8
    context_k: int = 4
    max_repair_attempts: int = 2
    temperature: float = 0.2
    max_tokens: int = 32768


class AgenticIpReuseAgent:
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

    def run(
        self,
        prompt: str,
        *,
        target_hdl: Optional[str] = None,
        top_module: Optional[str] = None,
        constraints: Optional[Iterable[str]] = None,
    ) -> AgenticIpReuseResult:
        target = target_hdl or self.config.target_hdl
        constraint_list = list(constraints or [])
        llm_traces: List[LlmTrace] = []
        retrieval_traces: List[Dict[str, Any]] = []

        self._stage("agent_start", "running", top_module=top_module, target_hdl=target)
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

        decisions: List[ModuleReuseDecision] = []
        for module in modules:
            query = module.reuse_query or f"{module.category} {module.name} {module.purpose}"
            self._stage("ip_search", "running", module=module.name, category=module.category, query=query)
            hits = self.retrieval_context.prepare(
                query=query,
                retrieve_k=self.config.retrieve_k,
                context_k=self.config.context_k,
            )
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

    def _complete_json(self, stage: str, prompt: str, traces: List[LlmTrace]) -> Dict[str, Any]:
        response = self._complete_text(stage, prompt, traces)
        parsed = _parse_json_object(response)
        traces[-1].parsed = parsed is not None
        return parsed or {}

    def _complete_text(self, stage: str, prompt: str, traces: List[LlmTrace]) -> str:
        self._stage(f"llm:{stage}", "running")
        response = self.llm.complete(
            prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        traces.append(
            LlmTrace(
                stage=stage,
                prompt_preview=preview_text(prompt),
                response_preview=preview_text(response),
                parsed=False,
            )
        )
        self._stage(f"llm:{stage}", "complete", response_chars=len(response))
        return response

    def _stage(self, stage: str, status: str, **payload: Any) -> None:
        if self.stage_callback is None:
            return
        event = {"stage": stage, "status": status, **payload}
        self.stage_callback(event)

    def _verify_or_empty(self, rtl: str, top_module: Optional[str]) -> VerificationReport:
        if rtl:
            return self.verifier.verify(rtl, top_module=top_module)
        return VerificationReport(
            syntax_passed=False,
            lint_passed=False,
            diagnostics=[
                Diagnostic(
                    tool="rtl_extraction",
                    passed=False,
                    stderr="final model response did not contain parsable RTL",
                )
            ],
        )


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
        criteria={criterion: _metadata_value(metadata, criterion) for criterion in CRITERIA},
    )


def dumps_result(result: AgenticIpReuseResult, *, indent: Optional[int] = 2) -> str:
    return dumps_json(result.to_dict(), indent=indent)


def _metadata_value(metadata: Dict[str, Any], criterion: str) -> str:
    for key in METADATA_ALIASES[criterion]:
        if key in metadata and metadata[key] not in (None, ""):
            value = metadata[key]
            if isinstance(value, (dict, list)):
                return json.dumps(value, default=json_default, ensure_ascii=False)
            return str(value)
    return "unknown"


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in JSON_BLOCK_RE.finditer(text))
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _requirements_from_payload(payload: Dict[str, Any], original_prompt: str) -> SystemRequirements:
    return SystemRequirements(
        functionality=_string_or_unknown(payload.get("functionality")) if payload else original_prompt,
        performance_target=_string_or_unknown(payload.get("performance_target")),
        io_interface=_string_or_unknown(payload.get("io_interface")),
        ppa_constraints=_string_list(payload.get("ppa_constraints")),
        clock_reset=_string_or_unknown(payload.get("clock_reset")),
        assumptions=_string_list(payload.get("assumptions")),
        unknowns=_string_list(payload.get("unknowns")),
    )


def _modules_from_payload(payload: Dict[str, Any], requirements: SystemRequirements) -> List[ModuleSpec]:
    modules_payload = payload.get("modules") if isinstance(payload, dict) else None
    modules = []
    if isinstance(modules_payload, list):
        modules = [_module_from_payload(item) for item in modules_payload if isinstance(item, dict)]
    modules = [module for module in modules if module.name and module.category]
    if modules:
        return modules
    return [
        ModuleSpec(
            category="Processing Core",
            name="processing_core",
            purpose=requirements.functionality,
            required_interface=requirements.io_interface,
            performance_target=requirements.performance_target,
            ppa_constraints=requirements.ppa_constraints,
            reuse_query=f"{requirements.functionality} {requirements.io_interface}",
        )
    ]


def _module_from_payload(payload: Dict[str, Any]) -> ModuleSpec:
    category = _string_or_unknown(payload.get("category"))
    if category not in MODULE_CATEGORIES:
        category = category if category != "unknown" else "Processing Core"
    return ModuleSpec(
        category=category,
        name=_string_or_unknown(payload.get("name")),
        purpose=_string_or_unknown(payload.get("purpose")),
        required_interface=_string_or_unknown(payload.get("required_interface")),
        performance_target=_string_or_unknown(payload.get("performance_target")),
        ppa_constraints=_string_list(payload.get("ppa_constraints")),
        reuse_query=_string_or_unknown(payload.get("reuse_query")),
        omitted_reason=_optional_string(payload.get("omitted_reason")),
    )


def _decision_from_payload(
    module: ModuleSpec,
    candidates: List[IpCandidate],
    payload: Dict[str, Any],
) -> ModuleReuseDecision:
    by_doc_id = {candidate.doc_id: candidate for candidate in candidates}
    for item in payload.get("candidate_evaluations", []):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "")
        candidate = by_doc_id.get(doc_id)
        if candidate is None:
            continue
        criteria = item.get("criteria")
        if isinstance(criteria, dict):
            candidate.criteria = {
                criterion: _string_or_unknown(criteria.get(criterion, candidate.criteria.get(criterion)))
                for criterion in CRITERIA
            }
        candidate.rationale = _string_or_unknown(item.get("rationale"))

    selected = _optional_string(payload.get("selected_doc_id"))
    if selected not in by_doc_id:
        selected = None
    action = _string_or_unknown(payload.get("action"))
    if action not in ACTION_VALUES:
        action = "new" if selected is None else "adapt"

    return ModuleReuseDecision(
        module=module,
        candidates=candidates,
        selected_doc_id=selected,
        action=action,
        parameterization=_dict_or_empty(payload.get("parameterization")),
        integration_notes=_string_or_unknown(payload.get("integration_notes")),
        rationale=_string_or_unknown(payload.get("rationale")),
    )


def _string_or_unknown(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
