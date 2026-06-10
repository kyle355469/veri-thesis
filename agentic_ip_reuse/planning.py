from __future__ import annotations

import json
from typing import Any, Dict, List

from .constants import ACTION_VALUES, CRITERIA, MODULE_CATEGORIES
from .serialization import dict_or_empty, optional_string, string_list, string_or_unknown
from .types import IpCandidate, ModuleReuseDecision, ModuleSpec, SystemRequirements


def requirements_from_payload(payload: Dict[str, Any], original_prompt: str) -> SystemRequirements:
    return SystemRequirements(
        functionality=string_or_unknown(payload.get("functionality")) if payload else original_prompt,
        performance_target=string_or_unknown(payload.get("performance_target")),
        io_interface=string_or_unknown(payload.get("io_interface")),
        ppa_constraints=string_list(payload.get("ppa_constraints")),
        clock_reset=string_or_unknown(payload.get("clock_reset")),
        assumptions=string_list(payload.get("assumptions")),
        unknowns=string_list(payload.get("unknowns")),
    )


def modules_from_payload(payload: Dict[str, Any], requirements: SystemRequirements) -> List[ModuleSpec]:
    modules_payload = payload.get("modules") if isinstance(payload, dict) else None
    modules = []
    if isinstance(modules_payload, list):
        modules = [module_from_payload(item) for item in modules_payload if isinstance(item, dict)]
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


def module_from_payload(payload: Dict[str, Any]) -> ModuleSpec:
    category = string_or_unknown(payload.get("category"))
    if category not in MODULE_CATEGORIES:
        category = category if category != "unknown" else "Processing Core"
    return ModuleSpec(
        category=category,
        name=string_or_unknown(payload.get("name")),
        purpose=string_or_unknown(payload.get("purpose")),
        required_interface=string_or_unknown(payload.get("required_interface")),
        performance_target=string_or_unknown(payload.get("performance_target")),
        ppa_constraints=string_list(payload.get("ppa_constraints")),
        reuse_query=string_or_unknown(payload.get("reuse_query")),
        omitted_reason=optional_string(payload.get("omitted_reason")),
    )


def decision_from_payload(
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
                criterion: string_or_unknown(criteria.get(criterion, candidate.criteria.get(criterion)))
                for criterion in CRITERIA
            }
        candidate.rationale = string_or_unknown(item.get("rationale"))

    selected = optional_string(payload.get("selected_doc_id"))
    if selected not in by_doc_id:
        selected = None
    action = string_or_unknown(payload.get("action"))
    if action not in ACTION_VALUES:
        action = "new" if selected is None else "adapt"

    return ModuleReuseDecision(
        module=module,
        candidates=candidates,
        selected_doc_id=selected,
        action=action,
        parameterization=dict_or_empty(payload.get("parameterization")),
        integration_notes=string_or_unknown(payload.get("integration_notes")),
        rationale=string_or_unknown(payload.get("rationale")),
    )


def requirements_from_manifest(manifest: Dict[str, Any]) -> SystemRequirements:
    summary = str(manifest.get("system_summary") or "unknown")
    clocks = string_list(manifest.get("clocks"))
    resets = string_list(manifest.get("resets"))
    top_ports = manifest["top_module"]["ports"]
    return SystemRequirements(
        functionality=summary,
        performance_target="; ".join(string_list(manifest.get("shared_constraints"))) or "unknown",
        io_interface=json.dumps(top_ports, ensure_ascii=False),
        ppa_constraints=string_list(manifest.get("shared_constraints")),
        clock_reset="; ".join([*clocks, *resets]) or "unknown",
        assumptions=string_list(manifest.get("assumptions")),
        unknowns=string_list(manifest.get("unknowns")),
    )


def module_from_manifest(payload: Dict[str, Any]) -> ModuleSpec:
    return ModuleSpec(
        category=str(payload.get("category") or "Processing Core"),
        name=str(payload["name"]),
        purpose=str(payload.get("purpose") or "unknown"),
        required_interface=json.dumps(payload.get("ports") or [], ensure_ascii=False),
        performance_target="unknown",
        ppa_constraints=[],
        reuse_query=str(payload.get("reuse_query") or payload["name"]),
    )
