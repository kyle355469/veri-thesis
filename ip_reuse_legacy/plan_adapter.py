from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .serialization import dict_or_empty, optional_string, string_list, string_or_unknown
from .types import IpCandidate, IpReusePlan, ModuleReuseDecision, ModuleSpec, SystemRequirements


def load_agentic_plan(path: str | Path) -> IpReusePlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("plan file must contain a JSON object")
    return agentic_plan_from_payload(payload)


def agentic_plan_from_payload(payload: Dict[str, Any]) -> IpReusePlan:
    plan = _unwrap_plan(payload)
    requirements = _requirements_from_agentic_plan(plan)
    modules = _modules_from_agentic_plan(plan, requirements)
    decisions = _decisions_from_agentic_plan(plan, modules)
    return IpReusePlan(requirements=requirements, modules=modules, decisions=decisions)


def _unwrap_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    structured = payload.get("structured_plan")
    if isinstance(structured, dict):
        return structured
    return payload


def _requirements_from_agentic_plan(plan: Dict[str, Any]) -> SystemRequirements:
    raw = dict_or_empty(plan.get("requirements"))
    unresolved = string_list(plan.get("unresolved_assumptions"))
    assumptions = [*string_list(raw.get("assumptions")), *unresolved]
    return SystemRequirements(
        functionality=_compact(raw.get("functionality")) or _compact(plan.get("system_summary")) or "unknown",
        performance_target=_fuzzy_value(raw, _PERFORMANCE_KEYS) or "unknown",
        io_interface=_fuzzy_value(raw, _INTERFACE_KEYS) or "unknown",
        ppa_constraints=string_list(raw.get("ppa_constraints")) or string_list(_fuzzy_raw_value(raw, _PPA_KEYS)),
        clock_reset=_fuzzy_value(raw, _CLOCK_RESET_KEYS) or "unknown",
        assumptions=assumptions,
        unknowns=unresolved,
    )


# Planner output is free-form JSON, so the same fact arrives under many spellings
# ("io_interface", "I/O interface", "ports", ...). Keys are normalized to lowercase
# alphanumerics before matching.
_INTERFACE_KEYS = (
    "iointerface",
    "iointerfaces",
    "interface",
    "interfaces",
    "io",
    "ios",
    "ports",
    "portlist",
    "signals",
)
_PERFORMANCE_KEYS = ("performance", "performancetarget", "timing")
_PPA_KEYS = ("ppaconstraints", "ppa", "ppatargets", "ppaconstraint")
_CLOCK_RESET_KEYS = ("clockreset", "clock", "reset", "clocksandresets")


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _fuzzy_raw_value(raw: Dict[str, Any], normalized_keys: tuple[str, ...]) -> Any:
    for target in normalized_keys:
        for key, value in raw.items():
            if _normalize_key(key) == target:
                return value
    return None


def _fuzzy_value(raw: Dict[str, Any], normalized_keys: tuple[str, ...]) -> str:
    return _compact(_fuzzy_raw_value(raw, normalized_keys))


def _modules_from_agentic_plan(plan: Dict[str, Any], requirements: SystemRequirements) -> List[ModuleSpec]:
    modules = []
    for item in _list_of_dicts(plan.get("modules")):
        name = string_or_unknown(item.get("name"))
        purpose = _compact(item.get("role")) or _compact(item.get("sub_spec")) or _compact(item.get("purpose"))
        interface = (
            _fuzzy_value(item, _INTERFACE_KEYS)
            or _compact(item.get("required_interface"))
            or _compact(dict_or_empty(item.get("sub_spec")).get("interface"))
        )
        if not interface and _is_top_module(item) and requirements.io_interface != "unknown":
            interface = requirements.io_interface
        modules.append(
            ModuleSpec(
                category=_category(item),
                name=name,
                purpose=purpose or requirements.functionality,
                required_interface=interface or "unknown",
                performance_target=_compact(item.get("performance_target")) or requirements.performance_target,
                ppa_constraints=string_list(item.get("ppa_constraints")) or requirements.ppa_constraints,
                reuse_query=_reuse_query(item, name, purpose),
                omitted_reason=optional_string(item.get("omitted_reason")),
            )
        )
    if modules:
        if len(modules) == 1 and modules[0].required_interface == "unknown" and requirements.io_interface != "unknown":
            modules[0].required_interface = requirements.io_interface
        return modules
    return [
        ModuleSpec(
            category="Processing Core",
            name="top",
            purpose=requirements.functionality,
            required_interface=requirements.io_interface,
            performance_target=requirements.performance_target,
            ppa_constraints=requirements.ppa_constraints,
            reuse_query=requirements.functionality,
        )
    ]


def _is_top_module(item: Dict[str, Any]) -> bool:
    category = (optional_string(item.get("category")) or optional_string(item.get("type")) or "").strip().lower()
    return category == "top"


def _decisions_from_agentic_plan(plan: Dict[str, Any], modules: List[ModuleSpec]) -> List[ModuleReuseDecision]:
    raw_by_module: Dict[str, Dict[str, Any]] = {}
    for item in _list_of_dicts(plan.get("reuse_decisions")):
        module_name = (
            optional_string(item.get("module_name"))
            or optional_string(item.get("module"))
            or optional_string(item.get("name"))
        )
        if module_name:
            raw_by_module[module_name] = item
    for item in _list_of_dicts(plan.get("modules")):
        module_name = optional_string(item.get("name"))
        if not module_name or module_name in raw_by_module:
            continue
        reuse = optional_string(item.get("reuse")) or optional_string(dict_or_empty(item.get("sub_spec")).get("reuse"))
        if reuse:
            raw_by_module[module_name] = {"selected_ip": reuse, "notes": _compact(item.get("notes"))}

    decisions: List[ModuleReuseDecision] = []
    for module in modules:
        raw = raw_by_module.get(module.name, {})
        selected = _selected_ip(raw)
        action = _action(raw, selected)
        notes = _integration_notes(raw)
        candidates = [_candidate_from_decision(selected, module, raw, notes)] if selected else []
        decisions.append(
            ModuleReuseDecision(
                module=module,
                candidates=[candidate for candidate in candidates if candidate is not None],
                selected_doc_id=selected,
                action=action,
                parameterization=dict_or_empty(raw.get("parameterization")),
                integration_notes=notes or "No prior reuse decision; generate or infer implementation as needed.",
                rationale=_compact(raw.get("rationale")) or _compact(raw.get("notes")) or "Imported from agentic plan.",
            )
        )
    return decisions


def _candidate_from_decision(
    selected: Optional[str],
    module: ModuleSpec,
    raw: Dict[str, Any],
    notes: str,
) -> Optional[IpCandidate]:
    if not selected:
        return None
    criteria = dict_or_empty(raw.get("criteria"))
    return IpCandidate(
        doc_id=selected,
        score=1.0,
        rerank_score=None,
        tags=string_list(raw.get("tags")),
        problem=module.purpose,
        solution=notes or _compact(raw.get("notes")) or f"Reusable IP selected for {module.name}.",
        metadata={"source": "agentic_ip_reuse_plan"},
        criteria={key: string_or_unknown(value) for key, value in criteria.items()},
        rationale=_compact(raw.get("rationale")) or _compact(raw.get("notes")),
    )


def _selected_ip(raw: Dict[str, Any]) -> Optional[str]:
    return (
        optional_string(raw.get("selected_ip"))
        or optional_string(raw.get("reuse_ip"))
        or optional_string(raw.get("selected_doc_id"))
        or optional_string(raw.get("ip_id"))
        or optional_string(raw.get("ip"))
        or optional_string(raw.get("reuse"))
    )


def _action(raw: Dict[str, Any], selected: Optional[str]) -> str:
    if bool(raw.get("new_rtl_required")) or not selected:
        return "new"
    if string_list(raw.get("required_adapters")):
        return "adapt"
    if dict_or_empty(raw.get("parameterization")):
        return "configure"
    return "reuse"


def _integration_notes(raw: Dict[str, Any]) -> str:
    parts = [
        _compact(raw.get("notes")),
        _compact(raw.get("integration_notes")),
        _labelled_list("Adapters", raw.get("required_adapters")),
        _labelled_list("Risks", raw.get("risk_notes")),
    ]
    return "; ".join(part for part in parts if part)


def _category(item: Dict[str, Any]) -> str:
    category = optional_string(item.get("category")) or optional_string(item.get("type"))
    if category:
        return category
    name = string_or_unknown(item.get("name")).lower()
    reuse = _compact(item.get("reuse_preference")).lower()
    text = f"{name} {reuse}"
    if "fifo" in text or "buffer" in text:
        return "Buffer / FIFO"
    if "interface" in text:
        return "Interface"
    if "memory" in text:
        return "Memory Controller"
    return "Processing Core"


def _reuse_query(item: Dict[str, Any], name: str, purpose: str) -> str:
    parts = [
        name,
        purpose,
        _compact(item.get("reuse_preference")),
        _compact(item.get("interfaces")),
    ]
    return " ".join(part for part in parts if part).strip() or name


def _labelled_list(label: str, value: Any) -> str:
    items = string_list(value)
    return f"{label}: {', '.join(items)}" if items else ""


def _compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
