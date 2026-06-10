from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List

from rag_rtl.json_utils import preview_text

from .constants import MODULE_DECL_RE
from .types import ModuleReuseDecision


def validate_single_module_rtl(rtl: str, expected_name: str) -> None:
    names = MODULE_DECL_RE.findall(rtl)
    if names != [expected_name]:
        raise RuntimeError(
            f"expected exactly one generated module named {expected_name}, found {names or 'none'}"
        )


def combine_dependency_rtl(
    generated_modules: Dict[str, str],
    module_payloads: Dict[str, Dict[str, Any]],
    dependencies: List[str],
    module_rtl: str,
) -> str:
    needed: set[str] = set()

    def collect(name: str) -> None:
        if name in needed:
            return
        needed.add(name)
        for dependency in module_payloads[name]["dependencies"]:
            collect(dependency)

    for dependency in dependencies:
        collect(dependency)
    dependency_rtl = [
        rtl
        for name, rtl in generated_modules.items()
        if name in needed
    ]
    return "\n\n".join([*dependency_rtl, module_rtl]).strip()


def combine_recursive_wrapper_rtl(
    generated_modules: Dict[str, str],
    module_payloads: Dict[str, Dict[str, Any]],
    external_dependencies: List[str],
    subtree_order: List[str],
    wrapper_rtl: str,
) -> str:
    external_with_marker = combine_dependency_rtl(
        generated_modules,
        module_payloads,
        external_dependencies,
        "",
    )
    parts = [external_with_marker] if external_with_marker else []
    seen = set(external_dependencies)
    for name in subtree_order:
        if name not in seen:
            parts.append(generated_modules[name])
            seen.add(name)
    parts.append(wrapper_rtl)
    return "\n\n".join(parts).strip()


def combine_final_rtl(
    generated_modules: Dict[str, str],
    generation_order: List[str],
    wrapper_rtl: str,
) -> str:
    return "\n\n".join([*(generated_modules[name] for name in generation_order), wrapper_rtl]).strip()


def decision_generation_payload(decision: ModuleReuseDecision) -> Dict[str, Any]:
    selected = next(
        (candidate for candidate in decision.candidates if candidate.doc_id == decision.selected_doc_id),
        None,
    )
    return {
        "action": decision.action,
        "selected_doc_id": decision.selected_doc_id,
        "parameterization": decision.parameterization,
        "integration_notes": decision.integration_notes,
        "rationale": decision.rationale,
        "selected_candidate": asdict(selected) if selected else None,
    }


def module_signature(rtl: str) -> str:
    match = re.search(r"(?ms)^\s*module\s+[A-Za-z_][A-Za-z0-9_$]*\b.*?;", rtl)
    return match.group(0).strip() if match else preview_text(rtl, 1000)
