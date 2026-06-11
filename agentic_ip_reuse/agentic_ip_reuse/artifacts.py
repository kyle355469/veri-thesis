from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .json_utils import json_default


ARTIFACT_NAMES = {
    "requirements": "requirements.md",
    "module_decomposition": "module_decomposition.md",
    "ip_reuse_matrix": "ip_reuse_matrix.md",
    "integration_plan": "integration_plan.md",
    "verification_plan": "verification_plan.md",
    "result": "result.json",
}


def write_standard_artifacts(plan: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "requirements": _requirements_md(plan),
        "module_decomposition": _modules_md(plan),
        "ip_reuse_matrix": _reuse_md(plan),
        "integration_plan": _list_md("Integration Plan", plan.get("integration_plan", []), plan.get("unresolved_assumptions", [])),
        "verification_plan": _verification_md(plan),
        "result": json.dumps(plan, default=json_default, indent=2, ensure_ascii=False) + "\n",
    }
    paths: Dict[str, str] = {}
    for key, content in artifacts.items():
        path = root / ARTIFACT_NAMES[key]
        path.write_text(content, encoding="utf-8")
        paths[key] = str(path)
    return paths


def _requirements_md(plan: Dict[str, Any]) -> str:
    req = _dict(plan.get("requirements"))
    lines = ["# System Requirements", ""]
    for title, key in [
        ("Functionality", "functionality"),
        ("Performance", "performance"),
        ("I/O Interfaces", "io_interfaces"),
        ("Protocols", "protocols"),
        ("PPA Constraints", "ppa_constraints"),
        ("Clock / Reset", "clock_reset"),
        ("Assumptions", "assumptions"),
    ]:
        lines.extend([f"## {title}", *_bullet_list(req.get(key, [])), ""])
    return "\n".join(lines).rstrip() + "\n"


def _modules_md(plan: Dict[str, Any]) -> str:
    lines = ["# Module Decomposition", ""]
    for module in _list(plan.get("modules")):
        item = _dict(module)
        lines.append(f"## {item.get('name', 'Unnamed Module')}")
        lines.append(f"- Role: {item.get('role', '')}")
        lines.append(f"- Interfaces: {', '.join(_string_list(item.get('interfaces', []))) or 'TBD'}")
        lines.append(f"- Reuse preference: {item.get('reuse_preference', 'TBD')}")
        needs = ", ".join(_string_list(item.get("verification_needs", []))) or "TBD"
        lines.append(f"- Verification needs: {needs}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _reuse_md(plan: Dict[str, Any]) -> str:
    lines = [
        "# IP Reuse Matrix",
        "",
        "| Module | Selected IP | New RTL Required | Required Adapters | Risk Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for decision in _list(plan.get("reuse_decisions")):
        item = _dict(decision)
        lines.append(
            "| {module} | {selected} | {new_rtl} | {adapters} | {risks} |".format(
                module=item.get("module_name", ""),
                selected=item.get("selected_ip") or "None",
                new_rtl=str(bool(item.get("new_rtl_required", False))),
                adapters=", ".join(_string_list(item.get("required_adapters", []))) or "None",
                risks=", ".join(_string_list(item.get("risk_notes", []))) or "None",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _verification_md(plan: Dict[str, Any]) -> str:
    lines = ["# Verification, Synthesis, and Debug Plan", ""]
    lines.extend(["## Verification", *_bullet_list(plan.get("verification_plan", [])), ""])
    lines.extend(["## Debug", *_bullet_list(plan.get("debug_plan", [])), ""])
    return "\n".join(lines).rstrip() + "\n"


def _list_md(title: str, items: Any, assumptions: Any) -> str:
    lines = [f"# {title}", "", *_bullet_list(items), ""]
    if assumptions:
        lines.extend(["## Unresolved Assumptions", *_bullet_list(assumptions), ""])
    return "\n".join(lines).rstrip() + "\n"


def _bullet_list(items: Any) -> list[str]:
    values = _string_list(items)
    return [f"- {item}" for item in values] if values else ["- TBD"]


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
