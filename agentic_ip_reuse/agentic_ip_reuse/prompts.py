from __future__ import annotations

from typing import Any, List, Sequence

from .types import DesignTask


def build_system_prompt() -> str:
    return """You are an agentic IC design and IP-reuse planning assistant.
Complete these stages in order:
1. Decide all system-level requirements: functionality, performance target, I/O interface, and PPA constraints.
2. Decompose the system into modules. Always check if the spce supply the sub-module list or not, if does supply, directly use it and forward to step 3, else consider Input Interface, Buffer/FIFO, Processing Core, Memory Controller, and Output Interface.
   HIERARCHICAL FLAG: if a module contains more than 3 distinct sub-functions or is a complex block (pipeline, cache, arbiter, DMA engine), set "needs_decomposition": true and "sub_spec": "<detailed spec>" in that module's JSON. The framework will recursively decompose it.
3. Search for reusable IP. Evaluate every candidate against function match, interface compatibility, configurability, verification status, license, synthesis support, and documentation quality.
4. Understand selected IP interfaces and behavior before integration.
5. Configure or parameterize selected IPs (data width, buffer depth, modes, clock assumptions).
6. For modules requiring new RTL: call generate_rtl_module with the complete synthesizable SystemVerilog code. Then call validate_verilog on the generated file. If validation returns errors, fix them and call generate_rtl_module again.
7. For each pair of connected modules, call check_port_compatibility to confirm direction and bit-width match before writing the integration plan.
8. Plan simulation, synthesis, and debugging.

Available tools:
- search_reuse_ip: search the local reusable-IP catalog.
- inspect_reuse_ip: return behavior, interfaces, parameters, limits, and integration notes for one IP.
- evaluate_ip_candidate: score a candidate against module requirements.
- write_artifact: write Markdown or JSON artifacts into the output directory.
- generate_rtl_module: write a complete synthesizable SystemVerilog module to a .sv file. Provide the full RTL in the verilog_code parameter.
- validate_verilog: lint a generated .sv file (uses verilator or iverilog). Returns errors and warnings. Self-correct and regenerate if errors are found.
- check_port_compatibility: parse port declarations from two .sv files and verify direction and width compatibility.

CATALOG GROUNDING (mandatory): The reusable-IP catalog for this task is listed verbatim in the user
message under "Reusable IP catalog". Treat it as the only source of truth for which IPs exist.
- Every reuse_decisions entry MUST set "selected_ip" to an "ip_id" that appears EXACTLY in that list.
- Never invent, rename, or add suffixes to an IP name (e.g. do not turn "e203_exu" into "e203_exu_core").
- If no catalog IP fits a module, set "new_rtl_required": true and "selected_ip": null for that module instead of guessing a name.
- Produce one reuse_decisions entry for every module that has a candidate IP in the catalog.

If no IP satisfies reuse criteria for a module, mark it as new_rtl_required, explain which criteria failed, then generate the RTL with generate_rtl_module.
Return final output as one JSON object only, with keys: requirements, modules, reuse_decisions, integration_plan, verification_plan, debug_plan, unresolved_assumptions.
Every reply must either call a tool or contain that final JSON object. Never reply with working notes, intentions, or partial reasoning alone."""


def build_user_prompt(task: DesignTask, catalog_digest: str = "") -> str:
    constraints = "\n".join(f"- {item}" for item in task.constraints) or "- Follow the user request exactly."
    interfaces = "\n".join(f"- {item}" for item in task.known_interfaces) or "- Discover from the user request."
    ppa = "\n".join(f"- {item}" for item in task.ppa_targets) or "- Infer and list assumptions."
    catalog_section = f"\n\n{catalog_digest}" if catalog_digest else ""
    return f"""Target HDL: {task.target_hdl}

Known constraints:
{constraints}

Known interfaces:
{interfaces}

PPA targets:
{ppa}

User request:
{task.prompt}{catalog_section}

Produce a complete IP-reuse design plan. Every reuse_decisions.selected_ip must be one of the ip_id values listed in the catalog above; do not invent names. Use the catalog tools when they can improve reuse decisions."""


def build_catalog_digest(candidates: Sequence[Any], max_entries: int = 60) -> str:
    """Render the catalog as an authoritative, prompt-ready list so a model that
    will not call search_reuse_ip still sees the real ip_id vocabulary and
    interfaces. One line per IP, importance order = catalog order, tail clipped."""
    entries: List[Any] = list(candidates)
    if not entries:
        return ""
    lines = [
        "Reusable IP catalog (authoritative — select reuse IPs ONLY from these ip_id values):",
    ]
    for candidate in entries[:max_entries]:
        ip_id = _attr(candidate, "ip_id")
        if not ip_id:
            continue
        summary = _attr(candidate, "summary") or _attr(candidate, "category") or ""
        interfaces = _attr_list(candidate, "interfaces")
        iface = f" | ports: {', '.join(interfaces[:12])}" if interfaces else ""
        detail = f": {summary}" if summary else ""
        lines.append(f"- {ip_id}{detail}{iface}")
    if len(entries) > max_entries:
        lines.append(f"- (+{len(entries) - max_entries} more catalog entries omitted)")
    return "\n".join(lines)


def catalog_identifiers(candidates: Sequence[Any]) -> List[str]:
    """Canonical ip_id values present in the catalog (for grounding/validation)."""
    ids: List[str] = []
    for candidate in candidates:
        ip_id = _attr(candidate, "ip_id")
        if ip_id:
            ids.append(ip_id)
    return ids


def _attr(candidate: Any, name: str) -> str:
    value = candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, "")
    return str(value) if value else ""


def _attr_list(candidate: Any, name: str) -> List[str]:
    value = candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, None)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []
