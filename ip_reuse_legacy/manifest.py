from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import HDL_NAME_RE
from .serialization import string_list


def prepare_workspace(workspace_dir: Optional[str | Path]) -> Path:
    if workspace_dir is None:
        root = Path.cwd() / "runs" / "agentic_ip_reuse_workspaces"
        root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="large_spec_", dir=root))
    else:
        workspace = Path(workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    for child in ("specs", "rtl", "combined", "errors"):
        (workspace / child).mkdir(parents=True, exist_ok=True)
    return workspace


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def leaf_decomposition(module_payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "decision": "leaf",
        "reason": reason,
        "parent_module": {
            "name": module_payload["name"],
            "ports": module_payload["ports"],
            "instances": [],
        },
        "children": [],
    }


def recursive_decomposition_validation_errors(
    decomposition: Dict[str, Any],
    parent_payload: Dict[str, Any],
    *,
    existing_names: set[str],
    ancestors: List[str],
) -> List[str]:
    errors: List[str] = []
    decision = str(decomposition.get("decision") or "")
    if decision not in {"leaf", "decompose"}:
        errors.append("decision must be leaf or decompose")
    if not str(decomposition.get("reason") or "").strip():
        errors.append("reason must not be empty")
    parent = decomposition.get("parent_module")
    if not isinstance(parent, dict):
        return [*errors, "parent_module must be an object"]
    if parent.get("name") != parent_payload["name"]:
        errors.append(f"parent module name must remain exactly {parent_payload['name']}")
    if port_contract(parent.get("ports")) != port_contract(parent_payload.get("ports")):
        errors.append("parent module public port contract must remain unchanged")

    children = decomposition.get("children")
    instances = parent.get("instances")
    if decision == "leaf":
        if children != []:
            errors.append("leaf decomposition must have an empty children list")
        if instances != []:
            errors.append("leaf decomposition must have an empty parent_module.instances list")
        return errors

    if not isinstance(children, list) or not children:
        errors.append("decompose decision requires a nonempty children list")
        return errors
    child_names = [
        str(child.get("name") or "")
        for child in children
        if isinstance(child, dict)
    ]
    duplicates = sorted({name for name in child_names if name and child_names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate immediate child names: {', '.join(duplicates)}")
    conflicts = sorted({name for name in child_names if name in existing_names or name in ancestors})
    if conflicts:
        errors.append(f"immediate child names conflict with existing hierarchy names: {', '.join(conflicts)}")
    if parent_payload["name"] in child_names:
        errors.append("recursive decomposition made no progress because a child repeats the parent name")

    synthetic = recursive_node_manifest(
        {
            "system_summary": parent_payload.get("purpose", "unknown"),
            "clocks": [],
            "resets": [],
            "parameters": [],
            "shared_constraints": [],
            "assumptions": [],
            "unknowns": [],
        },
        parent_payload,
        parent,
        children,
    )
    errors.extend(manifest_validation_errors(synthetic, parent_payload["name"]))
    return list(dict.fromkeys(errors))


def recursive_node_manifest(
    root_manifest: Dict[str, Any],
    parent_payload: Dict[str, Any],
    parent_module: Dict[str, Any],
    children: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "system_summary": parent_payload.get("purpose") or root_manifest.get("system_summary") or "unknown",
        "clocks": root_manifest.get("clocks", []),
        "resets": root_manifest.get("resets", []),
        "parameters": root_manifest.get("parameters", []),
        "shared_constraints": root_manifest.get("shared_constraints", []),
        "assumptions": root_manifest.get("assumptions", []),
        "unknowns": root_manifest.get("unknowns", []),
        "top_module": parent_module,
        "modules": children,
    }


def port_contract(ports: Any) -> List[tuple[str, str, str]]:
    if not isinstance(ports, list):
        return []
    return sorted(
        (
            str(port.get("name") or ""),
            str(port.get("direction") or "").lower(),
            str(port.get("width") or ""),
        )
        for port in ports
        if isinstance(port, dict)
    )


def sanitize_manifest(manifest: Any) -> Any:
    """Deterministically repair common model deviations before validation:
    glob/wildcard port names (e.g. ``*_icb_cmd_valid``) become behavioral notes,
    empty behavioral_requirements/width/description fields get filled, duplicate
    ports and modules are dropped, and unknown or self dependencies are removed.
    Structural problems (missing fields, missing instances) are left for the
    LLM correction round."""
    if not isinstance(manifest, dict):
        return manifest
    modules = [module for module in manifest.get("modules") or [] if isinstance(module, dict)]
    seen_module_names: set[str] = set()
    deduped_modules: List[Dict[str, Any]] = []
    for module in modules:
        name = str(module.get("name") or "")
        if name and name in seen_module_names:
            continue
        seen_module_names.add(name)
        deduped_modules.append(module)
    if isinstance(manifest.get("modules"), list):
        manifest["modules"] = deduped_modules

    top = manifest.get("top_module")
    owners = ([top] if isinstance(top, dict) else []) + deduped_modules
    for owner in owners:
        notes = _sanitize_ports(owner)
        is_submodule = owner is not top
        if is_submodule:
            requirements = owner.get("behavioral_requirements")
            if not isinstance(requirements, list):
                requirements = []
            requirements.extend(notes)
            if not requirements:
                fallback = str(owner.get("purpose") or "").strip()
                requirements = [fallback or "See the system summary for behavioral requirements."]
            owner["behavioral_requirements"] = requirements
            if not owner.get("ports"):
                # The spec text never compiled this list; clock/reset stubs keep the
                # manifest renderable instead of failing the whole condensation.
                stub_names = [
                    str(item) for item in (manifest.get("clocks") or []) + (manifest.get("resets") or [])
                    if HDL_NAME_RE.fullmatch(str(item))
                ] or ["clk"]
                owner["ports"] = [
                    {
                        "name": stub,
                        "direction": "input",
                        "width": "1",
                        "description": f"{stub} (full interface unspecified in source spec; derive from behavioral requirements)",
                    }
                    for stub in dict.fromkeys(stub_names)
                ]
                requirements.append(
                    "The source specification did not enumerate this module's ports; derive the full "
                    "interface from its purpose and its dependents."
                )
            name = str(owner.get("name") or "")
            if not str(owner.get("category") or "").strip():
                owner["category"] = "submodule"
            if not str(owner.get("purpose") or "").strip():
                owner["purpose"] = str(requirements[0])
            if not str(owner.get("reuse_query") or "").strip():
                owner["reuse_query"] = f"search for {name or 'submodule'} implementation"
            dependencies = owner.get("dependencies")
            if isinstance(dependencies, list):
                normalized_dependencies: List[str] = []
                for dependency in dependencies:
                    if isinstance(dependency, dict):
                        dependency = dependency.get("name") or dependency.get("module") or ""
                    dependency = str(dependency).strip()
                    if dependency and dependency in seen_module_names and dependency != name:
                        normalized_dependencies.append(dependency)
                owner["dependencies"] = normalized_dependencies
    if isinstance(top, dict):
        _sanitize_top_instances(top, deduped_modules)
    return manifest


def _sanitize_top_instances(top: Dict[str, Any], modules: List[Dict[str, Any]]) -> None:
    """Normalize instance connections against the (sanitized) module port lists:
    drop connections to ports the module does not declare, and connect missing
    ports to same-named nets. Connections are prompt text, not compiled RTL, so
    same-name wiring is the faithful default."""
    instances = top.get("instances")
    if not isinstance(instances, list):
        return
    ports_by_module = {
        str(module.get("name") or ""): {
            str(port.get("name") or "")
            for port in module.get("ports") or []
            if isinstance(port, dict)
        }
        for module in modules
    }
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        expected = ports_by_module.get(str(instance.get("module") or ""))
        connections = instance.get("connections")
        if expected is None or not isinstance(connections, dict):
            continue
        cleaned = {
            port: (str(signal).strip() or port)
            for port, signal in connections.items()
            if port in expected
        }
        for port in sorted(expected - set(cleaned)):
            cleaned[port] = port
        instance["connections"] = cleaned


def _sanitize_ports(owner: Dict[str, Any]) -> List[str]:
    """Repair the ports list in place; return note strings for ports that could
    not be kept (invalid names such as bus globs)."""
    ports = owner.get("ports")
    if not isinstance(ports, list):
        return []
    notes: List[str] = []
    kept: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for port in ports:
        if not isinstance(port, dict):
            continue
        name = str(port.get("name") or "")
        direction = str(port.get("direction") or "").lower()
        width = str(port.get("width") or "").strip()
        description = str(port.get("description") or "").strip()
        if not HDL_NAME_RE.fullmatch(name):
            # Glob/wildcard names (e.g. ``*_icb_cmd_valid``) usually stand for a
            # bus-signal family; keep the de-globbed identifier when one exists.
            deglobbed = re.sub(r"[^A-Za-z0-9_]+", "", name).strip("_")
            if HDL_NAME_RE.fullmatch(deglobbed) and deglobbed not in seen_names:
                description = (
                    f"{description + ' ' if description else ''}"
                    f"(one of a signal family written as {name} in the source spec)"
                ).strip()
                name = deglobbed
            else:
                label = name or "<unnamed>"
                notes.append(
                    f"Port group {label} ({direction or 'unknown direction'}, width {width or 'unspecified'}): "
                    f"{description or 'expand this signal family into explicit ports.'}"
                )
                continue
        if name in seen_names:
            continue
        seen_names.add(name)
        if direction not in {"input", "output", "inout"}:
            description = f"{description + ' ' if description else ''}(direction was unspecified; assumed input)".strip()
            direction = "input"
        port["name"] = name
        port["direction"] = direction
        port["width"] = width or "1"
        port["description"] = description or name
        kept.append(port)
    owner["ports"] = kept
    return notes


def manifest_validation_errors(manifest: Dict[str, Any], expected_top: Optional[str]) -> List[str]:
    errors: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    for key in (
        "system_summary",
        "clocks",
        "resets",
        "parameters",
        "shared_constraints",
        "assumptions",
        "unknowns",
        "top_module",
        "modules",
    ):
        if key not in manifest:
            errors.append(f"missing manifest field: {key}")
    for key in ("clocks", "resets", "parameters", "shared_constraints", "assumptions", "unknowns"):
        if key in manifest and not isinstance(manifest[key], list):
            errors.append(f"{key} must be a list")

    top = manifest.get("top_module")
    if not isinstance(top, dict):
        errors.append("top_module must be an object")
        return errors
    top_name = str(top.get("name") or "")
    if not HDL_NAME_RE.fullmatch(top_name):
        errors.append(f"invalid top module name: {top_name or '<empty>'}")
    if expected_top and top_name != expected_top:
        errors.append(f"top module must exactly match caller-provided name {expected_top}")
    errors.extend(port_validation_errors(top.get("ports"), f"top module {top_name or '<empty>'}"))

    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        errors.append("modules must be a nonempty list")
        return errors
    module_names: List[str] = []
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"modules[{index}] must be an object")
            continue
        name = str(module.get("name") or "")
        if not HDL_NAME_RE.fullmatch(name):
            errors.append(f"invalid submodule name: {name or '<empty>'}")
        if name == top_name:
            errors.append(f"submodule {name} duplicates the top module")
        module_names.append(name)
        for field_name in ("category", "purpose", "reuse_query"):
            if not str(module.get(field_name) or "").strip():
                errors.append(f"submodule {name or index} has empty {field_name}")
        errors.extend(port_validation_errors(module.get("ports"), f"submodule {name or index}"))
        for field_name in ("behavioral_requirements", "dependencies"):
            if not isinstance(module.get(field_name), list):
                errors.append(f"submodule {name or index} {field_name} must be a list")
        if isinstance(module.get("behavioral_requirements"), list) and not module["behavioral_requirements"]:
            errors.append(f"submodule {name or index} behavioral_requirements must not be empty")

    duplicates = sorted({name for name in module_names if name and module_names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate submodule names: {', '.join(duplicates)}")
    known_names = set(module_names)
    modules_by_name = {
        str(module.get("name") or ""): module
        for module in modules
        if isinstance(module, dict)
    }
    for module in modules:
        if not isinstance(module, dict):
            continue
        name = str(module.get("name") or "")
        dependencies = module.get("dependencies")
        if not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if dependency not in known_names:
                errors.append(f"submodule {name} references unknown dependency {dependency}")
            if dependency == name:
                errors.append(f"submodule {name} depends on itself")

    instances = top.get("instances")
    if not isinstance(instances, list) or not instances:
        errors.append("top_module.instances must be a nonempty list")
    else:
        instance_names: List[str] = []
        instantiated_modules: set[str] = set()
        for index, instance in enumerate(instances):
            if not isinstance(instance, dict):
                errors.append(f"top_module.instances[{index}] must be an object")
                continue
            module_name = str(instance.get("module") or "")
            instance_name = str(instance.get("instance_name") or "")
            if module_name not in known_names:
                errors.append(f"top instance {instance_name or index} references unknown module {module_name}")
            else:
                instantiated_modules.add(module_name)
            if not HDL_NAME_RE.fullmatch(instance_name):
                errors.append(f"invalid top instance name: {instance_name or '<empty>'}")
            connections = instance.get("connections")
            if not isinstance(connections, dict):
                errors.append(f"top instance {instance_name or index} connections must be an object")
            elif module_name in modules_by_name:
                expected_ports = {
                    str(port.get("name") or "")
                    for port in modules_by_name[module_name].get("ports", [])
                    if isinstance(port, dict)
                }
                missing_ports = sorted(expected_ports - set(connections))
                unknown_ports = sorted(set(connections) - expected_ports)
                empty_connections = sorted(
                    str(port)
                    for port, signal in connections.items()
                    if not str(signal).strip()
                )
                if missing_ports:
                    errors.append(
                        f"top instance {instance_name or index} is missing connections for: {', '.join(missing_ports)}"
                    )
                if unknown_ports:
                    errors.append(
                        f"top instance {instance_name or index} has unknown connected ports: {', '.join(unknown_ports)}"
                    )
                if empty_connections:
                    errors.append(
                        f"top instance {instance_name or index} has empty connections for: {', '.join(empty_connections)}"
                    )
            instance_names.append(instance_name)
        duplicate_instances = sorted(
            {name for name in instance_names if name and instance_names.count(name) > 1}
        )
        if duplicate_instances:
            errors.append(f"duplicate top instance names: {', '.join(duplicate_instances)}")
        missing_children = sorted(known_names - instantiated_modules)
        if missing_children:
            errors.append(
                f"one-layer decomposition parent does not directly instantiate children: {', '.join(missing_children)}"
            )

    if isinstance(modules, list) and all(isinstance(module, dict) and module.get("name") for module in modules):
        try:
            dependency_order(manifest)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def port_validation_errors(ports: Any, owner: str) -> List[str]:
    if not isinstance(ports, list) or not ports:
        return [f"{owner} ports must be a nonempty list"]
    errors: List[str] = []
    names: List[str] = []
    for index, port in enumerate(ports):
        if not isinstance(port, dict):
            errors.append(f"{owner} port {index} must be an object")
            continue
        name = str(port.get("name") or "")
        direction = str(port.get("direction") or "").lower()
        if not HDL_NAME_RE.fullmatch(name):
            errors.append(f"{owner} has invalid port name {name or '<empty>'}")
        if direction not in {"input", "output", "inout"}:
            errors.append(f"{owner} port {name or index} has invalid direction {direction or '<empty>'}")
        if not str(port.get("width") or "").strip():
            errors.append(f"{owner} port {name or index} has empty width")
        if not str(port.get("description") or "").strip():
            errors.append(f"{owner} port {name or index} has empty description")
        names.append(name)
    duplicates = sorted({name for name in names if name and names.count(name) > 1})
    if duplicates:
        errors.append(f"{owner} has duplicate ports: {', '.join(duplicates)}")
    return errors


def dependency_order(manifest: Dict[str, Any]) -> List[str]:
    dependencies = {
        str(module["name"]): [str(item) for item in module.get("dependencies", [])]
        for module in manifest["modules"]
    }
    order: List[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"submodule dependency cycle includes {name}")
        visiting.add(name)
        for dependency in dependencies.get(name, []):
            if dependency not in dependencies:
                raise ValueError(f"submodule {name} references unknown dependency {dependency}")
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for module_name in dependencies:
        visit(module_name)
    return order


def render_manifest_index(manifest: Dict[str, Any]) -> str:
    top = manifest["top_module"]
    lines = [
        f"TOP MODULE: {top['name']}",
        "",
        "SYSTEM SUMMARY",
        str(manifest.get("system_summary") or "unknown"),
        "",
        "TOP PUBLIC INTERFACE",
        json.dumps(top["ports"], ensure_ascii=False, indent=2),
        "",
        "TOP SUBMODULE INSTANCES AND CONNECTIONS",
        json.dumps(top["instances"], ensure_ascii=False, indent=2),
        "",
        "SHARED CLOCKS / RESETS / PARAMETERS / CONSTRAINTS",
        json.dumps(
            {
                "clocks": manifest.get("clocks", []),
                "resets": manifest.get("resets", []),
                "parameters": manifest.get("parameters", []),
                "shared_constraints": manifest.get("shared_constraints", []),
                "assumptions": manifest.get("assumptions", []),
                "unknowns": manifest.get("unknowns", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "SUBMODULE SPECIFICATION INDEX",
    ]
    for module in manifest["modules"]:
        lines.extend(
            [
                f"- {module['name']}: specs/{module['name']}.txt",
                f"  purpose: {module['purpose']}",
                f"  dependencies: {', '.join(module['dependencies']) or 'none'}",
                f"  interface: {json.dumps(module['ports'], ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_module_spec(
    manifest: Dict[str, Any],
    module: Dict[str, Any],
    module_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    lookup = module_lookup or {
        item["name"]: item
        for item in manifest.get("modules", [])
        if isinstance(item, dict) and item.get("name")
    }
    payload = {
        "module": module,
        "direct_dependency_interfaces": {
            name: lookup[name].get("ports", [])
            for name in module.get("dependencies", [])
            if name in lookup
        },
        "system_summary": manifest.get("system_summary"),
        "clocks": manifest.get("clocks", []),
        "resets": manifest.get("resets", []),
        "parameters": manifest.get("parameters", []),
        "shared_constraints": manifest.get("shared_constraints", []),
        "assumptions": manifest.get("assumptions", []),
        "unknowns": manifest.get("unknowns", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def markdown_chunks(text: str, max_chars: int) -> List[str]:
    max_chars = max(int(max_chars), 1000)
    sections: List[str] = []
    current: List[str] = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("#") and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))

    chunks: List[str] = []
    buffer = ""
    for section in sections:
        if len(section) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(split_bounded_text(section, max_chars))
        elif buffer and len(buffer) + len(section) > max_chars:
            chunks.append(buffer)
            buffer = section
        else:
            buffer += section
    if buffer:
        chunks.append(buffer)
    return chunks or [text]


def group_manifest_payloads(payloads: List[Dict[str, Any]], max_chars: int) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    for payload in payloads:
        payload_chars = len(json.dumps(payload, ensure_ascii=False))
        if current and current_chars + payload_chars > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(payload)
        current_chars += payload_chars
    if current:
        groups.append(current)
    if len(payloads) > 1 and all(len(group) == 1 for group in groups):
        return [payloads[index : index + 2] for index in range(0, len(payloads), 2)]
    return groups


def split_bounded_text(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_chars:
        boundary = remaining.rfind("\n\n", 0, max_chars)
        if boundary < max_chars // 2:
            boundary = remaining.rfind("\n", 0, max_chars)
        if boundary < max_chars // 2:
            boundary = max_chars
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def render_condensed_spec(
    manifest: Dict[str, Any],
    *,
    view: str = "generation",
    provided_modules: Optional[List[str]] = None,
    interface_excerpts: str = "",
) -> str:
    """Render a chunk&merge manifest as one compact condensed spec.

    Unlike the per-module workspace specs, every shared fact (clocks, resets,
    constraints) and every module interface appears exactly once, and sections
    are ordered by importance so head-truncation drops the least useful tail.

    view="planner": short behavioral digest per module, includes reuse queries.
    view="generation": full behavioral requirements for modules that must be
    generated; modules named in provided_modules are interface-only.
    """
    if view not in {"planner", "generation"}:
        raise ValueError("view must be 'planner' or 'generation'")
    provided = {str(name) for name in provided_modules or []}
    top = manifest.get("top_module") or {}
    modules = [item for item in manifest.get("modules", []) if isinstance(item, dict)]

    sections: List[str] = []
    sections.append(
        "\n".join(
            [
                f"TOP MODULE: {top.get('name', 'unknown')}",
                "",
                "SYSTEM SUMMARY",
                str(manifest.get("system_summary") or "unknown"),
                "",
                "TOP PUBLIC INTERFACE",
                json.dumps(top.get("ports", []), ensure_ascii=False, indent=1),
                "",
                "TOP SUBMODULE INSTANCES AND CONNECTIONS",
                json.dumps(top.get("instances", []), ensure_ascii=False, indent=1),
                "",
                "SHARED CLOCKS / RESETS / PARAMETERS / CONSTRAINTS",
                json.dumps(
                    {
                        "clocks": manifest.get("clocks", []),
                        "resets": manifest.get("resets", []),
                        "parameters": manifest.get("parameters", []),
                        "shared_constraints": manifest.get("shared_constraints", []),
                        "assumptions": manifest.get("assumptions", []),
                        "unknowns": manifest.get("unknowns", []),
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
            ]
        )
    )

    if interface_excerpts.strip():
        sections.append(
            "AUTHORITATIVE INTERFACE EXCERPTS (verbatim from the original specification; "
            "trust these over any summarized interface above)\n" + interface_excerpts.strip()
        )

    interface_lines = ["MODULE INTERFACES (each module listed exactly once)"]
    ordered = _condensed_module_order(manifest, modules, provided)
    for module in ordered:
        name = module.get("name", "unknown")
        tag = " [provided reusable IP - instantiate, do not regenerate]" if name in provided else ""
        interface_lines.append(f"- {name} ({module.get('category', 'unknown')}){tag}")
        interface_lines.append(f"  purpose: {module.get('purpose', 'unknown')}")
        dependencies = ", ".join(string_list(module.get("dependencies"))) or "none"
        interface_lines.append(f"  dependencies: {dependencies}")
        interface_lines.append(f"  ports: {json.dumps(module.get('ports', []), ensure_ascii=False)}")
        if view == "planner" and module.get("reuse_query"):
            interface_lines.append(f"  reuse_query: {module['reuse_query']}")
    sections.append("\n".join(interface_lines))

    requirement_lines = ["MODULE BEHAVIORAL REQUIREMENTS"]
    digest_limit = 3 if view == "planner" else None
    has_requirements = False
    for module in ordered:
        name = module.get("name", "unknown")
        if name in provided:
            continue
        requirements = string_list(module.get("behavioral_requirements"))
        if not requirements:
            continue
        has_requirements = True
        kept = requirements if digest_limit is None else requirements[:digest_limit]
        requirement_lines.append(f"=== {name} ===")
        requirement_lines.extend(f"- {item}" for item in kept)
        if digest_limit is not None and len(requirements) > digest_limit:
            requirement_lines.append(f"- ... plus {len(requirements) - digest_limit} more requirement(s)")
    if has_requirements:
        sections.append("\n".join(requirement_lines))

    return "\n\n".join(sections).rstrip() + "\n"


def _condensed_module_order(
    manifest: Dict[str, Any],
    modules: List[Dict[str, Any]],
    provided: set,
) -> List[Dict[str, Any]]:
    """Modules needing generation first (in dependency order), provided IP last,
    so head-truncation keeps what the generator actually has to build."""
    by_name = {module.get("name"): module for module in modules}
    try:
        order = [name for name in dependency_order(manifest) if name in by_name]
    except Exception:  # noqa: BLE001 - ordering is best-effort.
        order = [module.get("name") for module in modules]
    order.extend(name for name in by_name if name not in order)
    generated = [by_name[name] for name in order if name not in provided]
    reused = [by_name[name] for name in order if name in provided]
    return [*generated, *reused]


_PORT_TABLE_HINT_RE = re.compile(r"port|signal|direction|width|input|output", re.IGNORECASE)
_MODULE_HEADER_RE = re.compile(r"(?ms)^[ \t]*module\s+[A-Za-z_]\w*.*?;")


def verbatim_interface_excerpts(raw_spec: str, max_chars: int = 20000) -> str:
    """Deterministically pull port tables and module headers out of a raw spec.

    These are the facts an LLM summary is most likely to silently mangle, so they
    are carried into the condensed spec verbatim, bypassing the LLM entirely.
    """
    blocks: List[str] = []

    table: List[str] = []
    for line in [*raw_spec.splitlines(), ""]:
        if line.lstrip().startswith("|"):
            table.append(line.rstrip())
            continue
        if len(table) >= 3 and _PORT_TABLE_HINT_RE.search("\n".join(table[:2])):
            blocks.append("\n".join(table))
        table = []

    seen_headers: set = set()
    for match in _MODULE_HEADER_RE.finditer(raw_spec):
        header = match.group(0).strip()
        if len(header) > 4000:
            header = header[:4000] + " /* header truncated */"
        name_match = re.match(r"module\s+([A-Za-z_]\w*)", header)
        key = name_match.group(1) if name_match else header[:80]
        if key in seen_headers:
            continue
        seen_headers.add(key)
        blocks.append(header)

    kept: List[str] = []
    used = 0
    for block in blocks:
        if used + len(block) + 2 > max_chars:
            continue
        kept.append(block)
        used += len(block) + 2
    return "\n\n".join(kept)


_MACRO_RE = re.compile(r"`\s?([A-Za-z_]\w*)")
_VERILOG_DIRECTIVES = {
    "include", "define", "undef", "undefineall", "ifdef", "ifndef", "elsif", "else",
    "endif", "timescale", "default_nettype", "resetall", "celldefine", "endcelldefine",
    "line", "pragma", "begin_keywords", "end_keywords",
}


def condensation_fidelity_report(raw_spec: str, condensed: str) -> Dict[str, Any]:
    """Deterministic check that safety-critical identifiers survived condensation:
    macro names and identifiers from port-table rows. Misses become visible signal
    instead of silent loss."""
    macros = {
        name for name in _MACRO_RE.findall(raw_spec) if name.lower() not in _VERILOG_DIRECTIVES
    }
    identifiers: set = set()
    for line in raw_spec.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("`*") for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0]
        if re.fullmatch(r"[A-Za-z_]\w*", first) and not set(first) <= {"-", ":"}:
            identifiers.add(first)
    identifiers -= {"input", "output", "inout", "Port", "port", "Signal", "signal", "Name", "name"}

    missing_macros = sorted(name for name in macros if name not in condensed)
    missing_identifiers = sorted(name for name in identifiers if name not in condensed)
    return {
        "macros_total": len(macros),
        "macros_missing": missing_macros,
        "identifiers_total": len(identifiers),
        "identifiers_missing": missing_identifiers,
    }
