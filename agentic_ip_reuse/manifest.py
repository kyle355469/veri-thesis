from __future__ import annotations

import json
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
