from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .json_utils import json_default
from .types import IpRepository
from .verilog_tools import (
    check_port_compatibility as _check_port_compat,
    generate_rtl_module as _generate_rtl,
    validate_verilog as _validate_verilog,
)


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_reuse_ip",
            "description": "Search the reusable-IP catalog for candidates relevant to a module or system need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {"type": "object"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_reuse_ip",
            "description": "Inspect one IP candidate's behavior, interface, parameters, limits, and integration notes.",
            "parameters": {
                "type": "object",
                "properties": {"ip_id": {"type": "string"}},
                "required": ["ip_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_ip_candidate",
            "description": "Score one IP against a module's requirements using explicit IP-reuse criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip_id": {"type": "string"},
                    "module_requirements": {"type": "object"},
                },
                "required": ["ip_id", "module_requirements"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_artifact",
            "description": "Write an artifact into the configured output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_rtl_module",
            "description": (
                "Write a complete synthesizable SystemVerilog module to a .sv file in the output directory. "
                "Provide the full RTL code in verilog_code. Call validate_verilog afterward to catch syntax errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_name": {"type": "string", "description": "Top-level module name"},
                    "file_path": {"type": "string", "description": "Relative output path, e.g. rtl/my_module.sv"},
                    "verilog_code": {"type": "string", "description": "Complete .sv file content"},
                    "description": {"type": "string", "description": "One-line functional description"},
                },
                "required": ["module_name", "file_path", "verilog_code", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_verilog",
            "description": (
                "Lint and syntax-check a generated .sv file using verilator or iverilog. "
                "Returns errors and warnings. If errors are found, fix the code and call generate_rtl_module again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path of the .sv file to lint"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_port_compatibility",
            "description": (
                "Parse port declarations from two .sv files and verify width and direction compatibility "
                "for specified port connections. Returns mismatches and confirmed matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_a_path": {"type": "string", "description": "Relative path to the first module .sv file"},
                    "module_b_path": {"type": "string", "description": "Relative path to the second module .sv file"},
                    "port_pairs": {
                        "type": "array",
                        "description": "List of port pairs to check, e.g. [{\"a\": \"data_out\", \"b\": \"data_in\"}]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "string"},
                                "b": {"type": "string"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                },
                "required": ["module_a_path", "module_b_path", "port_pairs"],
            },
        },
    },
]


class AgentToolExecutor:
    def __init__(self, repository: IpRepository, output_dir: str | Path):
        self.repository = repository
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name == "search_reuse_ip":
                payload = self.search_reuse_ip(
                    query=str(arguments.get("query", "")),
                    filters=_dict_or_none(arguments.get("filters")),
                    top_k=int(arguments.get("top_k", 5)),
                )
            elif name == "inspect_reuse_ip":
                payload = self.inspect_reuse_ip(ip_id=str(arguments.get("ip_id", "")))
            elif name == "evaluate_ip_candidate":
                payload = self.evaluate_ip_candidate(
                    ip_id=str(arguments.get("ip_id", "")),
                    module_requirements=_dict_or_none(arguments.get("module_requirements")) or {},
                )
            elif name == "write_artifact":
                payload = self.write_artifact(
                    path=str(arguments.get("path", "")),
                    content=str(arguments.get("content", "")),
                )
            elif name == "generate_rtl_module":
                payload = self.generate_rtl_module(
                    module_name=str(arguments.get("module_name", "")),
                    file_path=str(arguments.get("file_path", "")),
                    verilog_code=str(arguments.get("verilog_code", "")),
                    description=str(arguments.get("description", "")),
                )
            elif name == "validate_verilog":
                payload = self.validate_verilog(
                    file_path=str(arguments.get("file_path", "")),
                )
            elif name == "check_port_compatibility":
                pairs = arguments.get("port_pairs") or []
                if not isinstance(pairs, list):
                    pairs = []
                payload = self.check_port_compatibility(
                    module_a_path=str(arguments.get("module_a_path", "")),
                    module_b_path=str(arguments.get("module_b_path", "")),
                    port_pairs=pairs,
                )
            else:
                payload = {"ok": False, "tool": name, "error": f"unknown tool: {name}"}
        except Exception as exc:
            payload = {"ok": False, "tool": name, "error": str(exc)}
        return json.dumps(payload, default=json_default, ensure_ascii=False)

    def search_reuse_ip(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> Dict[str, Any]:
        candidates = self.repository.search(query=query, filters=filters, top_k=top_k)
        return {"ok": True, "tool": "search_reuse_ip", "query": query, "candidates": [asdict(item) for item in candidates]}

    def inspect_reuse_ip(self, ip_id: str) -> Dict[str, Any]:
        return {"ok": True, "tool": "inspect_reuse_ip", "description": asdict(self.repository.inspect(ip_id))}

    def evaluate_ip_candidate(self, ip_id: str, module_requirements: Dict[str, Any]) -> Dict[str, Any]:
        description = self.repository.inspect(ip_id)
        assessment = self.repository.score(description.candidate, module_requirements)
        return {"ok": True, "tool": "evaluate_ip_candidate", "assessment": asdict(assessment)}

    def write_artifact(self, path: str, content: str) -> Dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "tool": "write_artifact", "path": str(target), "bytes": len(content.encode("utf-8"))}

    def generate_rtl_module(
        self,
        module_name: str,
        file_path: str,
        verilog_code: str,
        description: str,
    ) -> Dict[str, Any]:
        result = _generate_rtl(module_name, file_path, verilog_code, description, self.output_dir)
        result["tool"] = "generate_rtl_module"
        return result

    def validate_verilog(self, file_path: str) -> Dict[str, Any]:
        result = _validate_verilog(file_path, self.output_dir)
        result["tool"] = "validate_verilog"
        return result

    def check_port_compatibility(
        self,
        module_a_path: str,
        module_b_path: str,
        port_pairs: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        result = _check_port_compat(module_a_path, module_b_path, port_pairs, self.output_dir)
        result["tool"] = "check_port_compatibility"
        return result

    def _resolve(self, path: str) -> Path:
        if not path:
            raise ValueError("path must not be empty")
        target = (self.output_dir / path).resolve()
        if target != self.output_dir and self.output_dir not in target.parents:
            raise ValueError(f"artifact path escapes output directory: {path}")
        return target


def _dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None
