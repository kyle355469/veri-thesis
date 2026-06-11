from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PORT_RE = re.compile(
    r"\b(input|output|inout)\b"
    r"(?:\s+(?:wire|reg|logic|tri|supply[01]|var))?"
    r"(?:\s+(?:signed|unsigned))?"
    r"(?:\s*\[([^\]]+)\])?"
    r"\s+(\w+)",
    re.MULTILINE,
)

MODULE_NAME_RE = re.compile(r"\bmodule\s+(\w+)", re.MULTILINE)

FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*?)```\s*$", re.DOTALL)


@dataclass
class ParsedPort:
    name: str
    direction: str
    msb: Optional[int] = None
    lsb: Optional[int] = None
    parametric_range: Optional[str] = None

    @property
    def width(self) -> Optional[int]:
        if self.msb is not None and self.lsb is not None:
            return abs(self.msb - self.lsb) + 1
        if self.parametric_range is None and self.msb is None:
            return 1
        return None


def generate_rtl_module(
    module_name: str,
    file_path: str,
    verilog_code: str,
    description: str,
    output_dir: Path,
) -> Dict[str, Any]:
    code = _strip_fence(verilog_code)
    target = _resolve_safe(output_dir, file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")

    ports = _parse_ports(code)
    detected = _detect_module_name(code)

    return {
        "ok": True,
        "path": str(target),
        "module_name": detected or module_name,
        "description": description,
        "ports_detected": [
            {
                "name": p.name,
                "direction": p.direction,
                "width": p.width,
                "range": p.parametric_range,
            }
            for p in ports
        ],
        "bytes": len(code.encode("utf-8")),
    }


def validate_verilog(file_path: str, output_dir: Path) -> Dict[str, Any]:
    target = _resolve_safe(output_dir, file_path)
    if not target.exists():
        return {"ok": False, "error": f"file not found: {file_path}", "errors": [], "warnings": []}

    if shutil.which("verilator"):
        return _run_verilator(target)
    if shutil.which("iverilog"):
        return _run_iverilog(target)
    return _offline_syntax_check(target)


def check_port_compatibility(
    module_a_path: str,
    module_b_path: str,
    port_pairs: List[Dict[str, str]],
    output_dir: Path,
) -> Dict[str, Any]:
    path_a = _resolve_safe(output_dir, module_a_path)
    path_b = _resolve_safe(output_dir, module_b_path)

    if not path_a.exists():
        return {"ok": False, "error": f"file not found: {module_a_path}"}
    if not path_b.exists():
        return {"ok": False, "error": f"file not found: {module_b_path}"}

    code_a = path_a.read_text(encoding="utf-8")
    code_b = path_b.read_text(encoding="utf-8")
    ports_a = {p.name: p for p in _parse_ports(code_a)}
    ports_b = {p.name: p for p in _parse_ports(code_b)}
    mod_a = _detect_module_name(code_a) or module_a_path
    mod_b = _detect_module_name(code_b) or module_b_path

    if not port_pairs:
        return {
            "ok": True,
            "module_a": mod_a,
            "module_b": mod_b,
            "ports_a": [{"name": p.name, "direction": p.direction, "width": p.width} for p in ports_a.values()],
            "ports_b": [{"name": p.name, "direction": p.direction, "width": p.width} for p in ports_b.values()],
            "issues": [],
            "matched": [],
            "note": "No port_pairs specified; returned all ports for inspection.",
        }

    issues: List[Dict[str, str]] = []
    matched: List[Dict[str, str]] = []

    for pair in port_pairs:
        name_a = str(pair.get("a", ""))
        name_b = str(pair.get("b", ""))
        port_a = ports_a.get(name_a)
        port_b = ports_b.get(name_b)

        if port_a is None:
            issues.append({"port_a": name_a, "port_b": name_b, "issue": f"port '{name_a}' not found in {mod_a}"})
            continue
        if port_b is None:
            issues.append({"port_a": name_a, "port_b": name_b, "issue": f"port '{name_b}' not found in {mod_b}"})
            continue

        dir_ok = _direction_compatible(port_a.direction, port_b.direction)
        if not dir_ok:
            issues.append({
                "port_a": name_a,
                "port_b": name_b,
                "issue": (
                    f"direction mismatch: {mod_a}.{name_a} is {port_a.direction}, "
                    f"{mod_b}.{name_b} is {port_b.direction}"
                ),
            })

        w_a, w_b = port_a.width, port_b.width
        if w_a is not None and w_b is not None and w_a != w_b:
            issues.append({
                "port_a": name_a,
                "port_b": name_b,
                "issue": (
                    f"width mismatch: {mod_a}.{name_a} is {w_a}-bit, "
                    f"{mod_b}.{name_b} is {w_b}-bit"
                ),
            })
        elif (w_a is None or w_b is None) and (port_a.parametric_range or port_b.parametric_range):
            issues.append({
                "port_a": name_a,
                "port_b": name_b,
                "issue": (
                    f"parametric width: {mod_a}.{name_a} range=[{port_a.parametric_range}], "
                    f"{mod_b}.{name_b} range=[{port_b.parametric_range}]; verify parameters match."
                ),
            })

        if dir_ok and not any(p["port_a"] == name_a and p["port_b"] == name_b for p in issues):
            matched.append({"a": name_a, "b": name_b, "width": w_a})

    return {
        "ok": len(issues) == 0,
        "module_a": mod_a,
        "module_b": mod_b,
        "issues": issues,
        "matched": matched,
    }


def _parse_ports(code: str) -> List[ParsedPort]:
    ports: List[ParsedPort] = []
    for m in PORT_RE.finditer(code):
        direction = m.group(1)
        range_str = m.group(2)
        name = m.group(3)
        msb = lsb = None
        parametric: Optional[str] = None
        if range_str:
            parts = range_str.split(":")
            if len(parts) == 2:
                try:
                    msb = int(parts[0].strip())
                    lsb = int(parts[1].strip())
                except ValueError:
                    parametric = range_str.strip()
            else:
                parametric = range_str.strip()
        ports.append(ParsedPort(name=name, direction=direction, msb=msb, lsb=lsb, parametric_range=parametric))
    return ports


def _detect_module_name(code: str) -> Optional[str]:
    m = MODULE_NAME_RE.search(code)
    return m.group(1) if m else None


def _direction_compatible(dir_a: str, dir_b: str) -> bool:
    if "inout" in (dir_a, dir_b):
        return True
    return {dir_a, dir_b} == {"input", "output"}


def _strip_fence(text: str) -> str:
    m = FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _run_verilator(path: Path) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["verilator", "--lint-only", "--sv", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "linter": "verilator", "error": "timeout", "errors": [], "warnings": []}
    errors = [line for line in result.stderr.splitlines() if "%Error" in line]
    warnings = [line for line in result.stderr.splitlines() if "%Warning" in line]
    return {
        "ok": result.returncode == 0,
        "linter": "verilator",
        "errors": errors,
        "warnings": warnings,
        "raw_stderr": result.stderr[:2000],
    }


def _run_iverilog(path: Path) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["iverilog", "-g2012", "-t", "null", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "linter": "iverilog", "error": "timeout", "errors": [], "warnings": []}
    all_lines = result.stderr.splitlines() + result.stdout.splitlines()
    errors = [line for line in all_lines if "error:" in line.lower()]
    warnings = [line for line in all_lines if "warning:" in line.lower()]
    return {
        "ok": result.returncode == 0,
        "linter": "iverilog",
        "errors": errors,
        "warnings": warnings,
        "raw_stderr": result.stderr[:2000],
    }


def _offline_syntax_check(path: Path) -> Dict[str, Any]:
    code = path.read_text(encoding="utf-8")
    errors: List[str] = []
    n_module = len(re.findall(r"\bmodule\b", code))
    n_endmodule = len(re.findall(r"\bendmodule\b", code))
    if n_module != n_endmodule:
        errors.append(f"module/endmodule count mismatch: {n_module} module vs {n_endmodule} endmodule")
    if n_module == 0:
        errors.append("no module declaration found")
    return {
        "ok": len(errors) == 0,
        "linter": "offline_regex",
        "errors": errors,
        "warnings": ["verilator and iverilog not found; only structural checks performed — install either for full lint"],
        "note": "Install verilator (apt install verilator) or iverilog (apt install iverilog) for accurate results.",
    }


def _resolve_safe(output_dir: Path, path: str) -> Path:
    if not path:
        raise ValueError("path must not be empty")
    target = (output_dir / path).resolve()
    if target != output_dir and output_dir not in target.parents:
        raise ValueError(f"path escapes output directory: {path}")
    return target
