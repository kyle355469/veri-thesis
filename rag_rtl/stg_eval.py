from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .llm import extract_code
from .types import RtlTask

MODULE_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")

SPEC_FIELDS = ("spec", "prompt", "problem", "instruction", "description")
GOLDEN_FIELDS = ("golden_code", "golden", "reference_code", "reference", "solution", "completion")


@dataclass
class StgRunResult:
    passed: bool
    generate_returncode: Optional[int]
    execute_returncode: Optional[int]
    stdout: str
    stderr: str
    command: List[str]
    run_command: List[str]


def iter_dataset_records(path: str | Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for index, record in enumerate(records):
            if limit is not None and index >= limit:
                break
            if isinstance(record, dict):
                yield record
        return

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def run_stg_equivalence(
    candidate_code: str,
    golden_code: str,
    *,
    stg_bin: str = "stg",
    design_type: str = "combinational",
    dut_module: Optional[str] = None,
    golden_module: Optional[str] = None,
    timeout_s: int = 120,
    extra_stg_args: Sequence[str] = (),
) -> StgRunResult:
    if shutil.which(stg_bin) is None and not Path(stg_bin).exists():
        return StgRunResult(
            passed=False,
            generate_returncode=None,
            execute_returncode=None,
            stdout="",
            stderr=f"stg binary not found: {stg_bin}",
            command=[stg_bin],
            run_command=[],
        )
    stg_command = str(Path(stg_bin).resolve()) if Path(stg_bin).exists() else stg_bin

    with tempfile.TemporaryDirectory(prefix="rag_rtl_stg_") as tempdir_text:
        tempdir = Path(tempdir_text)
        dut_path = tempdir / "candidate.v"
        golden_path = tempdir / "golden.v"
        tb_path = tempdir / "tb.sv"
        exe_path = tempdir / "tb_exe"
        dut_path.write_text(candidate_code, encoding="utf-8")
        golden_path.write_text(golden_code, encoding="utf-8")

        command = [
            # "CCACHE_DISABLE=1",
            stg_command,
            "generate",
            "--verilog",
            str(dut_path),
            "--golden",
            str(golden_path),
            "--type",
            design_type,
            "--out",
            str(tb_path),
            "--out-exe",
            str(exe_path),
            "--emplace-module",
            "--verilator",
            *extra_stg_args,
        ]
        if dut_module:
            command.extend(["--module", dut_module])
        if golden_module:
            command.extend(["--golden-module", golden_module])

        try:
            generated = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tempdir,
            )
        except subprocess.TimeoutExpired as exc:
            return StgRunResult(
                passed=False,
                generate_returncode=None,
                execute_returncode=None,
                stdout=exc.stdout or "",
                stderr=str(exc),
                command=command,
                run_command=[],
            )

        if generated.returncode != 0:
            print(generated)
            print(f"STG generation failed with return code {generated.returncode}")
            return StgRunResult(
                passed=False,
                generate_returncode=generated.returncode,
                execute_returncode=None,
                stdout=generated.stdout,
                stderr=generated.stderr,
                command=command,
                run_command=[],
            )

        run_command = [str(exe_path)]
        try:
            executed = subprocess.run(
                run_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tempdir,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"STG execution timed out after {timeout_s} seconds")
            return StgRunResult(
                passed=False,
                generate_returncode=generated.returncode,
                execute_returncode=None,
                stdout=(generated.stdout or "") + "\n" + (exc.stdout or ""),
                stderr=(generated.stderr or "") + "\n" + str(exc),
                command=command,
                run_command=run_command,
            )

        return StgRunResult(
            passed=executed.returncode == 0,
            generate_returncode=generated.returncode,
            execute_returncode=executed.returncode,
            stdout=(generated.stdout or "") + "\n" + (executed.stdout or ""),
            stderr=(generated.stderr or "") + "\n" + (executed.stderr or ""),
            command=command,
            run_command=run_command,
        )


def run_stg_dataset_evaluation(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    pipeline: Any,
    stg_bin: str = "stg",
    target_hdl: str = "verilog",
    default_design_type: str = "combinational",
    limit: Optional[int] = None,
    timeout_s: int = 120,
    spec_field: Optional[str] = None,
    golden_field: Optional[str] = None,
    save_result_code_dir: Optional[str | Path] = None,
    save_passed_dir: Optional[str | Path] = None,
    extra_stg_args: Sequence[str] = (),
    retrieve_k: int = 8,
    context_k: int = 4,
    max_repair_attempts: int = 1,
    module_signature: Optional[str] = None,
    constraints: Sequence[str] = (),
    top_module: Optional[str] = None,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_code_dir = Path(save_result_code_dir) if save_result_code_dir else None
    if result_code_dir:
        result_code_dir.mkdir(parents=True, exist_ok=True)
    passed_dir = Path(save_passed_dir) if save_passed_dir else None
    if passed_dir:
        passed_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for index, record in enumerate(iter_dataset_records(dataset_path, limit=limit)):
        spec = _pick_text(record, [spec_field] if spec_field else SPEC_FIELDS)
        golden_text = _pick_text(record, [golden_field] if golden_field else GOLDEN_FIELDS)
        golden_code = extract_code(golden_text)
        design_type = str(record.get("type") or record.get("design_type") or default_design_type)
        record_top_module = _pick_optional_text(record, ("top_module", "module", "dut_module"))
        dut_module = record_top_module
        golden_module = _pick_optional_text(record, ("golden_module", "reference_module")) or infer_first_module_name(golden_code)

        result: Dict[str, Any] = {
            "index": index,
            "id": record.get("id") or record.get("task_id") or record.get("problem_id"),
            "design_type": design_type,
            "dut_module": dut_module,
            "golden_module": golden_module,
        }
        if not spec or not golden_code:
            result.update(
                {
                    "passed": False,
                    "error": "missing spec or golden code",
                    "has_spec": bool(spec),
                    "has_golden_code": bool(golden_code),
                }
            )
            records.append(result)
            continue

        task = RtlTask(
            prompt=spec,
            target_hdl=target_hdl,
            module_signature=_record_text(record, "module_signature") or module_signature,
            constraints=_record_constraints(record) or list(constraints),
            max_repair_attempts=max_repair_attempts,
            top_module=top_module or record_top_module,
        )
        response = pipeline.run(task, retrieve_k=retrieve_k, context_k=context_k)
        candidate_code = response.rtl
        result.update(
            {
                "rag_generation_passed": response.verification.passed,
                "syntax_passed": response.verification.syntax_passed,
                "lint_passed": response.verification.lint_passed,
                "repair_attempts": response.repair_attempts,
                "cache_source": response.cache_source,
                "retrieved_doc_ids": response.retrieved_doc_ids,
            }
        )
        if not candidate_code:
            result.update({"passed": False, "error": "RAG pipeline did not return HDL code"})
            records.append(result)
            continue
        result["generated"] = True
        if result_code_dir:
            code_path = result_code_dir / _result_code_filename(index, record, target_hdl)
            code_path.write_text(candidate_code, encoding="utf-8")
            result["generated_code_path"] = str(code_path)
        if not response.verification.passed:
            result.update(
                {
                    "passed": False,
                    "error": "RAG generation did not pass its configured verifier",
                    "verification_diagnostics": [
                        {
                            "tool": diagnostic.tool,
                            "passed": diagnostic.passed,
                            "returncode": diagnostic.returncode,
                            "missing": diagnostic.missing,
                            "stdout_tail": diagnostic.stdout[-2000:],
                            "stderr_tail": diagnostic.stderr[-2000:],
                        }
                        for diagnostic in response.verification.diagnostics
                    ],
                }
            )
            records.append(result)
            continue

        dut_module = record_top_module or infer_first_module_name(candidate_code)
        result["dut_module"] = dut_module
        stg_result = run_stg_equivalence(
            candidate_code,
            golden_code,
            stg_bin=stg_bin,
            design_type=design_type,
            dut_module=dut_module,
            golden_module=golden_module,
            timeout_s=timeout_s,
            extra_stg_args=extra_stg_args,
        )
        result.update(
            {
                "passed": stg_result.passed,
                "generate_returncode": stg_result.generate_returncode,
                "execute_returncode": stg_result.execute_returncode,
                "stdout_tail": stg_result.stdout[-4000:],
                "stderr_tail": stg_result.stderr[-4000:],
                "stg_command": stg_result.command,
                "run_command": stg_result.run_command,
            }
        )
        if stg_result.passed and passed_dir:
            code_path = passed_dir / f"passed_{index:05d}.v"
            code_path.write_text(candidate_code, encoding="utf-8")
            result["passed_code_path"] = str(code_path)
        records.append(result)

    count = max(len(records), 1)
    passed_count = sum(1 for item in records if item.get("passed"))
    summary = {
        "dataset": str(dataset_path),
        "num_records": len(records),
        "generated": sum(1 for item in records if item.get("generated")),
        "rag_generation_passed": sum(1 for item in records if item.get("rag_generation_passed")),
        "syntax_passed": sum(1 for item in records if item.get("syntax_passed")),
        "lint_passed": sum(1 for item in records if item.get("lint_passed")),
        "stg_checked": sum(1 for item in records if "generate_returncode" in item),
        "passed": passed_count,
        "pass_rate": passed_count / count,
        "total_s": time.perf_counter() - start,
        "result_code_dir": str(result_code_dir) if result_code_dir else None,
        "records": records,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def infer_first_module_name(code: str) -> Optional[str]:
    match = MODULE_RE.search(code)
    return match.group(1) if match else None


def _result_code_filename(index: int, record: Dict[str, Any], target_hdl: str) -> str:
    raw_id = record.get("id") or record.get("task_id") or record.get("problem_id")
    stem = f"result_{index:05d}"
    if raw_id is not None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_id)).strip("._")
        if safe_id:
            stem = f"{stem}_{safe_id[:80]}"
    return f"{stem}{_hdl_extension(target_hdl)}"


def _hdl_extension(target_hdl: str) -> str:
    normalized = target_hdl.lower()
    if normalized in {"systemverilog", "sv"}:
        return ".sv"
    if normalized in {"vhdl", "vhd"}:
        return ".vhd"
    return ".v"


def _pick_text(record: Dict[str, Any], fields: Sequence[Optional[str]]) -> str:
    for field in fields:
        if not field:
            continue
        value = record.get(field)
        text = _stringify_field(value)
        if text:
            return text
    return ""


def _pick_optional_text(record: Dict[str, Any], fields: Sequence[str]) -> Optional[str]:
    text = _pick_text(record, fields)
    return text or None


def _record_text(record: Dict[str, Any], field: str) -> str:
    return _stringify_field(record.get(field))


def _record_constraints(record: Dict[str, Any]) -> List[str]:
    value = record.get("constraints")
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _stringify_field(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""
