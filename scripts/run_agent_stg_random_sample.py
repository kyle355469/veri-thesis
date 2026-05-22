#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.llm import extract_code
from rag_rtl.stg_eval import infer_first_module_name, run_stg_equivalence
from rtl_agent.agent import dumps_result
from rtl_agent.cli import build_agent


DEFAULT_DATASET = "/home/kai/silicon_mind_dataset/prompts/train-qwen_46k.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample train-qwen style records, run the agent, and measure "
            "process errors separately from final STG verification failures."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", default="runs/train_qwen_46k_random100_agent_stg")
    parser.add_argument("--sample-file", help="Use an existing normalized sample file instead of drawing a new sample.")
    parser.add_argument("--resume", action="store_true", help="Skip records that already have a per-sample result JSON.")
    parser.add_argument("--preflight-only", action="store_true", help="Check the first sample and stop.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of samples to run concurrently.")

    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--target-hdl", default="verilog")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--llm-timeout-s", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-steps", type=int, default=8)

    parser.add_argument("--yosys-bin", default="yosys")
    parser.add_argument("--verilator-bin", default="verilator")
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--testbench")
    parser.add_argument("--test-command")
    parser.add_argument("--stg-bin", default="stg")
    parser.add_argument("--stg-type", default="combinational", choices=["combinational", "seq_clocked", "seq_done"])
    parser.add_argument("--stg-timeout-s", type=int, default=120)
    parser.add_argument("--stg-arg", action="append", default=[])
    parser.add_argument("--allow-command", action="append", default=[])
    parser.add_argument("--command-timeout-s", type=int, default=20)
    parser.add_argument("--command-max-output-chars", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = Path(args.sample_file) if args.sample_file else output_dir / "sample.json"
    records = load_or_create_sample(args, sample_path)
    run_count = 1 if args.preflight_only else len(records)

    start = time.perf_counter()
    result_records = run_batch(args, output_dir, records[:run_count])

    summary = build_summary(args, sample_path, result_records, elapsed_s=time.perf_counter() - start)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


def run_batch(args: argparse.Namespace, output_dir: Path, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs = max(1, int(args.jobs or 1))
    total = len(records)
    if jobs == 1:
        return [run_and_write_one(args, output_dir, position, record, total) for position, record in enumerate(records)]

    results: List[Dict[str, Any]] = []
    print(f"Running {total} samples with jobs={jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_and_write_one, args, output_dir, position, record, total): position
            for position, record in enumerate(records)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            position = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep collecting sibling results.
                record = records[position]
                result = {
                    "sample_position": position,
                    "source_index": record.get("source_index"),
                    "id": record.get("id"),
                    "process_error": True,
                    "final_stg_verification_fail": False,
                    "passed": False,
                    "outcome": "worker_exception",
                    "error_stage": "worker",
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc()[-4000:],
                }
                result_path = result_file_path(output_dir, position, record)
                result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append(result)
            print(
                f"[{completed}/{total}] sample={position:03d} source_index={result.get('source_index')} "
                f"outcome={result.get('outcome')} process_error={result.get('process_error')} "
                f"final_stg_fail={result.get('final_stg_verification_fail')}",
                flush=True,
            )
    results.sort(key=lambda item: int(item.get("sample_position") or 0))
    return results


def run_and_write_one(
    args: argparse.Namespace,
    output_dir: Path,
    position: int,
    record: Dict[str, Any],
    total: int,
) -> Dict[str, Any]:
    result_path = result_file_path(output_dir, position, record)
    if args.resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["resumed"] = True
        return result
    if max(1, int(args.jobs or 1)) == 1:
        print(
            f"[{position + 1}/{total}] source_index={record['source_index']} "
            f"id={record.get('id')}",
            flush=True,
        )
    result = run_one(args, output_dir, position, record)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if max(1, int(args.jobs or 1)) == 1:
        print(
            f"  outcome={result['outcome']} process_error={result['process_error']} "
            f"final_stg_fail={result['final_stg_verification_fail']}",
            flush=True,
        )
    return result


def result_file_path(output_dir: Path, position: int, record: Dict[str, Any]) -> Path:
    return output_dir / f"sample_{position:03d}_idx_{record['source_index']}.json"


def load_or_create_sample(args: argparse.Namespace, sample_path: Path) -> List[Dict[str, Any]]:
    if args.sample_file:
        return json.loads(sample_path.read_text(encoding="utf-8"))

    payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    raw_records = payload.get("dataset") if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        raise ValueError(f"{args.dataset}: expected a list or a dict with a dataset list")
    if args.sample_size > len(raw_records):
        raise ValueError(f"sample size {args.sample_size} exceeds dataset size {len(raw_records)}")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32)
    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(raw_records)), args.sample_size)
    records: List[Dict[str, Any]] = []
    for position, source_index in enumerate(sampled_indices):
        raw = raw_records[source_index]
        if not isinstance(raw, dict):
            continue
        spec = str(raw.get("input") or raw.get("spec") or raw.get("prompt") or "").strip()
        golden_text = str(raw.get("output") or raw.get("golden_code") or raw.get("golden") or "").strip()
        golden_code = extract_code(golden_text)
        records.append(
            {
                "sample_position": position,
                "source_index": source_index,
                "id": raw.get("i", source_index),
                "spec": spec,
                "golden_code": golden_code,
                "has_spec": bool(spec),
                "has_golden_code": bool(golden_code),
            }
        )

    sample_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "dataset": str(args.dataset),
        "sample_size": args.sample_size,
        "seed": seed,
        "sample_file": str(sample_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (sample_path.parent / "sample_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return records


def run_one(args: argparse.Namespace, output_dir: Path, position: int, record: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "sample_position": position,
        "source_index": record["source_index"],
        "id": record.get("id"),
        "has_spec": record.get("has_spec"),
        "has_golden_code": record.get("has_golden_code"),
        "process_error": False,
        "final_stg_verification_fail": False,
        "passed": False,
        "outcome": "unknown",
    }
    if not record.get("spec") or not record.get("golden_code"):
        return {
            **base,
            "process_error": True,
            "outcome": "missing_spec_or_golden",
            "error_stage": "dataset",
            "error": "missing spec or golden code",
        }

    run_args = make_agent_args(args, output_dir, position, record)
    try:
        agent = build_agent(run_args)
        from rag_rtl.types import RtlTask

        result = agent.run(
            RtlTask(
                prompt=record["spec"],
                target_hdl=args.target_hdl,
                module_signature=None,
                constraints=[],
                max_repair_attempts=0,
                top_module=None,
                prompt_profile="tool",
            )
        )
        agent_report_path = output_dir / f"agent_{position:03d}_idx_{record['source_index']}.json"
        agent_report_path.write_text(dumps_result(result), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - batch measurement should keep going.
        return {
            **base,
            "process_error": True,
            "outcome": "agent_exception",
            "error_stage": "agent_run",
            "error": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        }

    verification = result.verification
    failed_tools = [item.tool for item in verification.diagnostics if not item.passed]
    base.update(
        {
            "agent_report_path": str(agent_report_path),
            "rtl_chars": len(result.rtl or ""),
            "steps": result.steps,
            "used_tools": result.used_tools,
            "stopped_reason": result.stopped_reason,
            "syntax_passed": verification.syntax_passed,
            "lint_passed": verification.lint_passed,
            "agent_verification_passed": verification.passed,
            "failed_tools": failed_tools,
        }
    )
    if not result.rtl:
        return {
            **base,
            "process_error": True,
            "outcome": "no_parsable_rtl",
            "error_stage": "rtl_extraction",
        }
    if not verification.passed:
        return {
            **base,
            "process_error": True,
            "outcome": "agent_verification_fail",
            "error_stage": "agent_final_verification",
            "verification_diagnostics": diagnostics_to_dicts(verification.diagnostics),
        }

    stg_result = run_stg_equivalence(
        result.rtl,
        record["golden_code"],
        stg_bin=args.stg_bin,
        design_type=args.stg_type,
        dut_module=infer_first_module_name(result.rtl),
        golden_module=infer_first_module_name(record["golden_code"]),
        timeout_s=args.stg_timeout_s,
        extra_stg_args=args.stg_arg or [],
    )
    base.update(
        {
            "stg_generate_returncode": stg_result.generate_returncode,
            "stg_execute_returncode": stg_result.execute_returncode,
            "stg_stdout_tail": stg_result.stdout[-4000:],
            "stg_stderr_tail": stg_result.stderr[-4000:],
        }
    )
    if stg_result.passed:
        return {**base, "passed": True, "outcome": "pass"}

    if stg_result.generate_returncode == 0 and stg_result.execute_returncode not in (0, None):
        return {
            **base,
            "final_stg_verification_fail": True,
            "outcome": "final_stg_verification_fail",
        }
    return {
        **base,
        "process_error": True,
        "outcome": "stg_setup_or_generation_fail",
        "error_stage": "stg_generation",
    }


def make_agent_args(
    args: argparse.Namespace,
    output_dir: Path,
    position: int,
    record: Dict[str, Any],
) -> argparse.Namespace:
    return argparse.Namespace(
        index=args.index,
        embedder=args.embedder,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        llm_timeout_s=args.llm_timeout_s,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tool_choice=args.tool_choice,
        max_steps=args.max_steps,
        target_hdl=args.target_hdl,
        yosys_bin=args.yosys_bin,
        verilator_bin=args.verilator_bin,
        timeout_s=args.timeout_s,
        testbench=args.testbench,
        test_command=args.test_command,
        top_module=None,
        workspace_root=str(output_dir / "workspaces" / f"sample_{position:03d}_idx_{record['source_index']}"),
        allow_command=args.allow_command,
        command_timeout_s=args.command_timeout_s,
        command_max_output_chars=args.command_max_output_chars,
    )


def diagnostics_to_dicts(diagnostics: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "tool": item.tool,
            "passed": item.passed,
            "returncode": item.returncode,
            "missing": item.missing,
            "stdout_tail": item.stdout[-2000:],
            "stderr_tail": item.stderr[-2000:],
        }
        for item in diagnostics
    ]


def build_summary(
    args: argparse.Namespace,
    sample_path: Path,
    records: List[Dict[str, Any]],
    *,
    elapsed_s: float,
) -> Dict[str, Any]:
    total = len(records)
    denom = max(total, 1)
    outcomes = Counter(str(item.get("outcome")) for item in records)
    stages = Counter(str(item.get("error_stage")) for item in records if item.get("process_error"))
    failed_tools = Counter(
        tool for item in records for tool in item.get("failed_tools", []) if item.get("process_error")
    )
    process_errors = sum(1 for item in records if item.get("process_error"))
    final_stg_fails = sum(1 for item in records if item.get("final_stg_verification_fail"))
    passed = sum(1 for item in records if item.get("passed"))
    return {
        "dataset": str(args.dataset),
        "sample_file": str(sample_path),
        "num_records": total,
        "passed": passed,
        "final_stg_verification_fail": final_stg_fails,
        "process_errors_excluding_final_stg_verification_fail": process_errors,
        "process_error_rate": process_errors / denom,
        "pass_rate": passed / denom,
        "final_stg_fail_rate": final_stg_fails / denom,
        "outcomes": dict(outcomes),
        "process_error_stages": dict(stages),
        "process_error_failed_tools": dict(failed_tools),
        "elapsed_s": elapsed_s,
        "preflight_only": bool(args.preflight_only),
        "records": records,
    }


if __name__ == "__main__":
    main()
