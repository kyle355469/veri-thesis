#!/usr/bin/env python3
"""Direct RealBench spec-to-RTL baseline.

This runner feeds each RealBench problem prompt directly to the served model,
saves the first generated RTL, and evaluates it with the RealBench testbench.
It intentionally does not use RAG, planning, tool calls, or repair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.json_utils import dumps_json
from rag_rtl.llm import VllmClient, extract_code

from scripts.run_agentic_plan_legacy_realbench import (
    CatalogBundle,
    RealBenchEvalResult,
    RealBenchTask,
    WorkItem,
    build_task_catalog,
    discover_tasks,
    evaluate_realbench_code,
    generated_code_path,
    record_sort_key,
    safe_rate,
    task_constraints,
    template_provided_module_names,
    work_items,
    write_records,
    write_solution_jsonl,
)

HDL_START_RE = re.compile(r"(?im)^\s*(?:`[a-zA-Z_][a-zA-Z0-9_]*\b.*\n\s*)*(module|interface|package|primitive|program)\b")


@dataclass(frozen=True)
class DirectGeneration:
    prompt: str
    raw_text: str
    code: str
    finish_reason: Optional[str]
    content_from_reasoning: bool
    prompt_chars: int
    response_chars: int
    wall_s: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feed RealBench prompts directly to a vLLM model and evaluate the generated RTL."
    )
    parser.add_argument("--realbench-root", default="/home/kai/eval_dt/real_bench")
    parser.add_argument("--output-dir", default="runs/realbench_direct_model")
    parser.add_argument("--solution-name", default="direct_model")
    parser.add_argument("--task-level", choices=["module", "system", "both"], default="both")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-problems", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--realbench-verifier",
        choices=["native", "harness"],
        default="native",
        help="native runs RealBench verification Makefiles; harness delegates to run_verify.py.",
    )

    parser.add_argument("--base-url", help="OpenAI-compatible vLLM base URL. Defaults to VLLM_BASE_URL.")
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
    parser.add_argument("--llm-timeout-s", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=80000)
    parser.add_argument("--target-hdl", default="verilog")
    parser.add_argument(
        "--prompt-max-chars",
        type=int,
        default=0,
        help="Optional cap for the RealBench problem text before sending it to the model; 0 means no cap.",
    )

    parser.add_argument("--make-bin", default="make")
    parser.add_argument("--verification-timeout-s", type=int, default=120)
    return parser


def make_client(args: argparse.Namespace) -> VllmClient:
    return VllmClient(
        base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=args.llm_timeout_s,
    )


def direct_prompt(task: RealBenchTask, args: argparse.Namespace) -> str:
    problem = task.prompt
    if args.prompt_max_chars and len(problem) > args.prompt_max_chars:
        problem = problem[: args.prompt_max_chars] + "\n... [truncated] ..."
    constraints = "\n".join(f"- {item}" for item in task_constraints(task))
    provided = template_provided_module_names(task)
    provided_note = ""
    if provided:
        provided_note = (
            "\nThe RealBench verification directory already provides these helper modules. "
            "Do not redeclare them in your output unless the problem explicitly requires a replacement: "
            + ", ".join(provided)
            + "."
        )
    return f"""Generate the requested RealBench RTL implementation directly from the problem statement.

Target HDL: {args.target_hdl}
Required public top module name: {task.top_module}

Constraints:
{constraints}
- Return only synthesizable RTL source code.
- Do not include a testbench, explanations, markdown fences, or analysis text.
{provided_note}

### RealBench Problem
{problem}
"""


def run_direct_generation(task: RealBenchTask, args: argparse.Namespace, client: Any) -> DirectGeneration:
    prompt = direct_prompt(task, args)
    t0 = time.perf_counter()
    message = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    wall_s = time.perf_counter() - t0
    raw_text = str(message.get("content") or "")
    code = clean_generated_code(raw_text)
    return DirectGeneration(
        prompt=prompt,
        raw_text=raw_text,
        code=code,
        finish_reason=message.get("_finish_reason"),
        content_from_reasoning=bool(message.get("_content_from_reasoning")),
        prompt_chars=len(prompt),
        response_chars=len(raw_text),
        wall_s=wall_s,
    )


def clean_generated_code(model_text: str) -> str:
    code = extract_code(model_text).strip()
    if not code:
        return ""
    match = HDL_START_RE.search(code)
    if match and match.start() > 0:
        code = code[match.start() :].strip()
    return code + "\n"


def raw_response_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "raw_responses" / item.task.level / item.task.system / item.task.task / f"sample{item.sample:02d}.txt"


def prompt_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "prompts" / item.task.level / item.task.system / item.task.task / f"sample{item.sample:02d}.txt"


def report_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "reports" / item.task.level / item.task.system / f"{item.task.task}_sample{item.sample:02d}.json"


def run_one_direct(
    item: WorkItem,
    args: argparse.Namespace,
    output_dir: Path,
    catalog: CatalogBundle,
    client: Any,
) -> Dict[str, Any]:
    task = item.task
    code_path = generated_code_path(output_dir, item)
    raw_path = raw_response_path(output_dir, item)
    saved_prompt_path = prompt_path(output_dir, item)
    report = report_path(output_dir, item)
    for path in (code_path, raw_path, saved_prompt_path, report):
        path.parent.mkdir(parents=True, exist_ok=True)

    code = ""
    generation: Optional[DirectGeneration] = None
    generation_error: Optional[str] = None
    reused_existing = False

    if (args.resume or args.evaluate_only) and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        reused_existing = True
    elif args.evaluate_only:
        generation_error = f"missing generated file: {code_path}"
    elif args.dry_run:
        generation_error = "dry run"
    else:
        try:
            generation = run_direct_generation(task, args, client)
            saved_prompt_path.write_text(generation.prompt, encoding="utf-8")
            raw_path.write_text(generation.raw_text, encoding="utf-8")
            code = generation.code
            if code:
                code_path.write_text(code, encoding="utf-8")
            else:
                generation_error = "model response did not contain parsable RTL"
        except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
            generation_error = f"{exc}\n{traceback.format_exc()[-4000:]}"

    eval_result = evaluate_realbench_code(task, code, args) if code else RealBenchEvalResult(
        syntax=0,
        function=0,
        error=generation_error or "generation produced empty code",
    )
    record = build_record(
        item=item,
        catalog=catalog,
        code_path=code_path,
        raw_path=raw_path,
        saved_prompt_path=saved_prompt_path,
        generation=generation,
        generation_error=generation_error,
        reused_existing=reused_existing,
        eval_result=eval_result,
        generated=bool(code),
    )
    report.write_text(dumps_json(record, indent=2), encoding="utf-8")
    return record


def build_record(
    *,
    item: WorkItem,
    catalog: CatalogBundle,
    code_path: Path,
    raw_path: Path,
    saved_prompt_path: Path,
    generation: Optional[DirectGeneration],
    generation_error: Optional[str],
    reused_existing: bool,
    eval_result: RealBenchEvalResult,
    generated: bool,
) -> Dict[str, Any]:
    task = item.task
    return {
        "benchmark": "realbench",
        "pipeline": "direct_model",
        "task_level": task.level,
        "system": task.system,
        "task": task.task,
        "sample": item.sample,
        "top_module": task.top_module,
        "dependencies": task.dependencies,
        "dependency_paths": catalog.dependency_paths,
        "support_paths": catalog.support_paths,
        "missing_dependencies": catalog.missing_dependencies,
        "catalog_path": str(catalog.catalog_path),
        "catalog_doc_count": len(catalog.sources),
        "generated": generated,
        "reused_existing": reused_existing,
        "generation_error": generation_error,
        "generated_code_path": str(code_path) if generated else None,
        "raw_response_path": str(raw_path) if raw_path.exists() else None,
        "prompt_path": str(saved_prompt_path) if saved_prompt_path.exists() else None,
        "finish_reason": generation.finish_reason if generation else None,
        "content_from_reasoning": generation.content_from_reasoning if generation else False,
        "prompt_chars": generation.prompt_chars if generation else None,
        "response_chars": generation.response_chars if generation else None,
        "estimated_prompt_tokens": int(generation.prompt_chars / 4.0) if generation else None,
        "estimated_response_tokens": int(generation.response_chars / 4.0) if generation else None,
        "syntax": eval_result.syntax,
        "function": eval_result.function,
        "passed": eval_result.passed,
        "syntax_info": eval_result.syntax_info,
        "function_info": eval_result.function_info,
        "compile_returncode": eval_result.compile_returncode,
        "run_returncode": eval_result.run_returncode,
        "stdout_tail": eval_result.stdout_tail,
        "stderr_tail": eval_result.stderr_tail,
        "evaluation_error": eval_result.error,
        "wall_s": generation.wall_s if generation else 0.0,
    }


def summarize(records: Sequence[Dict[str, Any]], tasks: Sequence[RealBenchTask], elapsed_s: float) -> Dict[str, Any]:
    total = len(records)
    syntax = sum(1 for record in records if record.get("syntax") == 1)
    function = sum(1 for record in records if record.get("function") == 1)
    passed = sum(1 for record in records if record.get("passed"))
    prompt_tokens = sum(record.get("estimated_prompt_tokens") or 0 for record in records)
    response_tokens = sum(record.get("estimated_response_tokens") or 0 for record in records)
    walls = [float(record.get("wall_s") or 0.0) for record in records]
    return {
        "benchmark": "realbench",
        "pipeline": "direct_model",
        "num_tasks": len(tasks),
        "num_records": total,
        "samples_per_task": max((int(record["sample"]) for record in records), default=0),
        "generated": sum(1 for record in records if record.get("generated")),
        "syntax": syntax,
        "function": function,
        "passed": passed,
        "syntax_rate": safe_rate(syntax, total),
        "function_rate": safe_rate(function, total),
        "pass_rate": safe_rate(passed, total),
        "total_s": elapsed_s,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "estimated_prompt_tokens": prompt_tokens or None,
        "estimated_response_tokens": response_tokens or None,
        "estimated_total_tokens": (prompt_tokens + response_tokens) or None,
    }


def dry_run_record(item: WorkItem, catalog: CatalogBundle, output_dir: Path) -> Dict[str, Any]:
    return build_record(
        item=item,
        catalog=catalog,
        code_path=generated_code_path(output_dir, item),
        raw_path=raw_response_path(output_dir, item),
        saved_prompt_path=prompt_path(output_dir, item),
        generation=None,
        generation_error="dry run",
        reused_existing=False,
        eval_result=RealBenchEvalResult(0, 0, error="dry run"),
        generated=False,
    )


def run_realbench_direct(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tasks = discover_tasks(args)
    print(f"[realbench-direct] discovered {len(tasks)} task(s)")

    catalogs: Dict[str, CatalogBundle] = {}
    for task in tasks:
        bundle = build_task_catalog(task, output_dir)
        catalogs[task.task_id] = bundle
        print(
            f"[realbench-direct] catalog {task.task_id}: docs={len(bundle.sources)} "
            f"deps={len(bundle.dependency_paths)} missing={bundle.missing_dependencies}"
        )

    items = work_items(tasks, args.samples)
    if args.dry_run:
        records = [dry_run_record(item, catalogs[item.task.task_id], output_dir) for item in items]
    else:
        client = make_client(args)
        records = []
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
            futures = [
                executor.submit(run_one_direct, item, args, output_dir, catalogs[item.task.task_id], client)
                for item in items
            ]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                status = "PASS" if record["passed"] else "FAIL"
                print(
                    f"[realbench-direct] {status} {record['task_level']}/{record['system']}/{record['task']} "
                    f"sample {int(record['sample']):02d} syntax={record['syntax']} function={record['function']}"
                )

    elapsed_s = time.perf_counter() - start
    write_records(output_dir / "records.jsonl", records)
    write_solution_jsonl(output_dir, args.solution_name, records)
    summary = summarize(sorted(records, key=record_sort_key), tasks, elapsed_s)
    if args.dry_run:
        summary["dry_run"] = True
    (output_dir / "summary.json").write_text(dumps_json(summary, indent=2), encoding="utf-8")
    print(f"[realbench-direct] wrote results under {output_dir}")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_realbench_direct(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
