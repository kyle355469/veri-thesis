#!/usr/bin/env python3
# runnable
# python3 scripts/run_cvdp_eval.py \
#   --dataset /home/kai/eval_dt/cvdp_benchmark/example_dataset/cvdp_v1.1.0_example_nonagentic_code_generation_no_commercial.jsonl \
#   --pipeline fixed-pipe \
#   --index indexes/rtl_hash \
#   --cache-mode none \
#   --code-structure-index indexes/rtl_datapath_hash \
#   --cid 02 --cid 03 --cid 09 \
#   --samples 5 \
#   --output-dir runs/cvdp_eval

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.config import CacheConfig, FixedPipeConfig, RuntimeConfig, ToolCallingConfig
from rag_rtl.embeddings import make_embedder
from rag_rtl.json_utils import dumps_json
from rag_rtl.llm import VllmClient, extract_code
from rag_rtl.pipeline import FixedPipeRtlPipeline, RagRtlPipeline
from rag_rtl.types import PipelineResponse, RtlTask, VerificationReport
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

MODULE_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")
TOPLEVEL_RE = re.compile(r"(?m)^\s*TOPLEVEL\s*=\s*([A-Za-z_][A-Za-z0-9_$]*)\s*$")
VERILOG_SOURCE_RE = re.compile(r"/code/((?:rtl|verif|src)/[^\s]+?\.(?:sv|v|svh|vh))")
PASS_AT_KS = (1, 3, 5)
DEFAULT_CIDS = ("cid002", "cid003", "cid009")
CODE_COMPREHENSION_CIDS = {"cid006", "cid008", "cid009", "cid010"}
EVAL_LOCK = threading.Lock()


@dataclass(frozen=True)
class CvdpProblem:
    problem_id: str
    categories: Tuple[str, ...]
    dataset_path: Path
    record: Dict[str, Any]
    prompt: str
    context: Dict[str, str]
    expected_files: Tuple[str, ...]
    top_module: Optional[str]

    @property
    def primary_cid(self) -> str:
        for category in self.categories:
            if category.startswith("cid"):
                return category
        return self.categories[0] if self.categories else "unknown"

    @property
    def difficulty(self) -> str:
        for category in self.categories:
            if not category.startswith("cid"):
                return category
        return "unknown"

    @property
    def is_subjective(self) -> bool:
        return self.primary_cid in CODE_COMPREHENSION_CIDS


@dataclass(frozen=True)
class WorkItem:
    problem: CvdpProblem
    sample: int


@dataclass
class CvdpEvalResult:
    passed: bool
    passfail: str
    errors: Optional[int]
    tests: List[Dict[str, Any]]
    cvdp_work_dir: Optional[str] = None
    raw_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    evaluation_s: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CVDP cid002/cid003/cid009 records through the veri-thesis RTL generator, "
            "then evaluate objective records with the CVDP harness."
        )
    )
    parser.add_argument("--cvdp-root", default="/home/kai/eval_dt/cvdp_benchmark")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="CVDP JSONL dataset file; repeatable. If omitted, example_dataset/*.jsonl is scanned.",
    )
    parser.add_argument("--output-dir", default="runs/cvdp_eval")
    parser.add_argument("--pipeline", choices=["rag", "fixed-pipe"], default="rag")
    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--code-structure-index", default="indexes/rtl_datapath_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--cid",
        action="append",
        default=[],
        help="CVDP category id to include, e.g. 02, 003, or cid009. Defaults to 02/03/09.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Case-insensitive substring filter for problem id, category, or dataset path; repeatable.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing generated answer files")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip generation and evaluate existing answers")
    parser.add_argument("--dry-run", action="store_true", help="Only discover CVDP records and print the count")
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--structure-retrieve-k", type=int, default=8)
    parser.add_argument("--structure-context-k", type=int, default=4)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--second-edition-repair-attempts", type=int, default=1)
    parser.add_argument("--cache", default="data/history_cache.json")
    parser.add_argument("--monitor", default="runs/cvdp_eval_monitor.jsonl")
    parser.add_argument("--failed-log", default="runs/cvdp_eval_failed_attempts.jsonl")
    parser.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    parser.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    parser.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    parser.add_argument("--generation-temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--serving-url",
        "--base-url",
        dest="serving_url",
        help="OpenAI-compatible serving base URL. Overrides VLLM_BASE_URL.",
    )
    parser.add_argument("--verbose-generation", action="store_true")
    parser.add_argument("--enable-tool-calling", action="store_true")
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--network-name", help="Optional shared Docker network passed to the CVDP harness")
    parser.add_argument("--keep-cvdp-work", action="store_true", help="Keep synthetic CVDP JSONL and harness workdirs")
    return parser


def normalize_cid(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return value
    if value.startswith("cid"):
        suffix = value[3:]
    else:
        suffix = value
    return f"cid{int(suffix):03d}" if suffix.isdigit() else value


def dataset_paths(args: argparse.Namespace) -> List[Path]:
    if args.dataset:
        return [Path(item) for item in args.dataset]
    cvdp_root = Path(args.cvdp_root)
    candidates = sorted((cvdp_root / "dataset").glob("*.jsonl"))
    if candidates:
        return candidates
    examples = sorted((cvdp_root / "example_dataset").glob("*.jsonl"))
    preferred_examples = [path for path in examples if "_with_solutions" not in path.name]
    return preferred_examples or examples


def discover_problems(args: argparse.Namespace) -> List[CvdpProblem]:
    cids = {normalize_cid(item) for item in (args.cid or DEFAULT_CIDS)}
    filters = [item.lower() for item in args.include if item.strip()]
    problems: List[CvdpProblem] = []
    seen: set[Tuple[Path, str]] = set()

    for dataset_path in dataset_paths(args):
        if not dataset_path.exists():
            raise FileNotFoundError(f"CVDP dataset file not found: {dataset_path}")
        with dataset_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                categories = tuple(str(item).lower() for item in record.get("categories", ()))
                record_cids = {normalize_cid(item) for item in categories if str(item).lower().startswith("cid")}
                if cids and not (record_cids & cids):
                    continue
                problem_id = str(record.get("id") or f"{dataset_path.stem}_{line_no}")
                key = (dataset_path.resolve(), problem_id)
                if key in seen:
                    continue
                haystack = f"{problem_id} {' '.join(categories)} {dataset_path}".lower()
                if filters and not any(item in haystack for item in filters):
                    continue
                prompt, context = extract_prompt_and_context(record)
                expected_files = tuple(infer_expected_files(record))
                top_module = infer_top_module(record)
                problems.append(
                    CvdpProblem(
                        problem_id=problem_id,
                        categories=categories,
                        dataset_path=dataset_path,
                        record=record,
                        prompt=prompt,
                        context=context,
                        expected_files=expected_files,
                        top_module=top_module,
                    )
                )
                seen.add(key)
                if args.limit is not None and len(problems) >= args.limit:
                    return problems
    return problems


def extract_prompt_and_context(record: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    if isinstance(record.get("input"), dict):
        prompt = str(record["input"].get("prompt") or record["input"].get("text") or "")
        context = record["input"].get("context") or {}
    else:
        prompt = str(record.get("prompt") or "")
        context = record.get("context") or {}
    return prompt.strip(), {str(key): str(value) for key, value in context.items()}


def infer_expected_files(record: Dict[str, Any]) -> List[str]:
    output = record.get("output") if isinstance(record.get("output"), dict) else {}
    output_context = output.get("context") if isinstance(output.get("context"), dict) else {}
    if output_context:
        return sorted(str(path) for path in output_context.keys())
    patch = record.get("patch") if isinstance(record.get("patch"), dict) else {}
    if patch:
        return sorted(str(path) for path in patch.keys())

    harness = record.get("harness") or {}
    harness_files = harness.get("files") if isinstance(harness, dict) and isinstance(harness.get("files"), dict) else harness
    env_text = str(harness_files.get("src/.env", "")) if isinstance(harness_files, dict) else ""
    context = record.get("context") or (record.get("input", {}) if isinstance(record.get("input"), dict) else {}).get("context") or {}
    candidates = []
    for match in VERILOG_SOURCE_RE.findall(env_text):
        if match.startswith("rtl/") and match not in context:
            candidates.append(match)
    return sorted(dict.fromkeys(candidates))


def infer_top_module(record: Dict[str, Any]) -> Optional[str]:
    harness = record.get("harness") or {}
    harness_files = harness.get("files") if isinstance(harness, dict) and isinstance(harness.get("files"), dict) else harness
    if isinstance(harness_files, dict):
        env_text = str(harness_files.get("src/.env", ""))
        match = TOPLEVEL_RE.search(env_text)
        if match:
            return match.group(1)
    prompt, context = extract_prompt_and_context(record)
    text = prompt + "\n" + "\n".join(context.values())
    match = re.search(r"\bmodule\s+(?:named\s+)?`?([A-Za-z_][A-Za-z0-9_$]*)`?", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def build_pipeline(args: argparse.Namespace) -> Any:
    embedder = make_embedder(args.embedder)
    llm_client = build_llm_client(args)
    verifier = RtlVerifier()
    cache_config = CacheConfig(
        path=args.cache,
        mode=args.cache_mode,
        reuse_threshold=args.cache_reuse_threshold,
        evidence_threshold=args.cache_evidence_threshold,
    )
    runtime_config = RuntimeConfig(
        monitor_path=args.monitor,
        failed_log_path=args.failed_log,
        verbose_generation=args.verbose_generation,
        generation_temperature=args.generation_temperature,
        max_tokens=args.max_tokens,
    )
    tool_config = ToolCallingConfig(
        enabled=args.enable_tool_calling,
        choice=args.tool_choice,
        max_rounds=args.max_tool_rounds,
    )
    spec_store = VectorStore.load(args.index)

    if args.pipeline == "fixed-pipe":
        structure_store = VectorStore.load(args.code_structure_index)
        return FixedPipeRtlPipeline(
            spec_store=spec_store,
            code_structure_store=structure_store,
            embedder=embedder,
            llm_client=llm_client,
            verifier=verifier,
            cache_config=cache_config,
            runtime_config=runtime_config,
            tool_config=tool_config,
            fixed_pipe_config=FixedPipeConfig(second_edition_repair_attempts=args.second_edition_repair_attempts),
        )

    return RagRtlPipeline(
        store=spec_store,
        embedder=embedder,
        llm_client=llm_client,
        verifier=verifier,
        cache_config=cache_config,
        runtime_config=runtime_config,
        tool_config=tool_config,
    )


def build_llm_client(args: argparse.Namespace) -> VllmClient:
    client = VllmClient.from_env()
    if args.serving_url:
        client.base_url = args.serving_url
    return client


def build_task(problem: CvdpProblem, args: argparse.Namespace) -> RtlTask:
    context_text = format_context(problem.context)
    expected = ", ".join(problem.expected_files) if problem.expected_files else "the file expected by the harness"
    prompt = problem.prompt
    if context_text:
        prompt += f"\n\nExisting project context:\n{context_text}"
    constraints = [
        "Return the complete replacement content for the requested hardware implementation.",
        f"The CVDP harness expects output file path(s): {expected}.",
        "Do not include a testbench, markdown fences outside the HDL block, or explanatory text.",
    ]
    if problem.top_module:
        constraints.append(f"Use the top-level module name {problem.top_module}.")
    return RtlTask(
        prompt=prompt,
        target_hdl="verilog",
        constraints=constraints,
        max_repair_attempts=args.max_repair_attempts,
        top_module=problem.top_module,
        prompt_profile=getattr(args, "prompt_profile", "rag"),
    )


def format_context(context: Dict[str, str]) -> str:
    parts = []
    for path, content in sorted(context.items()):
        parts.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(parts)


def output_problem_dir(output_dir: Path, problem: CvdpProblem) -> Path:
    return output_dir / problem.primary_cid / safe_path_name(problem.problem_id)


def safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "problem"


def sample_stem(problem: CvdpProblem, sample: int) -> str:
    return f"{safe_path_name(problem.problem_id)}_sample{sample:02d}"


def generated_answer_path(output_dir: Path, item: WorkItem) -> Path:
    return output_problem_dir(output_dir, item.problem) / f"{sample_stem(item.problem, item.sample)}_answers.json"


def generation_log_path(output_dir: Path, item: WorkItem) -> Path:
    return output_problem_dir(output_dir, item.problem) / f"{sample_stem(item.problem, item.sample)}-generate.log"


def cvdp_eval_log_path(output_dir: Path, item: WorkItem) -> Path:
    return output_problem_dir(output_dir, item.problem) / f"{sample_stem(item.problem, item.sample)}-cvdp.log"


def run_generation(
    item: WorkItem,
    pipeline: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Optional[PipelineResponse], Optional[str]]:
    task = build_task(item.problem, args)
    try:
        if args.pipeline == "fixed-pipe":
            response = pipeline.run(
                task,
                retrieve_k=args.retrieve_k,
                context_k=args.context_k,
                structure_retrieve_k=args.structure_retrieve_k,
                structure_context_k=args.structure_context_k,
            )
        else:
            response = pipeline.run(task, retrieve_k=args.retrieve_k, context_k=args.context_k)
    except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
        return {}, None, str(exc)

    answers = normalize_generated_answers(response.rtl, item.problem)
    return answers, response, None


def normalize_generated_answers(raw_text: str, problem: CvdpProblem) -> Dict[str, str]:
    expected_files = list(problem.expected_files)
    if problem.is_subjective:
        return {"subjective.txt": raw_text.strip()}
    if not expected_files:
        expected_files = [f"rtl/{problem.top_module or problem.problem_id}.sv"]

    if len(expected_files) > 1:
        parsed = parse_json_answers(raw_text, expected_files)
        if parsed:
            return parsed

    code = extract_code(raw_text).strip()
    if problem.top_module:
        code = ensure_top_module_name(code, problem.top_module)
    return {expected_files[0]: code}


def parse_json_answers(raw_text: str, expected_files: Sequence[str]) -> Dict[str, str]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = extract_code(text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    result: Dict[str, str] = {}
    if isinstance(data, dict) and isinstance(data.get("code"), list):
        for item in data["code"]:
            if isinstance(item, dict):
                for path, content in item.items():
                    if path in expected_files:
                        result[str(path)] = str(content)
    elif isinstance(data, dict):
        for path in expected_files:
            if path in data:
                result[path] = str(data[path])
    return result


def ensure_top_module_name(code: str, top_module: str) -> str:
    module_names = MODULE_RE.findall(code)
    if not module_names or top_module in module_names:
        return code
    first_name = module_names[0]
    pattern = re.compile(rf"(?m)^(\s*module\s+){re.escape(first_name)}\b")
    return pattern.sub(lambda match: f"{match.group(1)}{top_module}", code, count=1)


def evaluate_cvdp(
    item: WorkItem,
    answers: Dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> CvdpEvalResult:
    if item.problem.is_subjective:
        return evaluate_subjective(item, answers, args, output_dir)
    return evaluate_objective(item, answers, args, output_dir)


def evaluate_objective(
    item: WorkItem,
    answers: Dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> CvdpEvalResult:
    start = time.perf_counter()
    cvdp_root = Path(args.cvdp_root).resolve()
    eval_dir = output_problem_dir(output_dir, item.problem) / "_cvdp_work" / f"sample{item.sample:02d}"
    synthetic_path = eval_dir / "dataset.jsonl"
    prefix = eval_dir / "work"
    eval_dir.mkdir(parents=True, exist_ok=True)

    synthetic_record = build_synthetic_copilot_record(item.problem, answers)
    synthetic_path.write_text(dumps_json(synthetic_record) + "\n", encoding="utf-8")

    try:
        result = run_cvdp_processor(cvdp_root, synthetic_path, prefix, item.problem.problem_id, args)
    except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
        return CvdpEvalResult(
            passed=False,
            passfail="E",
            errors=None,
            tests=[],
            cvdp_work_dir=str(prefix),
            error=str(exc),
            evaluation_s=time.perf_counter() - start,
        )

    tests = result.get("tests") if isinstance(result, dict) else []
    errors = int(result.get("errors") or 0) if isinstance(result, dict) else 1
    passed = errors == 0 and all(int(test.get("result") or 0) == 0 for test in tests)
    passfail = "." if passed else classify_cvdp_failure(tests, errors)
    if not args.keep_cvdp_work:
        cleanup_cvdp_work(eval_dir)
    return CvdpEvalResult(
        passed=passed,
        passfail=passfail,
        errors=errors,
        tests=tests,
        cvdp_work_dir=str(prefix),
        raw_result=result,
        evaluation_s=time.perf_counter() - start,
    )


def build_synthetic_copilot_record(problem: CvdpProblem, answers: Dict[str, str]) -> Dict[str, Any]:
    record = copy.deepcopy(problem.record)
    prompt, context = extract_prompt_and_context(problem.record)
    record["id"] = problem.problem_id
    record["categories"] = list(problem.categories)
    record["input"] = {"prompt": prompt, "context": context}
    record["output"] = {
        "response": "",
        "context": {path: answers.get(path, "") for path in problem.expected_files},
    }
    if not record["output"]["context"]:
        record["output"]["context"] = dict(answers)
    return record


def run_cvdp_processor(
    cvdp_root: Path,
    synthetic_path: Path,
    prefix: Path,
    problem_id: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not cvdp_root.exists():
        raise FileNotFoundError(f"CVDP root not found: {cvdp_root}")
    with EVAL_LOCK:
        original_cwd = Path.cwd()
        old_sys_path = list(sys.path)
        try:
            os.chdir(cvdp_root)
            if str(cvdp_root) not in sys.path:
                sys.path.insert(0, str(cvdp_root))
            from src.dataset_processor import CopilotProcessor  # type: ignore

            processor = CopilotProcessor(
                filename=str(synthetic_path),
                golden=True,
                threads=1,
                prefix=str(prefix),
                network_name=args.network_name,
                manage_network=args.network_name is None,
            )
            processor.process_json()
            prepared_id, obj, repo = processor.prepare(problem_id, None)
            return processor.run(id=prepared_id, obj=obj, repo=repo, model=None)
        finally:
            os.chdir(original_cwd)
            sys.path[:] = old_sys_path


def evaluate_subjective(
    item: WorkItem,
    answers: Dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> CvdpEvalResult:
    start = time.perf_counter()
    response = answers.get("subjective.txt") or next(iter(answers.values()), "")
    reference = ""
    output = item.problem.record.get("output")
    if isinstance(output, dict):
        reference = str(output.get("response") or "")
    if not reference:
        return CvdpEvalResult(
            passed=False,
            passfail="N",
            errors=1,
            tests=[{"result": 1, "log": None, "error_msg": "no reference response for subjective cid", "execution": 0.0}],
            error="no reference response for subjective cid",
            evaluation_s=time.perf_counter() - start,
        )

    log_path = cvdp_eval_log_path(output_dir, item)
    try:
        rouge, bleu = score_subjective(Path(args.cvdp_root).resolve(), response, reference)
        passed = rouge > 0.40 and bleu > 0.40
        test = {
            "result": 0 if passed else 1,
            "log": str(log_path),
            "error_msg": None if passed else "subjective score below threshold",
            "execution": time.perf_counter() - start,
            "rouge": rouge,
            "bleu": bleu,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(test, indent=2), encoding="utf-8")
        return CvdpEvalResult(
            passed=passed,
            passfail="." if passed else "R",
            errors=0 if passed else 1,
            tests=[test],
            raw_result={"tests": [test], "errors": 0 if passed else 1},
            evaluation_s=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
        return CvdpEvalResult(
            passed=False,
            passfail="E",
            errors=1,
            tests=[],
            error=str(exc),
            evaluation_s=time.perf_counter() - start,
        )


def score_subjective(cvdp_root: Path, response: str, reference: str) -> Tuple[float, float]:
    with EVAL_LOCK:
        old_sys_path = list(sys.path)
        try:
            if str(cvdp_root) not in sys.path:
                sys.path.insert(0, str(cvdp_root))
            from src import subjective  # type: ignore

            rouge = float(subjective.calculate_ROUGE(response, reference, 2))
            bleu = float(subjective.calculate_BLEU(response, reference, 2))
            return rouge, bleu
        finally:
            sys.path[:] = old_sys_path


def cleanup_cvdp_work(eval_dir: Path) -> None:
    for path in eval_dir.glob("work/*/harness/*/rundir"):
        shutil.rmtree(path, ignore_errors=True)


def classify_cvdp_failure(tests: Sequence[Dict[str, Any]], errors: Optional[int]) -> str:
    text = "\n".join(str(test.get("error_msg") or "") for test in tests).lower()
    if "syntax" in text or "compile" in text or "build" in text:
        return "C"
    if "timeout" in text:
        return "T"
    if errors:
        return "R"
    return "?"


def write_generation_log(
    path: Path,
    item: WorkItem,
    response: Optional[PipelineResponse],
    error: Optional[str],
    reused: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"problem = {item.problem.problem_id}",
        f"categories = {json.dumps(item.problem.categories)}",
        f"sample = {item.sample:02d}",
        f"top_module = {item.problem.top_module or ''}",
        f"dataset = {item.problem.dataset_path}",
        f"expected_files = {json.dumps(item.problem.expected_files)}",
        f"reused_existing = {str(reused).lower()}",
    ]
    if error:
        lines.append(f"error = {error}")
    if response:
        lines.extend(
            [
                f"syntax_passed = {response.verification.syntax_passed}",
                f"lint_passed = {response.verification.lint_passed}",
                f"verification_passed = {response.verification.passed}",
                f"cache_source = {response.cache_source}",
                f"repair_attempts = {response.repair_attempts}",
                f"retrieved_doc_ids = {json.dumps(response.retrieved_doc_ids)}",
                f"timings = {dumps_json(response.timings)}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cvdp_log(path: Path, result: CvdpEvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "passed": result.passed,
                "passfail": result.passfail,
                "errors": result.errors,
                "tests": result.tests,
                "cvdp_work_dir": result.cvdp_work_dir,
                "error": result.error,
                "evaluation_s": result.evaluation_s,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def response_metadata(response: Optional[PipelineResponse]) -> Dict[str, Any]:
    if not response:
        return {
            "rag_generation_passed": False,
            "syntax_passed": False,
            "lint_passed": False,
            "repair_attempts": None,
            "cache_source": None,
            "retrieved_doc_ids": [],
            "timings": {},
        }
    verification = response.verification
    return {
        "rag_generation_passed": verification.passed,
        "syntax_passed": verification.syntax_passed,
        "lint_passed": verification.lint_passed,
        "repair_attempts": response.repair_attempts,
        "cache_source": response.cache_source,
        "retrieved_doc_ids": response.retrieved_doc_ids,
        "timings": response.timings,
    }


def verification_diagnostics(report: Optional[VerificationReport]) -> List[Dict[str, Any]]:
    if not report:
        return []
    return [
        {
            "tool": diagnostic.tool,
            "passed": diagnostic.passed,
            "returncode": diagnostic.returncode,
            "missing": diagnostic.missing,
            "stdout_tail": diagnostic.stdout[-2000:],
            "stderr_tail": diagnostic.stderr[-2000:],
        }
        for diagnostic in report.diagnostics
    ]


def run_one(item: WorkItem, pipeline: Any, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    out_dir = output_problem_dir(output_dir, item.problem)
    out_dir.mkdir(parents=True, exist_ok=True)
    answer_path = generated_answer_path(output_dir, item)
    gen_log = generation_log_path(output_dir, item)
    eval_log = cvdp_eval_log_path(output_dir, item)

    response: Optional[PipelineResponse] = None
    generation_error: Optional[str] = None
    reused_existing = False

    if (args.resume or args.evaluate_only) and answer_path.exists():
        answers = json.loads(answer_path.read_text(encoding="utf-8"))
        reused_existing = True
    elif args.evaluate_only:
        answers = {}
        generation_error = f"missing generated answer file: {answer_path}"
    else:
        answers, response, generation_error = run_generation(item, pipeline, args)
        if answers:
            answer_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")

    write_generation_log(gen_log, item, response, generation_error, reused_existing)

    if not answers:
        eval_result = CvdpEvalResult(
            passed=False,
            passfail="G",
            errors=1,
            tests=[],
            error=generation_error or "generation produced no answers",
        )
    else:
        eval_result = evaluate_cvdp(item, answers, args, output_dir)
    write_cvdp_log(eval_log, eval_result)

    return {
        "problem": item.problem.problem_id,
        "cid": item.problem.primary_cid,
        "difficulty": item.problem.difficulty,
        "categories": list(item.problem.categories),
        "sample": item.sample,
        "dataset_path": str(item.problem.dataset_path),
        "top_module": item.problem.top_module,
        "expected_files": list(item.problem.expected_files),
        "generated_answer_path": str(answer_path),
        "generation_log_path": str(gen_log),
        "cvdp_eval_log_path": str(eval_log),
        "generated": bool(answers),
        "generation_error": generation_error,
        "reused_existing": reused_existing,
        **response_metadata(response),
        "verification_diagnostics": verification_diagnostics(response.verification if response else None),
        "passed": eval_result.passed,
        "passfail": eval_result.passfail,
        "cvdp_errors": eval_result.errors,
        "cvdp_tests": eval_result.tests,
        "cvdp_work_dir": eval_result.cvdp_work_dir,
        "evaluation_s": eval_result.evaluation_s,
        "evaluation_error": eval_result.error,
    }


def summarize(records: Sequence[Dict[str, Any]], args: argparse.Namespace, output_dir: Path, elapsed_s: float) -> Dict[str, Any]:
    count = len(records)
    denom = max(count, 1)
    passfail_counts: Dict[str, int] = {}
    for record in records:
        key = str(record.get("passfail") or "?")
        passfail_counts[key] = passfail_counts.get(key, 0) + 1
    pass_at_rates, pass_at_denominators = compute_pass_at(records, PASS_AT_KS)
    syntax_success_by_problem = {
        (record["cid"], record["problem"]) for record in records if record.get("syntax_passed")
    }
    func_success_by_problem = {
        (record["cid"], record["problem"]) for record in records if record.get("passed")
    }
    return {
        "cvdp_root": str(Path(args.cvdp_root)),
        "datasets": [str(path) for path in dataset_paths(args)],
        "cids": [normalize_cid(item) for item in (args.cid or DEFAULT_CIDS)],
        "pipeline": args.pipeline,
        "output_dir": str(output_dir),
        "num_records": count,
        "num_problems": len({(record["cid"], record["problem"]) for record in records}),
        "samples_per_problem": args.samples,
        "generated": sum(1 for record in records if record.get("generated")),
        "rag_generation_passed": sum(1 for record in records if record.get("rag_generation_passed")),
        "syntax_passed": sum(1 for record in records if record.get("syntax_passed")),
        "lint_passed": sum(1 for record in records if record.get("lint_passed")),
        "passed": sum(1 for record in records if record.get("passed")),
        "accuracy": sum(1 for record in records if record.get("passed")) / denom,
        "pass@1": pass_at_rates[1],
        "pass@3": pass_at_rates[3],
        "pass@5": pass_at_rates[5],
        "pass_at_denominators": {str(k): pass_at_denominators[k] for k in PASS_AT_KS},
        "syntax_success_problem_count": len(syntax_success_by_problem),
        "function_success_problem_count": len(func_success_by_problem),
        "passfail_counts": dict(sorted(passfail_counts.items())),
        "total_s": elapsed_s,
        "records": list(records),
    }


def group_records_by_problem(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("cid") or ""), str(record["problem"]))
        grouped.setdefault(key, []).append(record)
    for problem_records in grouped.values():
        problem_records.sort(key=lambda item: int(item.get("sample") or 0))
    return grouped


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> Optional[float]:
    if num_samples < k:
        return None
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    probability_all_wrong = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        probability_all_wrong *= 1.0 - (k / value)
    return 1.0 - probability_all_wrong


def compute_pass_at(records: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Tuple[Dict[int, Optional[float]], Dict[int, int]]:
    grouped = group_records_by_problem(records)
    rates: Dict[int, Optional[float]] = {}
    denominators: Dict[int, int] = {}
    for k in ks:
        estimates = [
            estimate
            for problem_records in grouped.values()
            if (estimate := estimate_pass_at_k(
                len(problem_records),
                sum(1 for record in problem_records if record.get("passed")),
                k,
            ))
            is not None
        ]
        denominators[k] = len(estimates)
        rates[k] = (sum(estimates) / len(estimates)) if estimates else None
    return rates, denominators


def format_summary_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def repair_attempts_label(value: Any) -> str:
    return "n/a" if value is None else str(value)


def write_csv_summary(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "problem",
        "cid",
        "difficulty",
        "sample",
        "top_module",
        "passed",
        "passfail",
        "cvdp_errors",
        "generated",
        "rag_generation_passed",
        "syntax_passed",
        "lint_passed",
        "repair_attempts",
        "cache_source",
        "generated_answer_path",
        "cvdp_eval_log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def iter_work_items(problems: Sequence[CvdpProblem], samples: int) -> Iterable[WorkItem]:
    for problem in problems:
        for sample in range(1, samples + 1):
            yield WorkItem(problem=problem, sample=sample)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = discover_problems(args)
    work_items = list(iter_work_items(problems, args.samples))

    if args.dry_run:
        print(f"discovered {len(problems)} CVDP problems and {len(work_items)} work items in {output_dir}")
        for problem in problems[:20]:
            files = ", ".join(problem.expected_files) or "n/a"
            print(f"{problem.problem_id}: cid={problem.primary_cid} top={problem.top_module or 'n/a'} files={files}")
        return

    pipeline = None if args.evaluate_only else build_pipeline(args)
    records: List[Dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records_lock = threading.Lock()
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = [executor.submit(run_one, item, pipeline, args, output_dir) for item in work_items]
        for future in as_completed(futures):
            record = future.result()
            with records_lock:
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(dumps_json(record) + "\n")
            print(
                f"completed {record['problem']} {record['cid']} sample {record['sample']:02d}: "
                f"{record['passfail']} passed={record['passed']} "
                f"repairs={repair_attempts_label(record.get('repair_attempts'))}"
            )

    records.sort(key=lambda item: (item["cid"], item["problem"], item["sample"]))
    summary = summarize(records, args, output_dir, time.perf_counter() - start)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv_summary(output_dir / "summary.csv", records)
    print(
        f"accuracy={summary['accuracy']:.4f} "
        f"pass@1={format_summary_metric(summary['pass@1'])} "
        f"pass@3={format_summary_metric(summary['pass@3'])} "
        f"pass@5={format_summary_metric(summary['pass@5'])} "
        f"passed={summary['passed']}/{summary['num_records']} "
        f"summary={output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
