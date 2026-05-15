from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import CacheConfig, FixedPipeConfig, RuntimeConfig, ToolCallingConfig
from .dataset import iter_jsonl_documents
from .datapath import build_datapath_vector_db
from .embeddings import encode_texts, make_embedder
from .evaluation import run_evaluation
from .json_utils import json_default
from .pipeline import FixedPipeRtlPipeline, RagRtlPipeline
from .reporting import build_latest_report
from .stg_eval import run_stg_dataset_evaluation
from .types import RtlTask
from .vector_store import VectorStore, build_vector_store
from .verifier import RtlVerifier


def cmd_index(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    documents = list(iter_jsonl_documents(args.corpus, limit=args.limit))
    texts = [document.retrieval_text for document in documents]
    vectors = encode_texts(embedder, texts, jobs=args.jobs)
    store = build_vector_store(documents, vectors)
    store.save(args.output)
    print(f"Indexed {len(documents)} documents into {args.output}")


def cmd_datapath_index(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    documents = list(iter_jsonl_documents(args.corpus, limit=args.limit))
    stats = build_datapath_vector_db(
        documents=documents,
        embedder=embedder,
        output=args.output,
        yosys_bin=args.yosys_bin,
        timeout_s=args.timeout_s,
        jobs=args.jobs,
    )
    print(
        "Built datapath graph VectorDB "
        f"from {stats.source_documents} documents into {args.output}: "
        f"{stats.graphs} graphs, {stats.skipped} skipped"
    )
    if stats.skipped:
        print(f"Yosys failures/skips were recorded in {stats.failures_path}")


def cmd_generate(args: argparse.Namespace) -> None:
    pipeline = build_generate_pipeline(args)
    task = build_task(args)
    response = pipeline.run(task, retrieve_k=args.retrieve_k, context_k=args.context_k)
    print(response.rtl)
    write_report(response, args.json_report)


def cmd_fixed_pipe(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    spec_store = VectorStore.load(args.spec_index)
    code_structure_store = VectorStore.load(args.code_structure_index)
    pipeline = FixedPipeRtlPipeline(
        spec_store=spec_store,
        code_structure_store=code_structure_store,
        embedder=embedder,
        verifier=build_verifier(args),
        cache_config=build_cache_config(args),
        runtime_config=build_runtime_config(args),
        tool_config=build_tool_config(args),
        fixed_pipe_config=build_fixed_pipe_config(args),
    )
    task = build_task(args)
    response = pipeline.run(
        task,
        retrieve_k=args.retrieve_k,
        context_k=args.context_k,
        structure_retrieve_k=args.structure_retrieve_k,
        structure_context_k=args.structure_context_k,
    )
    print(response.rtl)
    write_report(response, args.json_report)


def cmd_evaluate(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    store = VectorStore.load(args.index)
    summary = run_evaluation(
        tasks_path=args.tasks,
        store=store,
        embedder=embedder,
        mode=args.mode,
        output_path=args.output,
        cache_mode=args.cache_mode,
        cache_reuse_threshold=args.cache_reuse_threshold,
        cache_evidence_threshold=args.cache_evidence_threshold,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


def cmd_stg_evaluate(args: argparse.Namespace) -> None:
    pipeline = build_generate_pipeline(args)
    extra_stg_args = args.stg_arg or []
    result_code_dir = args.save_result_code_dir or f"{Path(args.output).with_suffix('')}_codes"
    summary = run_stg_dataset_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        pipeline=pipeline,
        stg_bin=args.stg_bin,
        target_hdl=args.target_hdl,
        default_design_type=args.type,
        limit=args.limit,
        timeout_s=args.timeout_s,
        spec_field=args.spec_field,
        golden_field=args.golden_field,
        save_result_code_dir=result_code_dir,
        save_passed_dir=args.save_passed_dir,
        extra_stg_args=extra_stg_args,
        retrieve_k=args.retrieve_k,
        context_k=args.context_k,
        max_repair_attempts=args.max_repair_attempts,
        module_signature=args.module_signature,
        constraints=args.constraint,
        top_module=args.top_module,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


def build_generate_pipeline(args: argparse.Namespace) -> RagRtlPipeline:
    embedder = make_embedder(args.embedder)
    store = VectorStore.load(args.index)
    return RagRtlPipeline(
        store=store,
        embedder=embedder,
        verifier=build_verifier(args),
        cache_config=build_cache_config(args),
        runtime_config=build_runtime_config(args),
        tool_config=build_tool_config(args),
    )


def build_task(args: argparse.Namespace) -> RtlTask:
    return RtlTask(
        prompt=read_prompt(args),
        target_hdl=args.target_hdl,
        module_signature=args.module_signature,
        constraints=args.constraint,
        max_repair_attempts=args.max_repair_attempts,
        top_module=args.top_module,
    )


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise ValueError("Provide --prompt or --prompt-file.")


def build_verifier(args: argparse.Namespace) -> RtlVerifier:
    return RtlVerifier(
        testbench_path=args.testbench,
        test_command=args.test_command,
    )


def build_cache_config(args: argparse.Namespace) -> CacheConfig:
    return CacheConfig(
        path=args.cache,
        mode=args.cache_mode,
        reuse_threshold=args.cache_reuse_threshold,
        evidence_threshold=args.cache_evidence_threshold,
    )


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        monitor_path=args.monitor,
        failed_log_path=args.failed_log,
        verbose_generation=args.verbose_generation,
        generation_temperature=getattr(args, "generation_temperature", 0.4),
        max_tokens=getattr(args, "max_tokens", 2048),
    )


def build_tool_config(args: argparse.Namespace) -> ToolCallingConfig:
    return ToolCallingConfig(
        enabled=args.enable_tool_calling,
        choice=args.tool_choice,
        max_rounds=args.max_tool_rounds,
    )


def build_fixed_pipe_config(args: argparse.Namespace) -> FixedPipeConfig:
    return FixedPipeConfig(
        yosys_bin=args.yosys_bin,
        yosys_timeout_s=args.yosys_timeout_s,
        second_edition_repair_attempts=args.second_edition_repair_attempts,
    )


def write_report(response, output_path: str | None) -> None:
    if not output_path:
        return
    report = build_latest_report(response)
    Path(output_path).write_text(json.dumps(report, default=json_default, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-assisted RTL generation prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build a vector index from merged.jsonl")
    index.add_argument("--corpus", default="merged.jsonl")
    index.add_argument("--output", default="indexes/rtl_hash")
    index.add_argument("--embedder", default="hash")
    index.add_argument("--limit", type=int, default=None)
    index.add_argument("--jobs", type=int, default=1, help="Number of parallel embedding workers")
    index.set_defaults(func=cmd_index)

    datapath_index = subparsers.add_parser(
        "datapath-index",
        help="Preprocess Verilog with Yosys and build a graph-wise vector index",
    )
    datapath_index.add_argument("--corpus", default="merged.jsonl")
    datapath_index.add_argument("--output", default="indexes/rtl_datapath_hash")
    datapath_index.add_argument("--embedder", default="hash")
    datapath_index.add_argument("--limit", type=int, default=None)
    datapath_index.add_argument("--yosys-bin", default="yosys")
    datapath_index.add_argument("--timeout-s", type=int, default=30)
    datapath_index.add_argument("--jobs", type=int, default=1, help="Number of parallel Yosys workers")
    datapath_index.set_defaults(func=cmd_datapath_index)

    generate = subparsers.add_parser("generate", help="Generate RTL with retrieval and verification feedback")
    generate.add_argument("--index", default="indexes/rtl_hash")
    generate.add_argument("--embedder", default="hash")
    generate.add_argument("--prompt")
    generate.add_argument("--prompt-file")
    generate.add_argument("--target-hdl", default="verilog")
    generate.add_argument("--module-signature")
    generate.add_argument("--constraint", action="append", default=[])
    generate.add_argument("--retrieve-k", type=int, default=8)
    generate.add_argument("--context-k", type=int, default=4)
    generate.add_argument("--max-repair-attempts", type=int, default=2)
    generate.add_argument("--cache", default="data/history_cache.json")
    generate.add_argument("--monitor", default="runs/monitor.jsonl")
    generate.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    generate.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    generate.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    generate.add_argument("--failed-log", default="runs/failed_attempts.jsonl")
    generate.add_argument("--verbose-generation", action="store_true")
    generate.add_argument("--generation-temperature", type=float, default=0.2)
    generate.add_argument("--max-tokens", type=int, default=32768)
    generate.add_argument("--enable-tool-calling", action="store_true")
    generate.add_argument("--tool-choice", default="auto", help="vLLM tool_choice value, for example auto or required.")
    generate.add_argument("--max-tool-rounds", type=int, default=4)
    generate.add_argument("--testbench")
    generate.add_argument("--top-module")
    generate.add_argument("--test-command")
    generate.add_argument("--json-report")
    generate.set_defaults(func=cmd_generate)

    fixed_pipe = subparsers.add_parser(
        "fixed-pipe",
        help="Run the thesis fixed pipeline: spec RAG, verify, Yosys graph, code-structure RAG, second edition",
    )
    fixed_pipe.add_argument("--spec-index", default="indexes/rtl_hash")
    fixed_pipe.add_argument("--code-structure-index", default="indexes/rtl_datapath_hash")
    fixed_pipe.add_argument("--embedder", default="hash")
    fixed_pipe.add_argument("--prompt")
    fixed_pipe.add_argument("--prompt-file")
    fixed_pipe.add_argument("--target-hdl", default="verilog")
    fixed_pipe.add_argument("--module-signature")
    fixed_pipe.add_argument("--constraint", action="append", default=[])
    fixed_pipe.add_argument("--retrieve-k", type=int, default=4)
    fixed_pipe.add_argument("--context-k", type=int, default=2)
    fixed_pipe.add_argument("--structure-retrieve-k", type=int, default=4)
    fixed_pipe.add_argument("--structure-context-k", type=int, default=2)
    fixed_pipe.add_argument("--max-repair-attempts", type=int, default=2)
    fixed_pipe.add_argument("--second-edition-repair-attempts", type=int, default=2)
    fixed_pipe.add_argument("--cache", default="data/history_cache.json")
    fixed_pipe.add_argument("--monitor", default="runs/monitor.jsonl")
    fixed_pipe.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    fixed_pipe.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    fixed_pipe.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    fixed_pipe.add_argument("--failed-log", default="runs/failed_attempts.jsonl")
    fixed_pipe.add_argument("--verbose-generation", action="store_true")
    fixed_pipe.add_argument("--generation-temperature", type=float, default=0.2)
    fixed_pipe.add_argument("--max-tokens", type=int, default=32768)
    fixed_pipe.add_argument("--enable-tool-calling", action="store_true")
    fixed_pipe.add_argument("--tool-choice", default="auto", help="vLLM tool_choice value, for example auto or required.")
    fixed_pipe.add_argument("--max-tool-rounds", type=int, default=4)
    fixed_pipe.add_argument("--testbench")
    fixed_pipe.add_argument("--top-module")
    fixed_pipe.add_argument("--test-command")
    fixed_pipe.add_argument("--yosys-bin", default="yosys")
    fixed_pipe.add_argument("--yosys-timeout-s", type=int, default=30)
    fixed_pipe.add_argument("--json-report")
    fixed_pipe.set_defaults(func=cmd_fixed_pipe)

    evaluate = subparsers.add_parser("evaluate", help="Run thesis baseline evaluation on a JSON/JSONL prompt set")
    evaluate.add_argument("--tasks", required=True, help="JSON/JSONL records with a prompt/spec field")
    evaluate.add_argument("--index", default="indexes/rtl_hash")
    evaluate.add_argument("--embedder", default="hash")
    evaluate.add_argument("--mode", choices=["llm_only", "rag", "rag_cache_verify"], default="rag_cache_verify")
    evaluate.add_argument("--output", default="runs/evaluation.json")
    evaluate.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    evaluate.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    evaluate.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    evaluate.set_defaults(func=cmd_evaluate)

    stg_evaluate = subparsers.add_parser(
        "stg-evaluate",
        help="Run the RAG generator on dataset specs, then check passing RTL against golden code with STG",
    )
    stg_evaluate.add_argument("--dataset", required=True, help="JSONL/JSON records with spec and golden code fields")
    stg_evaluate.add_argument("--index", default="indexes/rtl_hash")
    stg_evaluate.add_argument("--embedder", default="hash")
    stg_evaluate.add_argument("--output", default="runs/stg_evaluation.json")
    stg_evaluate.add_argument("--stg-bin", default="stg")
    stg_evaluate.add_argument("--target-hdl", default="verilog")
    stg_evaluate.add_argument("--module-signature")
    stg_evaluate.add_argument("--constraint", action="append", default=[])
    stg_evaluate.add_argument("--type", default="combinational", choices=["combinational", "seq_clocked", "seq_done"])
    stg_evaluate.add_argument("--limit", type=int, default=None)
    stg_evaluate.add_argument("--retrieve-k", type=int, default=8)
    stg_evaluate.add_argument("--context-k", type=int, default=4)
    stg_evaluate.add_argument("--max-repair-attempts", type=int, default=2)
    stg_evaluate.add_argument("--timeout-s", type=int, default=120)
    stg_evaluate.add_argument("--spec-field", help="Explicit dataset field containing the specification")
    stg_evaluate.add_argument("--golden-field", help="Explicit dataset field containing the golden/reference code")
    stg_evaluate.add_argument("--testbench")
    stg_evaluate.add_argument("--top-module")
    stg_evaluate.add_argument("--test-command")
    stg_evaluate.add_argument("--cache", default="data/history_cache.json")
    stg_evaluate.add_argument("--monitor", default="runs/stg_monitor.jsonl")
    stg_evaluate.add_argument("--failed-log", default="runs/stg_failed_attempts.jsonl")
    stg_evaluate.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    stg_evaluate.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    stg_evaluate.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    stg_evaluate.add_argument("--verbose-generation", action="store_true")
    stg_evaluate.add_argument("--generation-temperature", type=float, default=0.4)
    stg_evaluate.add_argument("--max-tokens", type=int, default=2048)
    stg_evaluate.add_argument("--enable-tool-calling", action="store_true")
    stg_evaluate.add_argument("--tool-choice", default="auto")
    stg_evaluate.add_argument("--max-tool-rounds", type=int, default=4)
    stg_evaluate.add_argument(
        "--save-result-code-dir",
        "--save-code-dir",
        dest="save_result_code_dir",
        help="Directory where every generated RTL result is written; defaults to OUTPUT stem plus _codes",
    )
    stg_evaluate.add_argument("--save-passed-dir", help="Directory where passing generated RTL files are written")
    stg_evaluate.add_argument(
        "--stg-arg",
        action="append",
        default=[],
        help="Additional argument passed to `stg generate`; repeat for multiple tokens, e.g. --stg-arg=--verilator",
    )
    stg_evaluate.set_defaults(func=cmd_stg_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) in {"generate", "fixed-pipe"} and not args.prompt and not args.prompt_file:
        parser.error(f"{args.command} requires --prompt or --prompt-file")
    args.func(args)


if __name__ == "__main__":
    main()
