from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import CacheConfig, FixedPipeConfig, RuntimeConfig, ToolCallingConfig
from .dataset import iter_jsonl_documents
from .datapath import build_datapath_vector_db
from .embeddings import make_embedder
from .evaluation import run_evaluation
from .json_utils import json_default
from .pipeline import FixedPipeRtlPipeline, RagRtlPipeline
from .reporting import build_latest_report
from .types import RtlTask
from .vector_store import VectorStore, build_vector_store
from .verifier import RtlVerifier


def cmd_index(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    documents = list(iter_jsonl_documents(args.corpus, limit=args.limit))
    texts = [document.retrieval_text for document in documents]
    vectors = embedder.encode(texts)
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
    )
    print(
        "Built datapath graph VectorDB "
        f"from {stats.source_documents} documents into {args.output}: "
        f"{stats.graphs} graphs, {stats.skipped} skipped"
    )
    if stats.skipped:
        print(f"Yosys failures/skips were recorded in {stats.failures_path}")


def cmd_generate(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    store = VectorStore.load(args.index)
    pipeline = RagRtlPipeline(
        store=store,
        embedder=embedder,
        verifier=RtlVerifier(
            testbench_path=args.testbench,
            test_command=args.test_command,
        ),
        cache_config=CacheConfig(
            path=args.cache,
            mode=args.cache_mode,
            reuse_threshold=args.cache_reuse_threshold,
            evidence_threshold=args.cache_evidence_threshold,
        ),
        runtime_config=RuntimeConfig(
            monitor_path=args.monitor,
            failed_log_path=args.failed_log,
            verbose_generation=args.verbose_generation,
        ),
        tool_config=ToolCallingConfig(
            enabled=args.enable_tool_calling,
            choice=args.tool_choice,
            max_rounds=args.max_tool_rounds,
        ),
    )
    task = RtlTask(
        prompt=args.prompt or Path(args.prompt_file).read_text(encoding="utf-8"),
        target_hdl=args.target_hdl,
        module_signature=args.module_signature,
        constraints=args.constraint,
        max_repair_attempts=args.max_repair_attempts,
        top_module=args.top_module,
    )
    response = pipeline.run(task, retrieve_k=args.retrieve_k, context_k=args.context_k)
    print(response.rtl)
    if args.json_report:
        report = build_latest_report(response)
        Path(args.json_report).write_text(json.dumps(report, default=json_default, indent=2), encoding="utf-8")


def cmd_fixed_pipe(args: argparse.Namespace) -> None:
    embedder = make_embedder(args.embedder)
    spec_store = VectorStore.load(args.spec_index)
    code_structure_store = VectorStore.load(args.code_structure_index)
    pipeline = FixedPipeRtlPipeline(
        spec_store=spec_store,
        code_structure_store=code_structure_store,
        embedder=embedder,
        verifier=RtlVerifier(
            testbench_path=args.testbench,
            test_command=args.test_command,
        ),
        cache_config=CacheConfig(
            path=args.cache,
            mode=args.cache_mode,
            reuse_threshold=args.cache_reuse_threshold,
            evidence_threshold=args.cache_evidence_threshold,
        ),
        runtime_config=RuntimeConfig(
            monitor_path=args.monitor,
            failed_log_path=args.failed_log,
            verbose_generation=args.verbose_generation,
        ),
        tool_config=ToolCallingConfig(
            enabled=args.enable_tool_calling,
            choice=args.tool_choice,
            max_rounds=args.max_tool_rounds,
        ),
        fixed_pipe_config=FixedPipeConfig(
            yosys_bin=args.yosys_bin,
            yosys_timeout_s=args.yosys_timeout_s,
            second_edition_repair_attempts=args.second_edition_repair_attempts,
        ),
    )
    task = RtlTask(
        prompt=args.prompt or Path(args.prompt_file).read_text(encoding="utf-8"),
        target_hdl=args.target_hdl,
        module_signature=args.module_signature,
        constraints=args.constraint,
        max_repair_attempts=args.max_repair_attempts,
        top_module=args.top_module,
    )
    response = pipeline.run(
        task,
        retrieve_k=args.retrieve_k,
        context_k=args.context_k,
        structure_retrieve_k=args.structure_retrieve_k,
        structure_context_k=args.structure_context_k,
    )
    print(response.rtl)
    if args.json_report:
        report = build_latest_report(response)
        Path(args.json_report).write_text(json.dumps(report, default=json_default, indent=2), encoding="utf-8")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-assisted RTL generation prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build a vector index from merged.jsonl")
    index.add_argument("--corpus", default="merged.jsonl")
    index.add_argument("--output", default="indexes/rtl_hash")
    index.add_argument("--embedder", default="hash")
    index.add_argument("--limit", type=int, default=None)
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
    generate.add_argument("--max-repair-attempts", type=int, default=1)
    generate.add_argument("--cache", default="data/history_cache.json")
    generate.add_argument("--monitor", default="runs/monitor.jsonl")
    generate.add_argument("--cache-mode", choices=["keywords", "direct"], default="keywords")
    generate.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    generate.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    generate.add_argument("--failed-log", default="runs/failed_attempts.jsonl")
    generate.add_argument("--verbose-generation", action="store_true")
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
    fixed_pipe.add_argument("--max-repair-attempts", type=int, default=1)
    fixed_pipe.add_argument("--second-edition-repair-attempts", type=int, default=1)
    fixed_pipe.add_argument("--cache", default="data/history_cache.json")
    fixed_pipe.add_argument("--monitor", default="runs/monitor.jsonl")
    fixed_pipe.add_argument("--cache-mode", choices=["keywords", "direct"], default="keywords")
    fixed_pipe.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    fixed_pipe.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    fixed_pipe.add_argument("--failed-log", default="runs/failed_attempts.jsonl")
    fixed_pipe.add_argument("--verbose-generation", action="store_true")
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

    evaluate = subparsers.add_parser("evaluate", help="Run thesis baseline evaluation on a JSONL prompt set")
    evaluate.add_argument("--tasks", required=True, help="JSONL with at least a 'prompt' field per row")
    evaluate.add_argument("--index", default="indexes/rtl_hash")
    evaluate.add_argument("--embedder", default="hash")
    evaluate.add_argument("--mode", choices=["llm_only", "rag", "rag_cache_verify"], default="rag_cache_verify")
    evaluate.add_argument("--output", default="runs/evaluation.json")
    evaluate.add_argument("--cache-mode", choices=["keywords", "direct"], default="keywords")
    evaluate.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    evaluate.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) in {"generate", "fixed-pipe"} and not args.prompt and not args.prompt_file:
        parser.error(f"{args.command} requires --prompt or --prompt-file")
    args.func(args)


if __name__ == "__main__":
    main()
