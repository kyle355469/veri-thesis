from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from rag_rtl.embeddings import make_embedder
from rag_rtl.llm import VllmClient
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

from .agent import AgenticIpReuseAgent, AgenticIpReuseConfig, dumps_result


def cmd_run(args: argparse.Namespace) -> None:
    agent = build_agent(args)
    result = agent.run(
        read_prompt(args),
        target_hdl=args.target_hdl,
        top_module=args.top_module,
        constraints=args.constraint,
    )
    print(result.rtl)
    if args.json_report:
        output = Path(args.json_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dumps_result(result), encoding="utf-8")
        print(f"\nReport written: {output}")


def build_agent(args: argparse.Namespace) -> AgenticIpReuseAgent:
    embedder = make_embedder(args.embedder)
    store = VectorStore.load(args.index)
    retrieval_context = RetrievalContext.from_store(store, embedder)
    verifier = RtlVerifier(
        yosys_bin=args.yosys_bin,
        verilator_bin=args.verilator_bin,
        timeout_s=args.timeout_s,
        testbench_path=args.testbench,
        test_command=args.test_command,
    )
    llm = VllmClient(
        base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=args.llm_timeout_s,
    )
    config = AgenticIpReuseConfig(
        target_hdl=args.target_hdl,
        retrieve_k=args.retrieve_k,
        context_k=args.context_k,
        max_repair_attempts=args.max_repair_attempts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return AgenticIpReuseAgent(llm, retrieval_context, verifier, config)


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise ValueError("Provide --prompt or --prompt-file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic IC design with IP reuse over existing RTL indexes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one agentic IP reuse design task")
    add_run_args(run)
    run.set_defaults(func=cmd_run)
    return parser


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--target-hdl", default="verilog")
    parser.add_argument("--top-module")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--max-repair-attempts", type=int, default=2)

    parser.add_argument("--base-url", help="OpenAI-compatible vLLM base URL. Defaults to VLLM_BASE_URL.")
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
    parser.add_argument("--llm-timeout-s", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=32768)

    parser.add_argument("--yosys-bin", default="yosys")
    parser.add_argument("--verilator-bin", default="verilator")
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--testbench")
    parser.add_argument("--test-command")
    parser.add_argument("--json-report")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and not args.prompt and not args.prompt_file:
        parser.error("run requires --prompt or --prompt-file")
    args.func(args)


if __name__ == "__main__":
    main()
