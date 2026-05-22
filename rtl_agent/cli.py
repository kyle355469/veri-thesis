from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from rag_rtl.embeddings import make_embedder
from rag_rtl.llm import VllmClient
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.stg_eval import infer_first_module_name, run_stg_equivalence
from rag_rtl.tool_calling import RTL_TOOL_SCHEMAS
from rag_rtl.types import RtlTask
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

from .agent import AgentConfig, AgenticRtlAgent, dumps_result
from .events import AgentEvent
from .harness import (
    DEFAULT_ALLOWED_COMMANDS,
    WORKSPACE_TOOL_SCHEMAS,
    CompositeToolExecutor,
    WorkspaceToolExecutor,
)


def cmd_run(args: argparse.Namespace) -> None:
    print_run_header(args)
    task = build_task(args)
    agent = build_agent(args)

    def print_event(event: AgentEvent) -> None:
        line = render_cli_event(event)
        if line:
            print(line, flush=True)

    result = agent.run(task, event_sink=print_event)
    run_final_stg_if_requested(args, result)
    print_final_result(result, show_failed_code=args.show_final_code)
    if args.json_report:
        output = Path(args.json_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dumps_result(result), encoding="utf-8")
        print(f"\nReport written: {output}")


def build_agent(args: argparse.Namespace) -> AgenticRtlAgent:
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
    config = AgentConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tool_choice=args.tool_choice,
        max_steps=args.max_steps,
        target_hdl=args.target_hdl,
        final_verify=True,
    )
    rtl_tools = retrieval_context.tool_executor(verifier=verifier, default_top_module=args.top_module)
    workspace_tools = WorkspaceToolExecutor(
        root=args.workspace_root,
        allowed_commands=sorted(DEFAULT_ALLOWED_COMMANDS | set(args.allow_command or [])),
        timeout_s=args.command_timeout_s,
        max_output_chars=args.command_max_output_chars,
    )
    return AgenticRtlAgent(
        llm_client=llm,
        tool_executor=CompositeToolExecutor(rtl_tools, workspace_tools),
        verifier=verifier,
        config=config,
        tool_schemas=[*RTL_TOOL_SCHEMAS, *WORKSPACE_TOOL_SCHEMAS],
    )


def build_task(args: argparse.Namespace) -> RtlTask:
    return RtlTask(
        prompt=read_prompt(args),
        target_hdl=args.target_hdl,
        module_signature=args.module_signature,
        constraints=args.constraint,
        max_repair_attempts=0,
        top_module=args.top_module,
        prompt_profile="tool",
    )


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise ValueError("Provide --prompt or --prompt-file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic RTL generation over vLLM tool calling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one agentic RTL generation task")
    add_run_args(run)
    run.set_defaults(func=cmd_run)
    return parser


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--target-hdl", default="verilog")
    parser.add_argument("--module-signature")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--top-module")

    parser.add_argument("--base-url", help="OpenAI-compatible vLLM base URL. Defaults to VLLM_BASE_URL.")
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
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
    parser.add_argument("--stg-golden", help="Golden/reference RTL code for final STG equivalence checking.")
    parser.add_argument("--stg-golden-file", help="File containing golden/reference RTL code for final STG equivalence checking.")
    parser.add_argument("--stg-bin", default="stg")
    parser.add_argument("--stg-type", default="combinational", choices=["combinational", "seq_clocked", "seq_done"])
    parser.add_argument("--stg-module", help="DUT module name passed to STG. Defaults to --top-module.")
    parser.add_argument("--stg-golden-module", help="Golden module name passed to STG. Defaults to first module in golden code.")
    parser.add_argument("--stg-timeout-s", type=int, default=120)
    parser.add_argument(
        "--stg-arg",
        action="append",
        default=[],
        help="Additional argument passed to `stg generate`; repeat for multiple args. Use --stg-arg=--flag for flag-like values.",
    )
    parser.add_argument("--workspace-root", default=".", help="Root directory for agent file and command tools.")
    parser.add_argument(
        "--allow-command",
        action="append",
        default=[],
        help="Additional executable name allowed for run_command; repeatable.",
    )
    parser.add_argument("--command-timeout-s", type=int, default=20)
    parser.add_argument("--command-max-output-chars", type=int, default=6000)

    parser.add_argument("--json-report")
    parser.add_argument(
        "--show-final-code",
        action="store_true",
        help="Also print final candidate code when verification fails. Passing code is always printed.",
    )


def print_run_header(args: argparse.Namespace) -> None:
    model = args.model or os.getenv("VLLM_MODEL", "siliconmind-server")
    base_url = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    print("")
    print("============================================================")
    print(" Agentic RTL Run")
    print("============================================================")
    print(f" model          : {model}")
    print(f" endpoint       : {base_url}")
    print(f" index          : {args.index}")
    print(f" workspace      : {args.workspace_root}")
    print(f" tool choice    : {args.tool_choice}")
    print(f" max steps      : {args.max_steps}")
    if args.top_module:
        print(f" top module     : {args.top_module}")
    if args.stg_golden or args.stg_golden_file:
        print(f" final STG      : enabled ({args.stg_type})")
    print("------------------------------------------------------------")
    print(" Trace")
    print("------------------------------------------------------------")


def run_final_stg_if_requested(args: argparse.Namespace, result: object) -> None:
    golden_code = read_stg_golden(args)
    if not golden_code:
        return
    print("  final   -> stg          running STG equivalence", flush=True)
    if not result.rtl:
        result.stg_result = {
            "passed": False,
            "stderr": "STG skipped because final model response did not contain parsable RTL.",
        }
        print("  final   <- stg          skipped: no parsable RTL", flush=True)
        return
    stg_result = run_stg_equivalence(
        result.rtl,
        golden_code,
        stg_bin=args.stg_bin,
        design_type=args.stg_type,
        dut_module=args.stg_module or args.top_module,
        golden_module=args.stg_golden_module or infer_first_module_name(golden_code),
        timeout_s=args.stg_timeout_s,
        extra_stg_args=args.stg_arg or [],
    )
    result.stg_result = stg_result
    status = "passed" if stg_result.passed else "failed"
    detail = stg_result.stderr[-240:] or stg_result.stdout[-240:]
    suffix = f": {detail.strip()}" if detail.strip() else ""
    print(f"  final   <- stg          {status}{suffix}", flush=True)


def read_stg_golden(args: argparse.Namespace) -> str:
    if args.stg_golden:
        return args.stg_golden
    if args.stg_golden_file:
        return Path(args.stg_golden_file).read_text(encoding="utf-8")
    return ""


def render_cli_event(event: AgentEvent) -> str:
    if event.event == "agent_start":
        return "  agent loop started"
    if event.event == "tool_call":
        return f"  step {event.step:<2} -> tool call      {event.tool}"
    if event.event == "tool_result":
        return f"  step {event.step:<2} <- tool result    {event.message}"
    if event.event == "model_final":
        return f"  step {event.step:<2} -> final answer   {event.message}"
    if event.event == "forced_final":
        return f"  step {event.step:<2} -> forced final   {event.message}"
    if event.event == "final_verification":
        return f"  final   -> verify       {event.message}"
    return f"  {event.render()}"


def print_final_result(result: object, show_failed_code: bool = False) -> None:
    verification = result.verification
    stg_result = getattr(result, "stg_result", None)
    stg_passed = True if stg_result is None else _stg_passed(stg_result)
    passed = verification.passed and stg_passed
    print("------------------------------------------------------------")
    print(" Result")
    print("------------------------------------------------------------")
    print(f" status         : {'PASS' if passed else 'FAIL'}")
    print(f" steps          : {result.steps}")
    print(f" used tools     : {'yes' if result.used_tools else 'no'}")
    print(f" stopped reason : {result.stopped_reason}")
    failed_tools = [item.tool for item in verification.diagnostics if not item.passed]
    if failed_tools:
        print(f" failed tools   : {', '.join(failed_tools)}")
    if stg_result is not None:
        print(f" stg            : {'PASS' if stg_passed else 'FAIL'}")

    should_print_code = passed or show_failed_code
    if should_print_code:
        print("------------------------------------------------------------")
        print(" Result Code")
        print("------------------------------------------------------------")
        code = result.rtl.strip()
        print(code if code else "<no parsable RTL>")
        print("------------------------------------------------------------")


def _stg_passed(stg_result: object) -> bool:
    if isinstance(stg_result, dict):
        return bool(stg_result.get("passed"))
    return bool(getattr(stg_result, "passed", False))


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and not args.prompt and not args.prompt_file:
        parser.error("run requires --prompt or --prompt-file")
    args.func(args)


if __name__ == "__main__":
    main()
