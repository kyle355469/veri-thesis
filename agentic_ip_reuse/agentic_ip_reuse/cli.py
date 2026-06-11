from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

from .agent import AgentConfig, AgenticIpReuseAgent, dumps_result
from .hierarchical import HierarchicalAgent, HierarchicalConfig
from .llm import MockLlmClient, VllmClient
from .repository import JsonIpRepository
from .tools import AgentToolExecutor
from .types import AgentResult, DesignTask


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic IC IP-reuse planning framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one IP-reuse planning task")
    add_run_args(run)
    run.set_defaults(func=cmd_run)
    return parser


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--target-hdl", default="systemverilog")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--known-interface", action="append", default=[])
    parser.add_argument("--ppa-target", action="append", default=[])
    parser.add_argument("--catalog", default=str(default_catalog_path()))
    parser.add_argument("--output-dir", default="agentic_ip_reuse_out")
    parser.add_argument("--json-report")
    parser.add_argument("--mock-llm", action="store_true")

    parser.add_argument("--base-url", help="OpenAI-compatible vLLM base URL. Defaults to VLLM_BASE_URL.")
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
    parser.add_argument("--llm-timeout-s", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--hierarchical", action="store_true", help="Enable recursive hierarchical module decomposition")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum recursion depth for hierarchical decomposition")


def cmd_run(args: argparse.Namespace) -> AgentResult:
    task = build_task(args)
    repository = JsonIpRepository(args.catalog)
    executor = AgentToolExecutor(repository, args.output_dir)
    llm = build_llm(args)
    agent_config = AgentConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tool_choice=args.tool_choice,
        max_steps=args.max_steps,
    )
    print_run_header(args)

    if getattr(args, "hierarchical", False):
        h_agent = HierarchicalAgent(
            llm_client=llm,
            base_executor=executor,
            agent_config=agent_config,
            h_config=HierarchicalConfig(max_depth=args.max_depth),
        )
        h_plan = h_agent.run(task)
        result = h_plan.result
        summary_path = h_plan.write_hierarchical_summary(Path(args.output_dir))
        print_hierarchical_result(h_plan, summary_path)
    else:
        agent = AgenticIpReuseAgent(
            llm_client=llm,
            tool_executor=executor,
            config=agent_config,
        )
        result = agent.run(task)
        print_final_result(result)

    report_path = Path(args.json_report) if args.json_report else Path(args.output_dir) / "agent_result.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(dumps_result(result), encoding="utf-8")
    return result


def build_task(args: argparse.Namespace) -> DesignTask:
    return DesignTask(
        prompt=read_prompt(args),
        target_hdl=args.target_hdl,
        constraints=list(args.constraint or []),
        known_interfaces=list(args.known_interface or []),
        ppa_targets=list(args.ppa_target or []),
    )


def build_llm(args: argparse.Namespace):
    if args.mock_llm:
        return MockLlmClient()
    return VllmClient(
        base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=args.llm_timeout_s,
    )


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise ValueError("Provide --prompt or --prompt-file.")


def print_run_header(args: argparse.Namespace) -> None:
    mode = "mock" if args.mock_llm else "vLLM"
    hierarchical = getattr(args, "hierarchical", False)
    print("")
    print("============================================================")
    print(" Agentic IP Reuse Run")
    print("============================================================")
    print(f" mode           : {mode}")
    print(f" catalog        : {args.catalog}")
    print(f" output         : {args.output_dir}")
    print(f" max steps      : {args.max_steps}")
    if hierarchical:
        print(f" hierarchical   : enabled (max depth {args.max_depth})")
    print("------------------------------------------------------------")


def print_final_result(result: AgentResult) -> None:
    print("Result")
    print(f" status         : OK")
    print(f" stopped reason : {result.stopped_reason}")
    print(f" steps          : {result.steps}")
    print(f" used tools     : {result.used_tools}")
    print(" artifacts")
    for key, path in sorted(result.artifact_paths.items()):
        print(f"  {key:22}: {path}")


def print_hierarchical_result(h_plan: Any, summary_path: Path) -> None:
    print("Result (hierarchical)")
    print(f" status         : OK")
    _print_hierarchy(h_plan, indent=1)
    print(f" summary        : {summary_path}")


def _print_hierarchy(plan: Any, indent: int) -> None:
    prefix = "  " * indent
    print(f"{prefix}depth {plan.depth}: {plan.result.stopped_reason}, {plan.result.steps} steps, "
          f"{len(plan.result.structured_plan.get('modules', []))} modules")
    for child_name, child in plan.children.items():
        print(f"{prefix}  └─ {child_name}")
        _print_hierarchy(child, indent + 2)


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "ip_catalog.json"


if __name__ == "__main__":
    raise SystemExit(main())
