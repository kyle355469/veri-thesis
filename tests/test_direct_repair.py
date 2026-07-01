"""Tests for the direct (no-planning) flow's syntax/functional repair loops.

Covers three seams:
  * the agent's plan-free ``repair_rtl`` entry point + the prompt builders' ``plan=None`` mode,
  * the direct runner wiring (``run_one_direct`` invokes repair when enabled, off by default),
  * the router's translation of pipeline repair flags onto the direct flow.
"""

import argparse
import importlib.util
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

from ip_reuse_legacy.agent import AgenticIpReuseAgent
from ip_reuse_legacy.config import AgenticIpReuseConfig
from ip_reuse_legacy.prompts import build_functional_repair_prompt, build_repair_prompt
from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import Diagnostic, VerificationReport
from rag_rtl.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt, temperature=0.1, max_tokens=2048):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("no fake LLM response left")
        return self.responses.pop(0)


class SequenceVerifier:
    def __init__(self, reports):
        self.reports = list(reports)

    def verify(self, rtl, top_module=None):
        if not self.reports:
            raise AssertionError("no fake verifier report left")
        return self.reports.pop(0)


@dataclass
class FuncReport:
    function_passed: bool
    function_info: str = ""
    syntax_ok: bool = True


class StubFunctionalVerifier:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = []

    def verify_functional(self, rtl, top_module=None):
        self.calls.append(rtl)
        return self.reports.pop(0)


def passing_report():
    return VerificationReport(syntax_passed=True, lint_passed=True, diagnostics=[Diagnostic(tool="stub", passed=True)])


def failing_report():
    return VerificationReport(
        syntax_passed=False,
        lint_passed=False,
        diagnostics=[Diagnostic(tool="verilator", passed=False, stderr="%Error-PARSE: bad")],
    )


def empty_retrieval_context():
    embedder = HashingEmbedder(dim=128)
    return RetrievalContext.from_store(VectorStore([], embedder.encode([])), embedder)


def make_agent(llm, syntax_reports, functional_verifier=None, *, max_repair=2, func=False, max_func=2):
    return AgenticIpReuseAgent(
        llm,
        empty_retrieval_context(),
        SequenceVerifier(syntax_reports),
        config=AgenticIpReuseConfig(
            max_repair_attempts=max_repair,
            enable_functional_repair=func,
            max_functional_repair_attempts=max_func,
        ),
        functional_verifier=functional_verifier,
    )


def rtl_block(tag):
    return f"```verilog\nmodule adder(input [7:0] a, b, output [8:0] y); assign y = a + b; /*{tag}*/ endmodule\n```"


class RepairRtlTests(unittest.TestCase):
    def test_passing_rtl_needs_no_repair_and_no_llm_call(self):
        llm = FakeLlm([])
        agent = make_agent(llm, [passing_report()])

        result = agent.repair_rtl("module adder; endmodule", top_module="adder")

        self.assertTrue(result.verification.passed)
        self.assertEqual(result.repair_attempts, 0)
        self.assertEqual(llm.prompts, [])  # no generation, no repair
        self.assertIn("module adder", result.rtl)

    def test_syntax_repair_loop_runs_on_provided_rtl(self):
        # First lint fails, repaired candidate compiles.
        llm = FakeLlm([rtl_block("fixed")])
        agent = make_agent(llm, [failing_report(), passing_report()], max_repair=2)

        result = agent.repair_rtl(
            "module adder; broken endmodule",
            top_module="adder",
            original_spec="Build an adder with output y = a + b.",
        )

        self.assertTrue(result.verification.passed)
        self.assertEqual(result.repair_attempts, 1)
        self.assertIn("fixed", result.rtl)
        # plan-free repair prompt: no IP-reuse-plan framing, but spec + diagnostics present.
        prompt = llm.prompts[0]
        self.assertNotIn("IP reuse plan:", prompt)
        self.assertNotIn("Keep the same IP reuse intent", prompt)
        self.assertIn("Build an adder", prompt)
        self.assertIn("PARSE", prompt)

    def test_functional_repair_loop_runs_after_compile(self):
        verifier = StubFunctionalVerifier(
            [FuncReport(False, "Output y has 4 mismatches. First at time 10"), FuncReport(True, "")]
        )
        llm = FakeLlm([rtl_block("logicfix")])
        agent = make_agent(llm, [passing_report()], verifier, func=True)

        result = agent.repair_rtl("module adder; endmodule", top_module="adder")

        self.assertTrue(result.verification.passed)
        self.assertEqual(result.functional_repair_attempts, 1)
        self.assertIn("logicfix", result.rtl)
        func_prompt = llm.prompts[0]
        self.assertIn("compiles cleanly", func_prompt)
        self.assertIn("4 mismatches", func_prompt)
        self.assertNotIn("IP reuse plan:", func_prompt)


class PlanNonePromptTests(unittest.TestCase):
    def test_repair_prompt_drops_plan_section(self):
        diags = [{"tool": "verilator", "stderr": "%Error-PARSE"}]
        with_none = build_repair_prompt(None, "module m; endmodule", diags, "verilog", "m")
        self.assertNotIn("IP reuse plan:", with_none)
        self.assertIn("Diagnostics:", with_none)
        self.assertIn("Top module: m", with_none)

    def test_functional_prompt_drops_plan_section(self):
        with_none = build_functional_repair_prompt(None, "module m; endmodule", "Output q has 2 mismatches", "verilog", "m")
        self.assertNotIn("IP reuse plan:", with_none)
        self.assertIn("2 mismatches", with_none)


# --- Router translation of repair flags onto the direct flow -------------------


def load_router():
    from scripts.run_realbench_routed import build_direct_parser, direct_repair_extra_args
    from scripts.run_agentic_plan_legacy_realbench import build_parser as build_pipeline_parser

    return build_pipeline_parser, build_direct_parser, direct_repair_extra_args


class RouterRepairFlagTests(unittest.TestCase):
    def test_pipeline_defaults_enable_direct_syntax_repair(self):
        build_pipeline_parser, build_direct_parser, derive = load_router()
        pipe_args = build_pipeline_parser().parse_args([])
        extra = derive(pipe_args)
        direct_args = build_direct_parser().parse_args(extra)

        # syntax repair on by default (mirrors the always-on pipeline syntax loop),
        # functional repair off (separate experimental arm, like --legacy-functional-repair).
        self.assertEqual(direct_args.max_repair_attempts, 2)
        self.assertFalse(direct_args.functional_repair)

    def test_functional_repair_propagates_to_direct(self):
        build_pipeline_parser, build_direct_parser, derive = load_router()
        pipe_args = build_pipeline_parser().parse_args(
            ["--legacy-functional-repair", "--legacy-max-functional-repair-attempts", "3"]
        )
        extra = derive(pipe_args)
        direct_args = build_direct_parser().parse_args(extra)

        self.assertTrue(direct_args.functional_repair)
        self.assertEqual(direct_args.max_functional_repair_attempts, 3)

    def test_forwarded_direct_flag_overrides_derived(self):
        build_pipeline_parser, build_direct_parser, derive = load_router()
        pipe_args = build_pipeline_parser().parse_args([])
        # derived flags come first; an explicit forwarded direct flag parses last and wins.
        extra = derive(pipe_args) + ["--max-repair-attempts", "5"]
        direct_args = build_direct_parser().parse_args(extra)
        self.assertEqual(direct_args.max_repair_attempts, 5)


# --- Direct runner wiring ------------------------------------------------------


def load_direct():
    path = REPO_ROOT / "scripts" / "run_realbench_direct_model.py"
    spec = importlib.util.spec_from_file_location("run_realbench_direct_model_t", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_realbench_direct_model_t"] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_fake_realbench(root: Path) -> None:
    write(root / "benchmark_info.py", "benchmark_info = {'aes': {'aes_sbox': []}}\nsystem_info = {}\n")
    write(
        root / "problems" / "aes" / "problems.jsonl",
        json.dumps({"task": "aes_sbox", "problem": "Build an aes_sbox with input a and output b."}) + "\n",
    )
    write(root / "aes" / "aes_sbox" / "verification" / "aes_sbox_top.sv", "module old; endmodule\n")


class FakeClient:
    def chat(self, messages, temperature, max_tokens):
        return {
            "content": "```verilog\nmodule aes_sbox(input [7:0] a, output [7:0] b); assign b = a; endmodule\n```",
            "_finish_reason": "stop",
        }


@dataclass
class FakeRepairResult:
    rtl: str
    repair_attempts: int = 1
    functional_repair_attempts: int = 0
    function_info: str = ""


class DirectRunnerWiringTests(unittest.TestCase):
    def _args(self, module, root, output, *extra):
        return module.build_parser().parse_args(
            [
                "--realbench-root", str(root),
                "--output-dir", str(output),
                "--task-level", "module",
                "--no-prepare-problems",
                "--realbench-verifier", "native",
                "--make-bin", str(self.fake_make),
                *extra,
            ]
        )

    def setUp(self):
        self.module = load_direct()

    def _fake_make(self, tmp_path):
        fake = tmp_path / "fake_make.py"
        write(
            fake,
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "text = Path('aes_sbox_top.sv').read_text(encoding='utf-8')\n"
            "raise SystemExit(0 if 'module aes_sbox' in text else 1)\n",
        )
        fake.chmod(0o755)
        return fake

    def test_repair_off_by_default_does_not_call_repair(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root, output = tmp_path / "rb", tmp_path / "out"
            make_fake_realbench(root)
            self.fake_make = self._fake_make(tmp_path)
            called = []
            self.module.repair_direct_code = lambda *a, **k: called.append(True)  # type: ignore
            args = self._args(self.module, root, output)
            task = self.module.discover_tasks(args)[0]
            catalog = self.module.build_task_catalog(task, output)
            item = self.module.WorkItem(task=task, sample=1)

            record = self.module.run_one_direct(item, args, output, catalog, FakeClient())

            self.assertEqual(called, [])
            self.assertFalse(record["direct_repair"])
            self.assertIsNone(record["direct_repair_attempts"])

    def test_repair_enabled_uses_repaired_code(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root, output = tmp_path / "rb", tmp_path / "out"
            make_fake_realbench(root)
            self.fake_make = self._fake_make(tmp_path)
            repaired = "module aes_sbox(input [7:0] a, output [7:0] b); assign b = a ^ 8'h63; endmodule\n"
            self.module.repair_direct_code = lambda *a, **k: FakeRepairResult(rtl=repaired, repair_attempts=2)  # type: ignore
            args = self._args(self.module, root, output, "--max-repair-attempts", "2")
            task = self.module.discover_tasks(args)[0]
            catalog = self.module.build_task_catalog(task, output)
            item = self.module.WorkItem(task=task, sample=1)

            record = self.module.run_one_direct(item, args, output, catalog, FakeClient())

            self.assertTrue(record["direct_repair"])
            self.assertEqual(record["direct_repair_attempts"], 2)
            saved = Path(record["generated_code_path"]).read_text(encoding="utf-8")
            self.assertIn("8'h63", saved)  # the repaired body, not the raw generation


if __name__ == "__main__":
    unittest.main()
