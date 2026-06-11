import json
import tempfile
import unittest
from pathlib import Path

from ip_reuse_legacy.agent import AgenticIpReuseAgent
from ip_reuse_legacy.cli import build_parser
from ip_reuse_legacy.plan_adapter import agentic_plan_from_payload, load_agentic_plan
from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import Diagnostic, VerificationReport
from rag_rtl.vector_store import VectorStore


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


def passing_report():
    return VerificationReport(
        syntax_passed=True,
        lint_passed=True,
        diagnostics=[Diagnostic(tool="stub", passed=True)],
    )


def empty_retrieval_context():
    embedder = HashingEmbedder(dim=128)
    return RetrievalContext.from_store(VectorStore([], embedder.encode([])), embedder)


def sample_agentic_payload():
    return {
        "structured_plan": {
            "requirements": {
                "functionality": ["add two inputs"],
                "performance": ["combinational"],
                "io_interfaces": ["plain ports"],
                "ppa_constraints": ["small area"],
                "clock_reset": ["none"],
            },
            "modules": [
                {
                    "name": "adder",
                    "role": "produce y = a + b",
                    "interfaces": ["input [7:0] a", "input [7:0] b", "output [8:0] y"],
                    "reuse_preference": "new RTL",
                }
            ],
            "reuse_decisions": [
                {
                    "module_name": "adder",
                    "selected_ip": None,
                    "new_rtl_required": True,
                    "risk_notes": ["simple generated module"],
                }
            ],
            "integration_plan": ["instantiate adder as the top module"],
            "verification_plan": ["lint generated RTL"],
            "debug_plan": [],
            "unresolved_assumptions": [],
        }
    }


class LegacyPlanPipelineTests(unittest.TestCase):
    def test_parser_accepts_run_plan_args(self):
        args = build_parser().parse_args(
            [
                "run-plan",
                "--plan-file",
                "runs/plan.json",
                "--target-hdl",
                "systemverilog",
                "--top-module",
                "adder",
                "--json-report",
                "runs/report.json",
            ]
        )

        self.assertEqual(args.command, "run-plan")
        self.assertEqual(args.plan_file, "runs/plan.json")
        self.assertEqual(args.target_hdl, "systemverilog")
        self.assertEqual(args.top_module, "adder")

    def test_agentic_plan_adapter_maps_to_legacy_plan(self):
        plan = agentic_plan_from_payload(sample_agentic_payload())

        self.assertEqual(plan.requirements.functionality, "add two inputs")
        self.assertEqual(plan.modules[0].name, "adder")
        self.assertEqual(plan.modules[0].purpose, "produce y = a + b")
        self.assertEqual(plan.decisions[0].action, "new")

    def test_run_from_plan_skips_decomposition_and_generates_rtl(self):
        plan = agentic_plan_from_payload(sample_agentic_payload())
        llm = FakeLlm(["```verilog\nmodule adder(input [7:0] a,b, output [8:0] y); assign y = a + b; endmodule\n```"])
        agent = AgenticIpReuseAgent(llm, empty_retrieval_context(), SequenceVerifier([passing_report()]))

        result = agent.run_from_plan(plan, top_module="adder")

        self.assertTrue(result.verification.passed)
        self.assertIn("module adder", result.rtl)
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("Generate integrated", llm.prompts[0])
        self.assertNotIn("Extract system-level requirements", llm.prompts[0])
        self.assertNotIn("Decompose the IC system", llm.prompts[0])

    def test_load_agentic_plan_accepts_agent_result_json_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_result.json"
            path.write_text(json.dumps(sample_agentic_payload()), encoding="utf-8")

            plan = load_agentic_plan(path)

        self.assertEqual(plan.modules[0].name, "adder")


if __name__ == "__main__":
    unittest.main()
