import unittest
from dataclasses import dataclass

from ip_reuse_legacy.agent import AgenticIpReuseAgent
from ip_reuse_legacy.config import AgenticIpReuseConfig
from ip_reuse_legacy.plan_adapter import agentic_plan_from_payload
from ip_reuse_legacy.stages.functional_repair import _mismatch_count
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
        if not self.reports:
            raise AssertionError("no fake functional report left")
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


def sample_payload():
    return {
        "structured_plan": {
            "requirements": {"functionality": ["add"], "performance": ["comb"], "io_interfaces": ["ports"]},
            "modules": [{"name": "adder", "role": "y=a+b", "interfaces": ["output [8:0] y"], "reuse_preference": "new RTL"}],
            "reuse_decisions": [{"module_name": "adder", "selected_ip": None, "new_rtl_required": True}],
            "integration_plan": ["top adder"],
            "verification_plan": [],
            "debug_plan": [],
            "unresolved_assumptions": [],
        }
    }


def rtl_block(tag):
    return f"```verilog\nmodule adder(input [7:0] a, b, output [8:0] y); assign y = a + b; /*{tag}*/ endmodule\n```"


def make_agent(llm, syntax_reports, functional_verifier, *, enable=True, max_func=2, max_repair=2):
    return AgenticIpReuseAgent(
        llm,
        empty_retrieval_context(),
        SequenceVerifier(syntax_reports),
        config=AgenticIpReuseConfig(
            max_repair_attempts=max_repair,
            enable_functional_repair=enable,
            max_functional_repair_attempts=max_func,
        ),
        functional_verifier=functional_verifier,
    )


class MismatchCountTests(unittest.TestCase):
    def test_sums_counts_and_handles_edges(self):
        self.assertEqual(_mismatch_count("Output y has 5 mismatches. First at time 15"), 5)
        self.assertEqual(_mismatch_count("Output a has 3 mismatches\nOutput b has 7 mismatches"), 10)
        self.assertEqual(_mismatch_count(""), 0)
        self.assertGreater(_mismatch_count("garbage with no number"), 1_000_000)


class FunctionalRepairTests(unittest.TestCase):
    def test_disabled_skips_phase_even_with_verifier(self):
        verifier = StubFunctionalVerifier([FuncReport(True)])
        llm = FakeLlm([rtl_block("v0")])
        agent = make_agent(llm, [passing_report()], verifier, enable=False)

        result = agent.run_from_plan(agentic_plan_from_payload(sample_payload()), top_module="adder")

        self.assertEqual(verifier.calls, [])
        self.assertEqual(result.functional_repair_attempts, 0)
        self.assertEqual(result.function_info, "")
        self.assertEqual(len(llm.prompts), 1)

    def test_not_entered_when_syntax_fails(self):
        verifier = StubFunctionalVerifier([FuncReport(True)])
        # generation + one syntax repair, both fail to compile.
        llm = FakeLlm([rtl_block("v0"), rtl_block("v1")])
        agent = make_agent(llm, [failing_report(), failing_report()], verifier, max_repair=1)

        result = agent.run_from_plan(agentic_plan_from_payload(sample_payload()), top_module="adder")

        self.assertFalse(result.verification.passed)
        self.assertEqual(verifier.calls, [])
        self.assertEqual(result.functional_repair_attempts, 0)

    def test_functional_pass_on_first_check_needs_no_repair(self):
        verifier = StubFunctionalVerifier([FuncReport(True, "")])
        llm = FakeLlm([rtl_block("v0")])
        agent = make_agent(llm, [passing_report()], verifier)

        result = agent.run_from_plan(agentic_plan_from_payload(sample_payload()), top_module="adder")

        self.assertTrue(result.verification.passed)
        self.assertEqual(result.functional_repair_attempts, 0)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(len(llm.prompts), 1)  # generation only, no repair turn

    def test_functional_repair_fixes_logic(self):
        verifier = StubFunctionalVerifier(
            [FuncReport(False, "Output y has 5 mismatches. First at time 15"), FuncReport(True, "")]
        )
        llm = FakeLlm([rtl_block("v0"), rtl_block("v1")])
        agent = make_agent(llm, [passing_report()], verifier)

        result = agent.run_from_plan(agentic_plan_from_payload(sample_payload()), top_module="adder")

        self.assertTrue(result.verification.passed)
        self.assertEqual(result.functional_repair_attempts, 1)
        self.assertIn("v1", result.rtl)
        # The functional repair prompt frames it as a logic fix and carries the mismatch report.
        repair_prompt = llm.prompts[1]
        self.assertIn("compiles cleanly", repair_prompt)
        self.assertIn("5 mismatches", repair_prompt)
        self.assertIn("Do NOT change the module's port interface", repair_prompt)
        events = [event["event"] for event in result.functional_repair_events]
        self.assertEqual(events, ["verify", "repair"])

    def test_keeps_best_compiling_candidate_when_never_passing(self):
        verifier = StubFunctionalVerifier(
            [
                FuncReport(False, "Output y has 10 mismatches. First at time 5"),
                FuncReport(False, "Output y has 3 mismatches. First at time 5"),  # best
                FuncReport(False, "Output y has 7 mismatches. First at time 5"),
            ]
        )
        llm = FakeLlm([rtl_block("v0"), rtl_block("v1"), rtl_block("v2")])
        agent = make_agent(llm, [passing_report()], verifier, max_func=2)

        result = agent.run_from_plan(agentic_plan_from_payload(sample_payload()), top_module="adder")

        self.assertFalse(result.verification.passed)
        self.assertEqual(result.functional_repair_attempts, 2)
        self.assertIn("v1", result.rtl)  # lowest mismatch count wins
        self.assertIn("3 mismatches", result.function_info)
        # Per-attempt mismatch trajectory is captured for downstream analysis: the
        # verify baseline, the improving turn (accepted), and the regressing turn.
        events = result.functional_repair_events
        self.assertEqual([e["mismatches"] for e in events], [10, 3, 7])
        self.assertEqual([e["accepted_as_best"] for e in events[1:]], [True, False])
        self.assertEqual(events[2]["prev_best_mismatches"], 3)

    def test_syntax_regressing_candidate_is_discarded(self):
        verifier = StubFunctionalVerifier(
            [
                FuncReport(False, "Output y has 10 mismatches. First at time 5", syntax_ok=True),
                FuncReport(False, "", syntax_ok=False),  # candidate broke compilation
            ]
        )
        llm = FakeLlm([rtl_block("v0"), rtl_block("v1")])
        agent = make_agent(llm, [passing_report()], verifier, max_func=1)

        result = agent.run_from_plan(agentic_plan_from_payload(sample_payload()), top_module="adder")

        self.assertFalse(result.verification.passed)
        self.assertIn("v0", result.rtl)  # kept the last compiling design
        self.assertNotIn("v1", result.rtl)
        self.assertIn("10 mismatches", result.function_info)


if __name__ == "__main__":
    unittest.main()
