import unittest

from ip_reuse_legacy.agent import AgenticIpReuseAgent
from ip_reuse_legacy.config import AgenticIpReuseConfig
from ip_reuse_legacy.plan_adapter import agentic_plan_from_payload
from ip_reuse_legacy.spec_slice import (
    extract_diagnostic_signals,
    extract_function_signals,
    slice_spec_for_diagnostics,
    spec_sections,
)
from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import Diagnostic, VerificationReport
from rag_rtl.vector_store import VectorStore


SPEC = """# Overview

This module bridges the bus.

## Interface

| Signal | Direction | Width | Description |
| --- | --- | --- | --- |
| wb_clk_i | Input | 1 | bus clock |
| dat_o | Output | 8 | data out |

## Clocking Behavior

The wb_clk_i drives the synchronous logic; every wb_clk_i edge advances state.

## Data Output Behavior

dat_o presents the latched read data one cycle after the request.

## Unrelated Arithmetic

The multiplier uses booth encoding and is independent of the bus interface.
"""


class SignalExtractionTests(unittest.TestCase):
    def test_extract_diagnostic_signals_pulls_pin_and_code(self):
        diagnostics = [
            {"stderr": "%Error-PINNOTFOUND: t/x.sv:12:5: Pin not found: 'wb_clk_i'", "stdout": ""}
        ]
        codes, identifiers = extract_diagnostic_signals(diagnostics)
        self.assertIn("PINNOTFOUND", codes)
        self.assertIn("wb_clk_i", identifiers)

    def test_parse_error_quoting_punctuation_yields_no_identifier(self):
        diagnostics = [{"stderr": "%Error-PARSE: t/x.sv:5: syntax error, unexpected ';'", "stdout": ""}]
        codes, identifiers = extract_diagnostic_signals(diagnostics)
        self.assertIn("PARSE", codes)
        self.assertEqual(identifiers, set())

    def test_extract_function_signals_handles_quoted_and_unquoted(self):
        self.assertEqual(
            extract_function_signals("Output 'clk_out' has 101 mismatches. First at time 115."),
            {"clk_out"},
        )
        self.assertEqual(extract_function_signals("Output text_out has 287 mismatches"), {"text_out"})

    def test_spec_sections_split_on_headings(self):
        sections = spec_sections(SPEC)
        self.assertEqual(len(sections), 5)  # overview + 4 headed sections
        self.assertTrue(sections[1].lstrip().startswith("## Interface"))


class SliceTests(unittest.TestCase):
    def test_diagnostic_slice_keeps_relevant_drops_unrelated(self):
        diagnostics = [{"stderr": "%Error-PINNOTFOUND: x.sv:1: Pin not found: 'wb_clk_i'", "stdout": ""}]
        sliced = slice_spec_for_diagnostics(SPEC, diagnostics=diagnostics, max_chars=24000)
        self.assertIn("wb_clk_i", sliced)
        self.assertIn("Clocking Behavior", sliced)
        self.assertNotIn("booth", sliced)  # unrelated section excluded
        self.assertLessEqual(len(sliced), 24000)

    def test_functional_slice_targets_failing_output(self):
        sliced = slice_spec_for_diagnostics(SPEC, function_info="Output 'dat_o' has 5 mismatches", max_chars=24000)
        self.assertIn("dat_o", sliced)
        self.assertIn("Data Output Behavior", sliced)
        self.assertNotIn("booth", sliced)

    def test_no_identifier_returns_empty_for_fallback(self):
        diagnostics = [{"stderr": "%Error-PARSE: x.sv:5: syntax error, unexpected ';'", "stdout": ""}]
        self.assertEqual(slice_spec_for_diagnostics(SPEC, diagnostics=diagnostics, max_chars=24000), "")

    def test_empty_spec_returns_empty(self):
        self.assertEqual(slice_spec_for_diagnostics("", diagnostics=[{"stderr": "x"}], max_chars=100), "")


# --- agent integration: slice only swapped into the repair prompt when flag on ---


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
        return self.reports.pop(0)


def empty_retrieval_context():
    embedder = HashingEmbedder(dim=128)
    return RetrievalContext.from_store(VectorStore([], embedder.encode([])), embedder)


def sample_payload():
    return {
        "structured_plan": {
            "requirements": {"functionality": ["bridge"], "performance": ["sync"], "io_interfaces": ["wishbone"]},
            "modules": [{"name": "bridge", "role": "bus", "interfaces": ["input wb_clk_i"], "reuse_preference": "new RTL"}],
            "reuse_decisions": [{"module_name": "bridge", "selected_ip": None, "new_rtl_required": True}],
            "integration_plan": ["top bridge"],
            "verification_plan": [],
            "debug_plan": [],
            "unresolved_assumptions": [],
        }
    }


def pinnotfound_report():
    return VerificationReport(
        syntax_passed=False,
        lint_passed=False,
        diagnostics=[Diagnostic(tool="verilator", passed=False, stderr="%Error-PINNOTFOUND: x.sv:1: Pin not found: 'wb_clk_i'")],
    )


def passing_report():
    return VerificationReport(syntax_passed=True, lint_passed=True, diagnostics=[Diagnostic(tool="stub", passed=True)])


def run_repair(enable_slice):
    llm = FakeLlm(["```verilog\nmodule bridge(input clk); endmodule\n```", "```verilog\nmodule bridge(input wb_clk_i); endmodule\n```"])
    agent = AgenticIpReuseAgent(
        llm,
        empty_retrieval_context(),
        SequenceVerifier([pinnotfound_report(), passing_report()]),
        config=AgenticIpReuseConfig(max_repair_attempts=1, enable_repair_spec_slice=enable_slice),
    )
    agent.run_from_plan(agentic_plan_from_payload(sample_payload()), top_module="bridge", original_spec=SPEC)
    return llm.prompts[1]  # the repair prompt


class AgentSliceIntegrationTests(unittest.TestCase):
    def test_flag_on_injects_focused_slice(self):
        prompt = run_repair(enable_slice=True)
        self.assertIn("wb_clk_i", prompt)
        self.assertNotIn("booth", prompt)  # unrelated section sliced away

    def test_flag_off_keeps_full_spec(self):
        prompt = run_repair(enable_slice=False)
        self.assertIn("booth", prompt)  # full spec retained unchanged


if __name__ == "__main__":
    unittest.main()
