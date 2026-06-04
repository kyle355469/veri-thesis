import json
import unittest

from agentic_ip_reuse.agent import AgenticIpReuseAgent, AgenticIpReuseConfig, candidate_from_hit, dumps_result
from agentic_ip_reuse.cli import build_parser
from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import Diagnostic, RetrievalHit, RtlDocument, VerificationReport
from rag_rtl.vector_store import build_vector_store


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
        self.calls = []

    def verify(self, rtl, top_module=None):
        self.calls.append((rtl, top_module))
        if not self.reports:
            raise AssertionError("no fake verifier report left")
        return self.reports.pop(0)


def passing_report():
    return VerificationReport(
        syntax_passed=True,
        lint_passed=True,
        diagnostics=[Diagnostic(tool="stub", passed=True)],
    )


def failing_report():
    return VerificationReport(
        syntax_passed=False,
        lint_passed=False,
        diagnostics=[Diagnostic(tool="yosys", passed=False, stderr="syntax error")],
    )


def retrieval_context_with_docs(docs):
    embedder = HashingEmbedder(dim=128)
    store = build_vector_store(docs, embedder.encode([doc.retrieval_text for doc in docs]))
    return RetrievalContext.from_store(store, embedder)


def requirements_json():
    return json.dumps(
        {
            "functionality": "stream input data through a FIFO into a processing core",
            "performance_target": "one sample per cycle",
            "io_interface": "valid-ready",
            "ppa_constraints": ["small area"],
            "clock_reset": "single clock, active-low reset",
            "assumptions": ["synchronous design"],
            "unknowns": ["exact data width"],
        }
    )


class AgenticIpReuseTests(unittest.TestCase):
    def test_cli_parser_accepts_run_args(self):
        args = build_parser().parse_args(
            [
                "run",
                "--prompt",
                "Build an accelerator",
                "--index",
                "indexes/smoke",
                "--embedder",
                "hash",
                "--target-hdl",
                "systemverilog",
                "--top-module",
                "dut",
                "--retrieve-k",
                "6",
                "--context-k",
                "3",
                "--max-repair-attempts",
                "1",
                "--base-url",
                "http://localhost:18000/v1",
                "--json-report",
                "runs/ip.json",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.prompt, "Build an accelerator")
        self.assertEqual(args.index, "indexes/smoke")
        self.assertEqual(args.target_hdl, "systemverilog")
        self.assertEqual(args.top_module, "dut")
        self.assertEqual(args.retrieve_k, 6)
        self.assertEqual(args.context_k, 3)
        self.assertEqual(args.max_repair_attempts, 1)
        self.assertEqual(args.base_url, "http://localhost:18000/v1")

    def test_missing_metadata_becomes_unknown_criteria(self):
        document = RtlDocument(
            doc_id="fifo",
            problem="Parameterized FIFO",
            solution="module fifo; endmodule",
            tags=["fifo"],
            metadata={"license": "MIT"},
        )
        candidate = candidate_from_hit(RetrievalHit(document=document, score=0.9, rerank_score=0.8))

        self.assertEqual(candidate.criteria["license"], "MIT")
        self.assertEqual(candidate.criteria["verification_status"], "unknown")
        self.assertEqual(candidate.criteria["synthesis_support"], "unknown")
        self.assertEqual(candidate.criteria["documentation_quality"], "unknown")

    def test_retrieval_and_ip_evaluation_flow_records_candidates(self):
        docs = [
            RtlDocument("fifo_ip", "valid-ready FIFO", "module fifo; endmodule", ["fifo"]),
            RtlDocument("if_ip", "valid-ready input interface", "module input_if; endmodule", ["interface"]),
            RtlDocument("core_ip", "processing core", "module core; endmodule", ["core"]),
        ]
        llm = FakeLlm(
            [
                requirements_json(),
                json.dumps(
                    {
                        "modules": [
                            {
                                "category": "Buffer / FIFO",
                                "name": "stream_fifo",
                                "purpose": "buffer input samples",
                                "required_interface": "valid-ready",
                                "performance_target": "one sample per cycle",
                                "ppa_constraints": ["small area"],
                                "reuse_query": "valid-ready FIFO",
                                "omitted_reason": None,
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "candidate_evaluations": [
                            {
                                "doc_id": "fifo_ip",
                                "criteria": {
                                    "function_match": "matches FIFO buffering",
                                    "interface_compatibility": "valid-ready",
                                    "configurability": "unknown",
                                    "verification_status": "unknown",
                                    "license": "unknown",
                                    "synthesis_support": "unknown",
                                    "documentation_quality": "unknown",
                                },
                                "rationale": "best FIFO candidate",
                            }
                        ],
                        "selected_doc_id": "fifo_ip",
                        "action": "configure",
                        "parameterization": {"depth": 4},
                        "integration_notes": "set depth to 4",
                        "rationale": "closest reusable IP",
                    }
                ),
                "```verilog\nmodule dut(input clk); endmodule\n```",
            ]
        )
        agent = AgenticIpReuseAgent(
            llm,
            retrieval_context_with_docs(docs),
            SequenceVerifier([passing_report()]),
            AgenticIpReuseConfig(retrieve_k=3, context_k=3),
        )

        result = agent.run("Build a streaming processor", top_module="dut")

        self.assertTrue(result.verification.passed)
        self.assertEqual(result.plan.decisions[0].selected_doc_id, "fifo_ip")
        self.assertEqual(result.plan.decisions[0].action, "configure")
        self.assertIn("fifo_ip", result.retrieval_traces[0]["doc_ids"])

    def test_fake_llm_flow_produces_full_report(self):
        docs = [RtlDocument("core_ip", "processing core", "module core; endmodule", ["core"])]
        llm = FakeLlm(
            [
                requirements_json(),
                json.dumps(
                    {
                        "modules": [
                            {
                                "category": "Processing Core",
                                "name": "core",
                                "purpose": "process samples",
                                "required_interface": "valid-ready",
                                "performance_target": "one sample per cycle",
                                "ppa_constraints": [],
                                "reuse_query": "processing core",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "candidate_evaluations": [],
                        "selected_doc_id": "core_ip",
                        "action": "adapt",
                        "parameterization": {},
                        "integration_notes": "wrap core with valid-ready ports",
                        "rationale": "usable behavior",
                    }
                ),
                "```verilog\nmodule dut(input clk); endmodule\n```",
            ]
        )
        agent = AgenticIpReuseAgent(
            llm,
            retrieval_context_with_docs(docs),
            SequenceVerifier([passing_report()]),
        )

        result = agent.run("Build a core", top_module="dut")
        report = json.loads(dumps_result(result))

        self.assertIn("requirements", report)
        self.assertIn("modules", report)
        self.assertIn("ip_reuse_decisions", report)
        self.assertIn("module dut", report["rtl"])
        self.assertTrue(report["verification"]["syntax_passed"])
        self.assertEqual(report["ip_reuse_decisions"][0]["selected_doc_id"], "core_ip")

    def test_repair_loop_uses_diagnostics_until_verification_passes(self):
        docs = [RtlDocument("core_ip", "processing core", "module core; endmodule", ["core"])]
        llm = FakeLlm(
            [
                requirements_json(),
                json.dumps(
                    {
                        "modules": [
                            {
                                "category": "Processing Core",
                                "name": "core",
                                "purpose": "process samples",
                                "required_interface": "plain ports",
                                "performance_target": "unknown",
                                "ppa_constraints": [],
                                "reuse_query": "processing core",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "candidate_evaluations": [],
                        "selected_doc_id": "core_ip",
                        "action": "reuse",
                        "parameterization": {},
                        "integration_notes": "direct reuse",
                        "rationale": "simple match",
                    }
                ),
                "```verilog\nmodule dut(input clk)\nendmodule\n```",
                "```verilog\nmodule dut(input clk); endmodule\n```",
            ]
        )
        verifier = SequenceVerifier([failing_report(), passing_report()])
        agent = AgenticIpReuseAgent(
            llm,
            retrieval_context_with_docs(docs),
            verifier,
            AgenticIpReuseConfig(max_repair_attempts=2),
        )

        result = agent.run("Build a core", top_module="dut")

        self.assertTrue(result.verification.passed)
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(len(verifier.calls), 2)
        self.assertIn("syntax error", llm.prompts[-1])


if __name__ == "__main__":
    unittest.main()
