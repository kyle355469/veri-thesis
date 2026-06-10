import json
import tempfile
import unittest
from pathlib import Path

from agentic_ip_reuse.agent import (
    AgenticIpReuseAgent,
    AgenticIpReuseConfig,
    _dependency_order,
    _manifest_validation_errors,
    _recursive_decomposition_validation_errors,
    candidate_from_hit,
    dumps_result,
)
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


def large_spec_manifest():
    return {
        "system_summary": "A top wrapper around a leaf datapath.",
        "clocks": ["clk is the only clock"],
        "resets": [],
        "parameters": [],
        "shared_constraints": ["combinational output"],
        "assumptions": [],
        "unknowns": [],
        "top_module": {
            "name": "TopModule",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1", "description": "clock"},
                {"name": "y", "direction": "output", "width": "1", "description": "result"},
            ],
            "instances": [
                {
                    "module": "leaf",
                    "instance_name": "u_leaf",
                    "connections": {"clk": "clk", "y": "y"},
                }
            ],
        },
        "modules": [
            {
                "name": "leaf",
                "category": "Processing Core",
                "purpose": "drive the result",
                "ports": [
                    {"name": "clk", "direction": "input", "width": "1", "description": "clock"},
                    {"name": "y", "direction": "output", "width": "1", "description": "result"},
                ],
                "behavioral_requirements": ["drive y low"],
                "dependencies": [],
                "reuse_query": "constant output leaf",
            }
        ],
    }


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
                "--decomposition-mode",
                "chunking",
                "--base-url",
                "http://localhost:18000/v1",
                "--json-report",
                "runs/ip.json",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.decomposition_mode, "chunking")
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

    def test_exact_large_spec_threshold_keeps_legacy_flow(self):
        llm = FakeLlm(
            [
                requirements_json(),
                json.dumps(
                    {
                        "modules": [
                            {
                                "category": "Processing Core",
                                "name": "core",
                                "purpose": "process",
                                "required_interface": "plain ports",
                                "performance_target": "unknown",
                                "ppa_constraints": [],
                                "reuse_query": "core",
                            }
                        ]
                    }
                ),
                "```verilog\nmodule TopModule; endmodule\n```",
            ]
        )
        agent = AgenticIpReuseAgent(
            llm,
            retrieval_context_with_docs([]),
            SequenceVerifier([passing_report()]),
        )

        result = agent.run("x" * 15000, top_module="TopModule")

        self.assertIsNone(result.large_spec_manifest)
        self.assertIn("Extract system-level requirements", llm.prompts[0])

    def test_large_spec_staged_generation_writes_workspace_artifacts(self):
        llm = FakeLlm(
            [
                json.dumps(large_spec_manifest()),
                "```verilog\nmodule leaf(input clk, output y); assign y = 1'b0; endmodule\n```",
                (
                    "```verilog\nmodule TopModule(input clk, output y); "
                    "leaf u_leaf(.clk(clk), .y(y)); endmodule\n```"
                ),
            ]
        )
        verifier = SequenceVerifier([passing_report(), passing_report()])
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                verifier,
                AgenticIpReuseConfig(recursive_decomposition=False, large_spec_threshold_chars=15000),
            )

            result = agent.run("x" * 15001, top_module="TopModule", workspace_dir=workspace)

            self.assertTrue(result.verification.passed)
            self.assertEqual(result.large_spec_manifest["top_module"]["name"], "TopModule")
            self.assertIn("module leaf", result.rtl)
            self.assertIn("module TopModule", result.rtl)
            self.assertTrue((workspace / "original_spec.txt").exists())
            self.assertTrue((workspace / "spec_manifest.json").exists())
            self.assertTrue((workspace / "index.txt").exists())
            self.assertTrue((workspace / "specs" / "leaf.txt").exists())
            self.assertTrue((workspace / "rtl" / "leaf.v").exists())
            self.assertTrue((workspace / "rtl" / "TopModule.v").exists())
            self.assertTrue((workspace / "combined" / "TopModule.sv").exists())
            report = json.loads(dumps_result(result))
            self.assertEqual(report["workspace_dir"], str(workspace.resolve()))
            self.assertIn("combined_rtl", report["artifacts"])
            self.assertEqual(report["module_generation"][0]["module"], "leaf")
            module_prompt = next(prompt for prompt in llm.prompts if "Generate exactly one self-contained" in prompt)
            self.assertNotIn("x" * 1000, module_prompt)

    def test_invalid_full_split_falls_back_to_chunk_merge(self):
        llm = FakeLlm(
            [
                "{}",
                json.dumps({"system_summary": "partial"}),
                json.dumps(large_spec_manifest()),
                "```verilog\nmodule leaf(input clk, output y); assign y = 1'b0; endmodule\n```",
                "```verilog\nmodule TopModule(input clk, output y); leaf u_leaf(.clk(clk), .y(y)); endmodule\n```",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([passing_report(), passing_report()]),
                AgenticIpReuseConfig(
                    large_spec_threshold_chars=10,
                    large_spec_chunk_chars=1000,
                    recursive_decomposition=False,
                ),
            )

            result = agent.run("large specification text", top_module="TopModule", workspace_dir=tmp)

            self.assertTrue(result.verification.passed)
            self.assertTrue(any("Extract one-layer implementation facts from chunk" in prompt for prompt in llm.prompts))
            self.assertTrue(any("Merge partial large-hardware-specification" in prompt for prompt in llm.prompts))
            self.assertTrue((Path(tmp) / "errors" / "full_split_validation.json").exists())

    def test_chunking_decomposition_mode_skips_full_split(self):
        llm = FakeLlm(
            [
                json.dumps({"system_summary": "partial"}),
                json.dumps(large_spec_manifest()),
                "```verilog\nmodule leaf(input clk, output y); assign y = 1'b0; endmodule\n```",
                "```verilog\nmodule TopModule(input clk, output y); leaf u_leaf(.clk(clk), .y(y)); endmodule\n```",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([passing_report(), passing_report()]),
                AgenticIpReuseConfig(
                    decomposition_mode="chunking",
                    large_spec_threshold_chars=1000,
                    large_spec_chunk_chars=1000,
                    recursive_decomposition=False,
                ),
            )

            result = agent.run("short spec", top_module="TopModule", workspace_dir=tmp)

            self.assertTrue(result.verification.passed)
            self.assertIn("Extract one-layer implementation facts from chunk", llm.prompts[0])
            self.assertFalse(any("Partition this large hardware design specification" in prompt for prompt in llm.prompts))
            self.assertFalse((Path(tmp) / "errors" / "full_split_response.txt").exists())

    def test_chunking_decomposition_mode_retries_invalid_chunk_response(self):
        llm = FakeLlm(
            [
                "not json",
                json.dumps({"system_summary": "partial"}),
                json.dumps(large_spec_manifest()),
                "```verilog\nmodule leaf(input clk, output y); assign y = 1'b0; endmodule\n```",
                "```verilog\nmodule TopModule(input clk, output y); leaf u_leaf(.clk(clk), .y(y)); endmodule\n```",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([passing_report(), passing_report()]),
                AgenticIpReuseConfig(
                    decomposition_mode="chunking",
                    large_spec_chunk_chars=1000,
                    max_generation_retries=1,
                    recursive_decomposition=False,
                ),
            )

            result = agent.run("short spec", top_module="TopModule", workspace_dir=tmp)

            self.assertTrue(result.verification.passed)
            self.assertTrue((Path(tmp) / "errors" / "chunk_001_response.txt").exists())
            self.assertTrue((Path(tmp) / "errors" / "chunk_001_retry001_response.txt").exists())
            self.assertTrue(any(trace.stage == "large_spec_chunk:1:retry1" for trace in result.llm_traces))

    def test_chunking_decomposition_mode_retries_invalid_merge_response(self):
        llm = FakeLlm(
            [
                json.dumps({"system_summary": "partial"}),
                "not json",
                json.dumps(large_spec_manifest()),
                "```verilog\nmodule leaf(input clk, output y); assign y = 1'b0; endmodule\n```",
                "```verilog\nmodule TopModule(input clk, output y); leaf u_leaf(.clk(clk), .y(y)); endmodule\n```",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([passing_report(), passing_report()]),
                AgenticIpReuseConfig(
                    decomposition_mode="chunking",
                    large_spec_chunk_chars=1000,
                    max_generation_retries=1,
                    recursive_decomposition=False,
                ),
            )

            result = agent.run("short spec", top_module="TopModule", workspace_dir=tmp)

            self.assertTrue(result.verification.passed)
            self.assertTrue((Path(tmp) / "errors" / "merge_01_001_response.txt").exists())
            self.assertTrue((Path(tmp) / "errors" / "merge_01_001_retry001_response.txt").exists())
            self.assertTrue(any(trace.stage == "large_spec_merge:1:1:retry1" for trace in result.llm_traces))

    def test_invalid_chunk_merge_and_correction_fail_clearly(self):
        llm = FakeLlm(["{}", "{}", "{}", "{}"])
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([]),
                AgenticIpReuseConfig(large_spec_threshold_chars=10, large_spec_chunk_chars=1000),
            )

            with self.assertRaisesRegex(RuntimeError, "manifest validation failed"):
                agent.run("large specification text", top_module="TopModule", workspace_dir=tmp)

            self.assertTrue((Path(tmp) / "errors" / "corrected_split_response.txt").exists())
            self.assertTrue((Path(tmp) / "errors" / "manifest_validation.json").exists())

    def test_manifest_validation_rejects_names_dependencies_cycles_and_top_mismatch(self):
        invalid_name = large_spec_manifest()
        invalid_name["modules"][0]["name"] = "bad-name"
        self.assertTrue(any("invalid submodule name" in item for item in _manifest_validation_errors(invalid_name, "TopModule")))

        unknown_dependency = large_spec_manifest()
        unknown_dependency["modules"][0]["dependencies"] = ["missing"]
        self.assertTrue(any("unknown dependency" in item for item in _manifest_validation_errors(unknown_dependency, "TopModule")))

        cycle = large_spec_manifest()
        cycle["modules"].append(
            {
                **cycle["modules"][0],
                "name": "other",
                "dependencies": ["leaf"],
            }
        )
        cycle["modules"][0]["dependencies"] = ["other"]
        self.assertTrue(any("dependency cycle" in item for item in _manifest_validation_errors(cycle, "TopModule")))

        mismatch = large_spec_manifest()
        self.assertTrue(any("must exactly match" in item for item in _manifest_validation_errors(mismatch, "dut")))

        parent = large_spec_manifest()["modules"][0]
        no_progress = {
            "decision": "decompose",
            "reason": "bad recursive split",
            "parent_module": {
                "name": parent["name"],
                "ports": parent["ports"],
                "instances": [],
            },
            "children": [parent],
        }
        self.assertTrue(
            any(
                "conflict" in item or "no progress" in item
                for item in _recursive_decomposition_validation_errors(
                    no_progress,
                    parent,
                    existing_names={parent["name"]},
                    ancestors=["TopModule"],
                )
            )
        )

    def test_large_spec_repairs_one_module_before_top_integration(self):
        llm = FakeLlm(
            [
                json.dumps(large_spec_manifest()),
                "```verilog\nmodule leaf(input clk, output y) assign y = clk; endmodule\n```",
                "```verilog\nmodule leaf(input clk, output y); assign y = clk; endmodule\n```",
                "```verilog\nmodule TopModule(input clk, output y); leaf u_leaf(.clk(clk), .y(y)); endmodule\n```",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([failing_report(), passing_report(), passing_report()]),
                AgenticIpReuseConfig(
                    large_spec_threshold_chars=10,
                    max_repair_attempts=1,
                    recursive_decomposition=False,
                ),
            )

            result = agent.run("large specification text", top_module="TopModule", workspace_dir=tmp)

            self.assertTrue(result.verification.passed)
            self.assertEqual(result.repair_attempts, 1)
            self.assertEqual(result.module_generation[0]["repair_attempts"], 1)
            self.assertTrue(any("Repair exactly one verilog module" in prompt for prompt in llm.prompts))

    def test_dependency_order_places_dependencies_before_consumers(self):
        manifest = large_spec_manifest()
        parent = {
            **manifest["modules"][0],
            "name": "parent",
            "purpose": "instantiate leaf",
            "behavioral_requirements": ["forward leaf output"],
            "dependencies": ["leaf"],
        }
        manifest["modules"] = [parent, manifest["modules"][0]]

        self.assertEqual(_dependency_order(manifest), ["leaf", "parent"])

    def test_recursive_decomposition_builds_children_before_parent_wrapper(self):
        root_manifest = {
            "system_summary": "A CPU with one ALU.",
            "clocks": [],
            "resets": [],
            "parameters": [],
            "shared_constraints": [],
            "assumptions": [],
            "unknowns": [],
            "top_module": {
                "name": "cpu",
                "ports": [
                    {"name": "a", "direction": "input", "width": "8", "description": "operand"},
                    {"name": "b", "direction": "input", "width": "8", "description": "operand"},
                    {"name": "op", "direction": "input", "width": "1", "description": "operation"},
                    {"name": "y", "direction": "output", "width": "8", "description": "result"},
                ],
                "instances": [
                    {
                        "module": "cpu_alu",
                        "instance_name": "u_alu",
                        "connections": {"a": "a", "b": "b", "op": "op", "y": "y"},
                    }
                ],
            },
            "modules": [
                {
                    "name": "cpu_alu",
                    "category": "Processing Core",
                    "purpose": "perform arithmetic and logic",
                    "ports": [
                        {"name": "a", "direction": "input", "width": "8", "description": "operand"},
                        {"name": "b", "direction": "input", "width": "8", "description": "operand"},
                        {"name": "op", "direction": "input", "width": "1", "description": "operation"},
                        {"name": "y", "direction": "output", "width": "8", "description": "result"},
                    ],
                    "behavioral_requirements": ["select add or xor using op"],
                    "dependencies": [],
                    "reuse_query": "8-bit ALU",
                }
            ],
        }
        alu_decomposition = {
            "decision": "decompose",
            "reason": "arithmetic and logic are meaningful child blocks",
            "parent_module": {
                "name": "cpu_alu",
                "ports": root_manifest["modules"][0]["ports"],
                "instances": [
                    {
                        "module": "cpu_alu_adder",
                        "instance_name": "u_adder",
                        "connections": {"a": "a", "b": "b", "y": "add_y"},
                    },
                    {
                        "module": "cpu_alu_xor",
                        "instance_name": "u_xor",
                        "connections": {"a": "a", "b": "b", "y": "xor_y"},
                    },
                ],
            },
            "children": [
                {
                    "name": "cpu_alu_adder",
                    "category": "Processing Core",
                    "purpose": "add operands",
                    "ports": [
                        {"name": "a", "direction": "input", "width": "8", "description": "operand"},
                        {"name": "b", "direction": "input", "width": "8", "description": "operand"},
                        {"name": "y", "direction": "output", "width": "8", "description": "sum"},
                    ],
                    "behavioral_requirements": ["y equals a plus b"],
                    "dependencies": [],
                    "reuse_query": "8-bit adder",
                },
                {
                    "name": "cpu_alu_xor",
                    "category": "Processing Core",
                    "purpose": "xor operands",
                    "ports": [
                        {"name": "a", "direction": "input", "width": "8", "description": "operand"},
                        {"name": "b", "direction": "input", "width": "8", "description": "operand"},
                        {"name": "y", "direction": "output", "width": "8", "description": "xor result"},
                    ],
                    "behavioral_requirements": ["y equals a xor b"],
                    "dependencies": [],
                    "reuse_query": "8-bit xor",
                },
            ],
        }
        adder_leaf = {
            "decision": "leaf",
            "reason": "simple focused adder",
            "parent_module": {
                "name": "cpu_alu_adder",
                "ports": alu_decomposition["children"][0]["ports"],
                "instances": [],
            },
            "children": [],
        }
        xor_leaf = {
            "decision": "leaf",
            "reason": "simple focused xor",
            "parent_module": {
                "name": "cpu_alu_xor",
                "ports": alu_decomposition["children"][1]["ports"],
                "instances": [],
            },
            "children": [],
        }
        llm = FakeLlm(
            [
                json.dumps(root_manifest),
                json.dumps(alu_decomposition),
                json.dumps(adder_leaf),
                json.dumps(xor_leaf),
                "```verilog\nmodule cpu_alu_adder(input [7:0] a,b, output [7:0] y); assign y=a+b; endmodule\n```",
                "",  # testbench generation for cpu_alu_adder (empty → skip)
                json.dumps({"action": "new", "rationale": "generate fresh"}),  # ip_evaluation: xor vs live adder
                "```verilog\nmodule cpu_alu_xor(input [7:0] a,b, output [7:0] y); assign y=a^b; endmodule\n```",
                "",  # testbench generation for cpu_alu_xor (empty → skip)
                (
                    "```verilog\nmodule cpu_alu(input [7:0] a,b,input op,output [7:0] y); "
                    "wire [7:0] add_y,xor_y; cpu_alu_adder u_adder(.a(a),.b(b),.y(add_y)); "
                    "cpu_alu_xor u_xor(.a(a),.b(b),.y(xor_y)); assign y=op?xor_y:add_y; endmodule\n```"
                ),
                (
                    "```verilog\nmodule cpu(input [7:0] a,b,input op,output [7:0] y); "
                    "cpu_alu u_alu(.a(a),.b(b),.op(op),.y(y)); endmodule\n```"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            generation_before_tree_saved = []

            def stage_callback(event):
                if event["status"] != "running":
                    return
                if event["stage"] in {
                    "recursive_rtl_generation",
                    "module_generation",
                    "recursive_wrapper_generation",
                    "top_integration",
                }:
                    if not (Path(tmp) / "decomposition_tree.json").exists():
                        generation_before_tree_saved.append(event["stage"])

            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([passing_report(), passing_report(), passing_report(), passing_report()]),
                AgenticIpReuseConfig(large_spec_threshold_chars=10, recursive_decomposition=True),
                stage_callback=stage_callback,
            )

            result = agent.run("large CPU specification", top_module="cpu", workspace_dir=tmp)

            self.assertTrue(result.verification.passed)
            self.assertEqual(generation_before_tree_saved, [])
            self.assertEqual([item["module"] for item in result.module_generation], [
                "cpu_alu_adder",
                "cpu_alu_xor",
                "cpu_alu",
            ])
            first_generation_prompt = next(
                index
                for index, prompt in enumerate(llm.prompts)
                if "Generate exactly one self-contained" in prompt
            )
            self.assertTrue(
                all("Decide whether one hardware module" in prompt for prompt in llm.prompts[1:first_generation_prompt])
            )
            self.assertEqual(result.decomposition_tree["children"][0]["kind"], "composite")
            self.assertEqual(
                [child["module"] for child in result.decomposition_tree["children"][0]["children"]],
                ["cpu_alu_adder", "cpu_alu_xor"],
            )
            self.assertTrue((Path(tmp) / "decompositions" / "cpu_alu.json").exists())
            self.assertTrue((Path(tmp) / "indexes" / "cpu_alu.txt").exists())

    def test_recursive_max_depth_forces_leaf_generation(self):
        llm = FakeLlm(
            [
                json.dumps(large_spec_manifest()),
                "```verilog\nmodule leaf(input clk, output y); assign y = clk; endmodule\n```",
                "",  # testbench generation for leaf (empty → skip)
                "```verilog\nmodule TopModule(input clk, output y); leaf u_leaf(.clk(clk), .y(y)); endmodule\n```",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgenticIpReuseAgent(
                llm,
                retrieval_context_with_docs([]),
                SequenceVerifier([passing_report(), passing_report()]),
                AgenticIpReuseConfig(
                    large_spec_threshold_chars=10,
                    recursive_decomposition=True,
                    recursive_max_depth=1,
                ),
            )

            result = agent.run("large specification text", top_module="TopModule", workspace_dir=tmp)

            self.assertTrue(result.verification.passed)
            self.assertEqual(result.decomposition_tree["children"][0]["kind"], "leaf")
            self.assertFalse(any("Decide whether one hardware module" in prompt for prompt in llm.prompts))


if __name__ == "__main__":
    unittest.main()
