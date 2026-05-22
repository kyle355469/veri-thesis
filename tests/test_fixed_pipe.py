import tempfile
import unittest
from pathlib import Path

from rag_rtl.datapath import DatapathEdge, DatapathGraph, DatapathNode
from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.pipeline import FixedPipeRtlPipeline
from rag_rtl.types import Diagnostic, RtlDocument, RtlTask, VerificationReport
from rag_rtl.vector_store import build_vector_store


class PassingVerifier:
    def __init__(self):
        self.rtl_seen = []

    def verify(self, rtl, top_module=None):
        self.rtl_seen.append(rtl)
        return VerificationReport(
            syntax_passed=True,
            lint_passed=True,
            diagnostics=[Diagnostic(tool="stub", passed=True)],
        )


class SequencedVerifier:
    def __init__(self, reports):
        self.reports = list(reports)
        self.rtl_seen = []

    def verify(self, rtl, top_module=None):
        self.rtl_seen.append(rtl)
        return self.reports.pop(0)


class SequencedLlm:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt, temperature=0.1, max_tokens=2048):
        self.prompts.append(prompt)
        rtl = self.outputs.pop(0)
        return f"```verilog\n{rtl}\n```"


class FakeDatapathExtractor:
    def extract_document(self, document):
        return [
            DatapathGraph(
                graph_id=f"{document.doc_id}:and2",
                source_doc_id=document.doc_id,
                module="and2",
                nodes=[
                    DatapathNode("port:a", "port", "a", attrs={"direction": "input"}),
                    DatapathNode("port:b", "port", "b", attrs={"direction": "input"}),
                    DatapathNode("port:y", "port", "y", attrs={"direction": "output"}),
                ],
                edges=[
                    DatapathEdge(
                        "net:a",
                        "net:y",
                        "dependency",
                        attrs={"cell_type": "$and", "source_signal": "a", "target_signal": "y"},
                    ),
                    DatapathEdge(
                        "net:b",
                        "net:y",
                        "dependency",
                        attrs={"cell_type": "$and", "source_signal": "b", "target_signal": "y"},
                    ),
                ],
                operations={"$and": 1},
            )
        ]


class FixedPipeTests(unittest.TestCase):
    def test_fixed_pipe_generates_second_edition_from_graph_context(self):
        first_rtl = "module and2(input a, input b, output y); assign y = a & b; endmodule"
        second_rtl = "module and2(input a, input b, output y); wire n; assign n = a & b; assign y = n; endmodule"
        spec_docs = [RtlDocument("spec-and", "Design a 2-input and gate", first_rtl)]
        graph_docs = [
            RtlDocument(
                "graph-and",
                "Design a 2-input and gate",
                "datapath graph graph-and\nmodule and2\noperations $and:1\ndependencies\na -> y via $and\nb -> y via $and",
                tags=["datapath", "$and"],
            )
        ]
        embedder = HashingEmbedder(dim=128)
        spec_store = build_vector_store(spec_docs, embedder.encode([doc.retrieval_text for doc in spec_docs]))
        graph_store = build_vector_store(graph_docs, embedder.encode([doc.retrieval_text for doc in graph_docs]))
        llm = SequencedLlm([first_rtl, second_rtl])
        verifier = PassingVerifier()

        with tempfile.TemporaryDirectory() as tempdir:
            tmp_path = Path(tempdir)
            pipeline = FixedPipeRtlPipeline(
                spec_store=spec_store,
                code_structure_store=graph_store,
                embedder=embedder,
                llm_client=llm,
                verifier=verifier,
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                cache_mode="direct",
            )
            pipeline.datapath_extractor = FakeDatapathExtractor()

            response = pipeline.run(RtlTask(prompt="Design a 2-input and gate named and2.", max_repair_attempts=0))

        self.assertTrue(response.verification.passed)
        self.assertEqual(response.rtl, second_rtl)
        self.assertEqual(verifier.rtl_seen, [first_rtl, second_rtl])
        self.assertIn("graph-and", response.retrieved_doc_ids)
        self.assertEqual(response.metadata["first_edition_datapath"]["graph_count"], 1)
        self.assertEqual(response.metadata["second_edition"]["retrieved_doc_ids"], ["graph-and"])
        self.assertIn("### First-Edition Verified RTL", llm.prompts[1])
        self.assertIn("### First-Edition Datapath", llm.prompts[1])
        self.assertIn("### Retrieved Code-Structure Context", llm.prompts[1])
        self.assertIn("a -> y via $and", llm.prompts[1])

    def test_fixed_pipe_second_edition_retry_uses_repair_prompt(self):
        first_rtl = "module and2(input a, input b, output y); assign y = a & b; endmodule"
        bad_second_rtl = "module and2(input a, input b, output y); assign y = a | b; endmodule"
        fixed_second_rtl = "module and2(input a, input b, output y); assign y = a & b; endmodule"
        spec_docs = [RtlDocument("spec-and", "Design a 2-input and gate", first_rtl)]
        graph_docs = [
            RtlDocument(
                "graph-and",
                "Design a 2-input and gate",
                "datapath graph graph-and\nmodule and2\noperations $and:1\ndependencies\na -> y via $and\nb -> y via $and",
                tags=["datapath", "$and"],
            )
        ]
        embedder = HashingEmbedder(dim=128)
        spec_store = build_vector_store(spec_docs, embedder.encode([doc.retrieval_text for doc in spec_docs]))
        graph_store = build_vector_store(graph_docs, embedder.encode([doc.retrieval_text for doc in graph_docs]))
        llm = SequencedLlm([first_rtl, bad_second_rtl, fixed_second_rtl])
        verifier = SequencedVerifier(
            [
                VerificationReport(
                    syntax_passed=True,
                    lint_passed=True,
                    diagnostics=[Diagnostic(tool="stub", passed=True)],
                ),
                VerificationReport(
                    syntax_passed=False,
                    lint_passed=False,
                    diagnostics=[
                        Diagnostic(
                            tool="stub",
                            passed=False,
                            stdout="expected and gate",
                            stderr="behavior mismatch",
                            returncode=1,
                        )
                    ],
                ),
                VerificationReport(
                    syntax_passed=True,
                    lint_passed=True,
                    diagnostics=[Diagnostic(tool="stub", passed=True)],
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            tmp_path = Path(tempdir)
            pipeline = FixedPipeRtlPipeline(
                spec_store=spec_store,
                code_structure_store=graph_store,
                embedder=embedder,
                llm_client=llm,
                verifier=verifier,
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                cache_mode="direct",
                second_edition_repair_attempts=1,
            )
            pipeline.datapath_extractor = FakeDatapathExtractor()

            response = pipeline.run(RtlTask(prompt="Design a 2-input and gate named and2.", max_repair_attempts=0))

        self.assertTrue(response.verification.passed)
        self.assertEqual(response.rtl, fixed_second_rtl)
        self.assertEqual(verifier.rtl_seen, [first_rtl, bad_second_rtl, fixed_second_rtl])
        self.assertIn("### Previous Second-Edition RTL", llm.prompts[2])
        self.assertIn(bad_second_rtl, llm.prompts[2])
        self.assertIn("expected and gate", llm.prompts[2])
        self.assertIn("behavior mismatch", llm.prompts[2])
        self.assertIn("Repair the second-edition RTL using the diagnostics", llm.prompts[2])
        retry_actions = [
            item for item in response.llm_actions
            if item["action"] == "second_edition_generation_attempt" and item["attempt"] == 1
        ]
        self.assertEqual(retry_actions[0]["retry_kind"], "verification")

    def test_fixed_pipe_keeps_verified_first_edition_when_second_edition_fails(self):
        first_rtl = "module and2(input a, input b, output y); assign y = a & b; endmodule"
        bad_second_rtl = "module and2(input a, input b, output y); assign y = a | b; endmodule"
        spec_docs = [RtlDocument("spec-and", "Design a 2-input and gate", first_rtl)]
        graph_docs = [
            RtlDocument(
                "graph-and",
                "Design a 2-input and gate",
                "datapath graph graph-and\nmodule and2\noperations $and:1\ndependencies\na -> y via $and\nb -> y via $and",
                tags=["datapath", "$and"],
            )
        ]
        embedder = HashingEmbedder(dim=128)
        spec_store = build_vector_store(spec_docs, embedder.encode([doc.retrieval_text for doc in spec_docs]))
        graph_store = build_vector_store(graph_docs, embedder.encode([doc.retrieval_text for doc in graph_docs]))
        llm = SequencedLlm([first_rtl, bad_second_rtl])
        verifier = SequencedVerifier(
            [
                VerificationReport(
                    syntax_passed=True,
                    lint_passed=True,
                    diagnostics=[Diagnostic(tool="stub", passed=True)],
                ),
                VerificationReport(
                    syntax_passed=False,
                    lint_passed=False,
                    diagnostics=[
                        Diagnostic(
                            tool="stub",
                            passed=False,
                            stdout="expected and gate",
                            stderr="behavior mismatch",
                            returncode=1,
                        )
                    ],
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tempdir:
            tmp_path = Path(tempdir)
            pipeline = FixedPipeRtlPipeline(
                spec_store=spec_store,
                code_structure_store=graph_store,
                embedder=embedder,
                llm_client=llm,
                verifier=verifier,
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                cache_mode="direct",
                second_edition_repair_attempts=0,
            )
            pipeline.datapath_extractor = FakeDatapathExtractor()

            response = pipeline.run(RtlTask(prompt="Design a 2-input and gate named and2.", max_repair_attempts=0))

        self.assertTrue(response.verification.passed)
        self.assertEqual(response.rtl, first_rtl)
        self.assertEqual(verifier.rtl_seen, [first_rtl, bad_second_rtl])
        self.assertFalse(response.metadata["second_edition"]["passed"])
        self.assertEqual(response.metadata["selected_edition"], "first_edition")
        fallback_actions = [
            item for item in response.llm_actions
            if item["action"] == "fixed_pipe_keep_first_edition"
        ]
        self.assertEqual(len(fallback_actions), 1)


if __name__ == "__main__":
    unittest.main()
