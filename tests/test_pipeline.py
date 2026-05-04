from pathlib import Path
import tempfile
import unittest

from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.llm import StubLlmClient
from rag_rtl.pipeline import RagRtlPipeline
from rag_rtl.types import Diagnostic, RtlDocument, RtlTask, VerificationReport
from rag_rtl.vector_store import build_vector_store


class PassingVerifier:
    def verify(self, rtl, top_module=None):
        return VerificationReport(
            syntax_passed=True,
            lint_passed=True,
            diagnostics=[Diagnostic(tool="stub", passed=True)],
        )


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_with_stub_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs = [RtlDocument("invert-doc", "Design inverter", "module invert(input i, output o); assign o = ~i; endmodule")]
            embedder = HashingEmbedder(dim=128)
            store = build_vector_store(docs, embedder.encode([docs[0].retrieval_text]))
            pipeline = RagRtlPipeline(
                store=store,
                embedder=embedder,
                llm_client=StubLlmClient("module invert(input i, output o); assign o = ~i; endmodule"),
                verifier=PassingVerifier(),
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
            )
            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=0))
            self.assertTrue(response.verification.passed)
            self.assertIn("module invert", response.rtl)
            self.assertEqual(response.retrieved_doc_ids, ["invert-doc"])


if __name__ == "__main__":
    unittest.main()
