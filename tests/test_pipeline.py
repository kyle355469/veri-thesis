import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rag_rtl.config import CacheConfig, RuntimeConfig, ToolCallingConfig
from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.llm import StubLlmClient
from rag_rtl.pipeline import RagRtlPipeline
from rag_rtl.reporting import build_latest_report
from rag_rtl.types import Diagnostic, RtlDocument, RtlTask, VerificationReport
from rag_rtl.vector_store import build_vector_store


class PassingVerifier:
    def verify(self, rtl, top_module=None):
        return VerificationReport(
            syntax_passed=True,
            lint_passed=True,
            diagnostics=[Diagnostic(tool="stub", passed=True)],
        )


class FailingVerifier:
    def verify(self, rtl, top_module=None):
        return VerificationReport(
            syntax_passed=False,
            lint_passed=False,
            diagnostics=[Diagnostic(tool="stub", passed=False, stderr="syntax failed")],
        )


class CountingVerifier:
    def __init__(self):
        self.calls = 0

    def verify(self, rtl, top_module=None):
        self.calls += 1
        return VerificationReport(
            syntax_passed=True,
            lint_passed=True,
            diagnostics=[Diagnostic(tool="stub", passed=True)],
        )


class RawTextLlm:
    def __init__(self, text):
        self.text = text

    def complete(self, prompt, temperature=0.1, max_tokens=2048):
        return self.text


class DictEmbedder:
    dim = 2

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def empty_store():
    return build_vector_store([], np.zeros((0, 2), dtype=np.float32))


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
            self.assertIn("llm_generation_attempt", [item["action"] for item in response.llm_actions])
            self.assertIn("rtl_extracted", [item["action"] for item in response.llm_actions])

            report = build_latest_report(response)
            self.assertEqual(report["summary"]["passed"], True)
            self.assertEqual(report["task"]["prompt"], "Build an inverter")
            self.assertEqual(report["rtl"]["code"], response.rtl)
            self.assertEqual(report["llm_actions"], response.llm_actions)

    def test_pipeline_accepts_structured_runtime_config(self):
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
                cache_config=CacheConfig(path=tmp_path / "cache.json", mode="direct"),
                runtime_config=RuntimeConfig(
                    monitor_path=tmp_path / "monitor.jsonl",
                    failed_log_path=tmp_path / "failed.jsonl",
                ),
                tool_config=ToolCallingConfig(enabled=False),
            )

            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=0))

            self.assertTrue(response.verification.passed)
            self.assertTrue((tmp_path / "cache.json").exists())
            self.assertTrue((tmp_path / "monitor.jsonl").exists())

    def test_pipeline_only_caches_verified_rtl_and_logs_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedder = HashingEmbedder(dim=128)
            docs = [RtlDocument("invert-doc", "Design inverter", "module invert(); endmodule")]
            store = build_vector_store(docs, embedder.encode([docs[0].retrieval_text]))
            cache_path = tmp_path / "cache.json"
            failed_log = tmp_path / "failed.jsonl"
            pipeline = RagRtlPipeline(
                store=store,
                embedder=embedder,
                llm_client=StubLlmClient("module broken"),
                verifier=FailingVerifier(),
                cache_path=cache_path,
                monitor_path=tmp_path / "monitor.jsonl",
                failed_log_path=failed_log,
            )

            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=0))

            self.assertFalse(response.verification.passed)
            self.assertFalse(cache_path.exists())
            records = [json.loads(line) for line in failed_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["prompt"], "Build an inverter")
            self.assertTrue(records[0]["final_attempt"])

    def test_pipeline_reports_extraction_failure_without_running_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedder = HashingEmbedder(dim=128)
            store = empty_store()
            verifier = CountingVerifier()
            pipeline = RagRtlPipeline(
                store=store,
                embedder=embedder,
                llm_client=RawTextLlm("I cannot provide code for this request."),
                verifier=verifier,
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                failed_log_path=tmp_path / "failed.jsonl",
                cache_mode="direct",
            )

            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=0))

            self.assertEqual(response.rtl, "")
            self.assertEqual(verifier.calls, 0)
            self.assertFalse(response.verification.passed)
            self.assertEqual(response.verification.diagnostics[0].tool, "rtl_extraction")
            self.assertIn("No RTL code was extracted", response.verification.diagnostics[0].stderr)

    def test_evidence_range_cache_match_is_prompt_evidence_not_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            similar = [0.90, math.sqrt(1.0 - 0.90**2)]
            embedder = DictEmbedder(
                {
                    "adder old": [1.0, 0.0],
                    "adder similar": similar,
                }
            )
            llm = StubLlmClient("module fresh; endmodule")
            pipeline = RagRtlPipeline(
                store=empty_store(),
                embedder=embedder,
                llm_client=llm,
                verifier=PassingVerifier(),
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                cache_mode="keywords",
            )
            pipeline.cache.put("adder old", "module old; endmodule")

            response = pipeline.run(RtlTask(prompt="adder similar", max_repair_attempts=0))

            self.assertEqual(response.cache_source, "history_evidence")
            self.assertEqual(len(llm.prompts), 1)
            self.assertGreaterEqual(len(llm.keyword_prompts), 2)
            self.assertIn("### Semantic History Evidence", llm.prompts[0])
            self.assertIn("module old; endmodule", llm.prompts[0])

    def test_response_reports_no_best_history_match_without_keyword_candidate_and_verbose_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedder = DictEmbedder(
                {
                    "fifo old": [1.0, 0.0],
                    "adder new": [1.0, 0.0],
                }
            )
            pipeline = RagRtlPipeline(
                store=empty_store(),
                embedder=embedder,
                llm_client=StubLlmClient("module adder; endmodule"),
                verifier=PassingVerifier(),
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                cache_mode="keywords",
                verbose_generation=True,
            )
            pipeline.cache.put("fifo old", "module fifo; endmodule")

            with contextlib.redirect_stdout(io.StringIO()):
                response = pipeline.run(RtlTask(prompt="adder new", max_repair_attempts=0))

            self.assertEqual(response.prompt, "adder new")
            self.assertIsNone(response.metadata["best_history_match"])
            events = [json.loads(line)["event"] for line in (tmp_path / "monitor.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("verbose_raw_model_text", events)
            self.assertIn("verbose_extracted_rtl", events)


if __name__ == "__main__":
    unittest.main()
