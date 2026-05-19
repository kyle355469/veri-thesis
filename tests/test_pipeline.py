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
from rag_rtl.prompting import build_generation_prompt
from rag_rtl.reporting import build_latest_report
from rag_rtl.types import Diagnostic, RetrievalHit, RtlDocument, RtlTask, VerificationReport
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


class FailingOnceVerifier:
    def __init__(self):
        self.calls = 0

    def verify(self, rtl, top_module=None):
        self.calls += 1
        if self.calls == 1:
            return VerificationReport(
                syntax_passed=False,
                lint_passed=False,
                diagnostics=[
                    Diagnostic(
                        tool="stub",
                        passed=False,
                        stdout="lint stdout",
                        stderr="syntax failed",
                        returncode=1,
                    )
                ],
            )
        return VerificationReport(
            syntax_passed=True,
            lint_passed=True,
            diagnostics=[Diagnostic(tool="stub", passed=True)],
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


class SequencedRawLlm:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt, temperature=0.1, max_tokens=2048):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class DictEmbedder:
    dim = 2

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def empty_store():
    return build_vector_store([], np.zeros((0, 2), dtype=np.float32))


class PipelineTests(unittest.TestCase):
    def test_model_prompt_profile_omits_rag_history_and_tool_sections(self):
        prompt = build_generation_prompt(
            RtlTask(
                prompt="Build module TopModule(input a, output y).",
                constraints=["Return complete RTL."],
                prompt_profile="model",
            ),
            hits=[],
        )

        self.assertIn("### Coding Problem", prompt)
        self.assertIn("Return complete RTL.", prompt)
        self.assertNotIn("### Retrieved Context", prompt)
        self.assertNotIn("### Semantic History Evidence", prompt)
        self.assertNotIn("### Verification Diagnostics", prompt)
        self.assertNotIn("If tool calling is available", prompt)

    def test_tool_prompt_profile_keeps_tools_but_omits_rag_history_sections(self):
        prompt = build_generation_prompt(
            RtlTask(prompt="Build an inverter.", prompt_profile="tool"),
            hits=[],
        )

        self.assertIn("If tool calling is available", prompt)
        self.assertNotIn("### Retrieved Context", prompt)
        self.assertNotIn("### Semantic History Evidence", prompt)

    def test_rag_prompt_profile_keeps_retrieval_but_omits_tool_instructions(self):
        hit = RetrievalHit(
            document=RtlDocument(
                "invert-doc",
                "Design inverter",
                "module invert(input i, output o); assign o = ~i; endmodule",
            ),
            score=0.9,
        )

        prompt = build_generation_prompt(
            RtlTask(prompt="Build an inverter.", prompt_profile="rag"),
            hits=[hit],
        )

        self.assertIn("### Retrieved Context", prompt)
        self.assertIn("invert-doc", prompt)
        self.assertIn("### Semantic History Evidence", prompt)
        self.assertNotIn("If tool calling is available", prompt)

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

    def test_pipeline_none_cache_mode_does_not_save_successful_rtl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs = [RtlDocument("invert-doc", "Design inverter", "module invert(input i, output o); assign o = ~i; endmodule")]
            embedder = HashingEmbedder(dim=128)
            store = build_vector_store(docs, embedder.encode([docs[0].retrieval_text]))
            cache_path = tmp_path / "cache.json"
            pipeline = RagRtlPipeline(
                store=store,
                embedder=embedder,
                llm_client=StubLlmClient("module invert(input i, output o); assign o = ~i; endmodule"),
                verifier=PassingVerifier(),
                cache_config=CacheConfig(path=cache_path, mode="none"),
                runtime_config=RuntimeConfig(
                    monitor_path=tmp_path / "monitor.jsonl",
                    failed_log_path=tmp_path / "failed.jsonl",
                ),
            )

            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=0))

            self.assertTrue(response.verification.passed)
            self.assertFalse(cache_path.exists())
            self.assertEqual(response.metadata["cache_decision"]["decision"], "disabled")

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

    def test_extraction_failure_uses_emergency_code_only_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedder = HashingEmbedder(dim=128)
            llm = SequencedRawLlm(
                [
                    "I will reason about the module instead of returning code.",
                    "```verilog\nmodule invert(input i, output o); assign o = ~i; endmodule\n```",
                ]
            )
            verifier = CountingVerifier()
            pipeline = RagRtlPipeline(
                store=empty_store(),
                embedder=embedder,
                llm_client=llm,
                verifier=verifier,
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                failed_log_path=tmp_path / "failed.jsonl",
                cache_mode="direct",
            )

            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=0))

            self.assertTrue(response.verification.passed)
            self.assertEqual(verifier.calls, 1)
            self.assertEqual(len(llm.prompts), 2)
            self.assertIn("did not contain a parsable fenced HDL code block", llm.prompts[1])
            self.assertIn("No reasoning. No explanation. No diagnostics.", llm.prompts[1])
            self.assertNotIn("### Retrieved Context", llm.prompts[1])
            self.assertNotIn("### Previous RTL", llm.prompts[1])
            self.assertIn("emergency_extraction_retry", [item["action"] for item in response.llm_actions])

    def test_extraction_retry_does_not_use_emergency_before_final_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedder = HashingEmbedder(dim=128)
            llm = SequencedRawLlm(
                [
                    "I will reason about the module instead of returning code.",
                    "```verilog\nmodule invert(input i, output o); assign o = ~i; endmodule\n```",
                ]
            )
            verifier = CountingVerifier()
            pipeline = RagRtlPipeline(
                store=empty_store(),
                embedder=embedder,
                llm_client=llm,
                verifier=verifier,
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                failed_log_path=tmp_path / "failed.jsonl",
                cache_mode="direct",
            )

            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=1))

            self.assertTrue(response.verification.passed)
            self.assertEqual(verifier.calls, 1)
            self.assertEqual(len(llm.prompts), 2)
            self.assertIn("did not contain a parsable fenced HDL code block", llm.prompts[1])
            self.assertIn("### Retrieved Context", llm.prompts[1])
            self.assertNotIn("emergency_extraction_retry", [item["action"] for item in response.llm_actions])

    def test_verification_retry_uses_previous_rtl_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedder = HashingEmbedder(dim=128)
            bad_rtl = "module invert(input i, output o); assign o = i; endmodule"
            fixed_rtl = "module invert(input i, output o); assign o = ~i; endmodule"
            llm = SequencedRawLlm(
                [
                    f"```verilog\n{bad_rtl}\n```",
                    f"```verilog\n{fixed_rtl}\n```",
                ]
            )
            pipeline = RagRtlPipeline(
                store=empty_store(),
                embedder=embedder,
                llm_client=llm,
                verifier=FailingOnceVerifier(),
                cache_path=tmp_path / "cache.json",
                monitor_path=tmp_path / "monitor.jsonl",
                failed_log_path=tmp_path / "failed.jsonl",
                cache_mode="direct",
            )

            response = pipeline.run(RtlTask(prompt="Build an inverter", max_repair_attempts=1))

            self.assertTrue(response.verification.passed)
            self.assertEqual(len(llm.prompts), 2)
            self.assertIn("### Previous RTL", llm.prompts[1])
            self.assertIn(bad_rtl, llm.prompts[1])
            self.assertIn("lint stdout", llm.prompts[1])
            self.assertIn("syntax failed", llm.prompts[1])
            self.assertIn("Repair the previous RTL using the diagnostics", llm.prompts[1])
            retry_actions = [
                item for item in response.llm_actions
                if item["action"] == "llm_generation_attempt" and item["attempt"] == 1
            ]
            self.assertEqual(retry_actions[0]["retry_kind"], "verification")

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
