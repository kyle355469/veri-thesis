import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.repair_cache import (
    DiagnosticSignature,
    RepairFixCache,
    diagnostic_keywords,
    normalize_diagnostics,
)

PINNOTFOUND_STDERR = (
    "%Error-PINNOTFOUND: t/x.sv:12:5: Pin not found: 'wb_clk_i'\n"
    "%Error: t/x.sv:40:1: Exiting due to 1 error(s)\n"
)
PINNOTFOUND_OTHER_TASK = "%Error-PINNOTFOUND: rtl/core.sv:9:3: Pin not found: 'core_clk'\n"


def diag(stderr):
    return [{"tool": "verilator", "passed": False, "stdout": "", "stderr": stderr}]


class NormalizationTests(unittest.TestCase):
    def test_strips_paths_identifiers_and_numbers(self):
        signature = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        self.assertIsNotNone(signature)
        self.assertIn("PINNOTFOUND", signature.error_codes)
        self.assertNotIn("t/x.sv", signature.text)
        self.assertNotIn("wb_clk_i", signature.text)
        self.assertIn("'<id>'", signature.text)
        self.assertIn("<n>", signature.text)

    def test_identifier_stripping_makes_signatures_recur_across_tasks(self):
        first = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        second = normalize_diagnostics(diag(PINNOTFOUND_OTHER_TASK))
        pin_lines_first = [line for line in first.text.splitlines() if "PINNOTFOUND" in line]
        pin_lines_second = [line for line in second.text.splitlines() if "PINNOTFOUND" in line]
        self.assertEqual(pin_lines_first, pin_lines_second)

    def test_no_diagnostic_lines_returns_none(self):
        self.assertIsNone(normalize_diagnostics(diag("clean compile, no problems")))
        self.assertIsNone(normalize_diagnostics([]))

    def test_keyword_extractor_returns_error_codes(self):
        signature = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        self.assertIn("pinnotfound", diagnostic_keywords(signature.text))


class RepairFixCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "repair_cache.json"

    def tearDown(self):
        self.tmp.cleanup()

    def make_cache(self, **kwargs):
        return RepairFixCache(embedder=HashingEmbedder(dim=256), path=self.path, **kwargs)

    def test_record_then_lookup_returns_hint_with_diff(self):
        cache = self.make_cache(evidence_threshold=0.5)
        signature = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        cache.record_fix(
            signature,
            "module top(input clk);\nendmodule\n",
            "module top(input wb_clk_i);\nendmodule\n",
            task_id="task_a",
            attempt=1,
        )
        hint = cache.lookup_hint(normalize_diagnostics(diag(PINNOTFOUND_OTHER_TASK)))
        self.assertIsNotNone(hint)
        self.assertIn("PINNOTFOUND", hint.text)
        self.assertIn("Previously verified fix", hint.text)
        self.assertIn("+module top(input wb_clk_i);", hint.text)
        stats = cache.stats()
        self.assertEqual(stats["puts"], 1)
        self.assertEqual(stats["hits"], 1)

    def test_below_evidence_threshold_returns_none(self):
        cache = self.make_cache(evidence_threshold=1.01, reuse_threshold=1.02)
        signature = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        cache.record_fix(signature, "module a;\nendmodule\n", "module b;\nendmodule\n")
        self.assertIsNone(cache.lookup_hint(signature))

    def test_unrelated_error_codes_do_not_match(self):
        cache = self.make_cache(evidence_threshold=0.1)
        signature = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        cache.record_fix(signature, "module a;\nendmodule\n", "module b;\nendmodule\n")
        moddup = normalize_diagnostics(
            diag("%Error-MODDUP: y.sv:3:1: Duplicate declaration of module: 'aes_core'\n")
        )
        self.assertIsNone(cache.lookup_hint(moddup))

    def test_none_signature_is_a_safe_noop(self):
        cache = self.make_cache()
        self.assertIsNone(cache.lookup_hint(None))
        cache.record_fix(None, "a", "b")
        self.assertEqual(cache.stats()["puts"], 0)

    def test_persistence_roundtrip(self):
        cache = self.make_cache(evidence_threshold=0.5)
        signature = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        cache.record_fix(signature, "module a(input x);\nendmodule\n", "module a(input y);\nendmodule\n")
        reloaded = self.make_cache(evidence_threshold=0.5)
        hint = reloaded.lookup_hint(signature)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.decision, "reuse")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertTrue(payload[0]["metadata"]["verified"])

    def test_hint_truncated_to_max_chars(self):
        cache = self.make_cache(evidence_threshold=0.5, max_hint_chars=120)
        signature = normalize_diagnostics(diag(PINNOTFOUND_STDERR))
        failing = "module a;\n" + "\n".join(f"wire w{i};" for i in range(200)) + "\nendmodule\n"
        cache.record_fix(signature, failing, failing.replace("w19;", "w_renamed;"))
        hint = cache.lookup_hint(signature)
        self.assertIsNotNone(hint)
        self.assertLessEqual(len(hint.text), 120 + len("\n... [hint truncated] ..."))


if __name__ == "__main__":
    unittest.main()
