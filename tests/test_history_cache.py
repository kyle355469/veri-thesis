import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag_rtl.history_cache import KEYWORD_EXTRACTION_PROMPT, HistorySemanticCache, LlmKeywordExtractor


class DictEmbedder:
    dim = 2

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class ExplodingEmbedder:
    dim = 2

    def encode(self, texts):
        raise AssertionError("cache mode none must not compute embeddings")


class KeywordLlm:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def complete(self, prompt, temperature=0.1, max_tokens=2048):
        self.prompts.append(prompt)
        return self.response


class HistoryCacheTests(unittest.TestCase):
    def test_llm_keyword_prompt_uses_structured_verilog_spec_schema(self):
        llm = KeywordLlm(
            """
            {
              "direction": "design",
              "module_name": ["invert"],
              "type": "combinational",
              "gate_usage": ["Inverter"],
              "signals": {
                "input": ["i"],
                "output": ["o"]
              },
              "keywords": [
                "invert",
                "logical not",
                "continuous assignment",
                "combinational"
              ]
            }
            """
        )
        extractor = LlmKeywordExtractor(llm)

        keywords = extractor("Create module invert with input i and output o. assign o = ~i;")

        self.assertIn("You are a Verilog specification keyword extraction assistant.", llm.prompts[0])
        self.assertIn("Specification:\nCreate module invert", llm.prompts[0])
        self.assertNotIn("{SPEC}", llm.prompts[0])
        self.assertIn("Return only valid JSON.", KEYWORD_EXTRACTION_PROMPT)
        self.assertEqual(
            keywords,
            [
                "invert",
                "combinational",
                "inverter",
                "i",
                "o",
                "logical_not",
                "continuous_assignment",
            ],
        )

    def test_llm_keyword_parser_accepts_behavior_and_constraints_fields(self):
        llm = KeywordLlm(
            """
            {
              "direction": "testbench",
              "module_name": ["counter_tb"],
              "type": "sequential",
              "gate_usage": ["Register"],
              "input_signals": ["clk", "rst_n"],
              "output_signals": ["done"],
              "behavior_keywords": ["counter", "state transition"],
              "constraints": ["active-low reset"]
            }
            """
        )
        extractor = LlmKeywordExtractor(llm)

        keywords = extractor("Write a counter testbench")

        self.assertEqual(
            keywords,
            [
                "counter_tb",
                "sequential",
                "register",
                "clk",
                "rst_n",
                "done",
                "counter",
                "state_transition",
                "active_low_reset",
            ],
        )

    def test_reuse_and_evidence_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            similar = [0.90, math.sqrt(1.0 - 0.90**2)]
            embedder = DictEmbedder(
                {
                    "adder old": [1.0, 0.0],
                    "adder old exact": [1.0, 0.0],
                    "adder similar": similar,
                }
            )
            cache = HistorySemanticCache(
                embedder,
                Path(tmp) / "cache.json",
                reuse_threshold=0.95,
                evidence_threshold=0.88,
                mode="keywords",
            )
            cache.put("adder old", "module old; endmodule")

            self.assertEqual(cache.lookup("adder old exact").decision, "reuse")
            evidence = cache.lookup("adder similar")
            self.assertEqual(evidence.decision, "evidence")
            self.assertIsNotNone(evidence.evidence_entry)

    def test_keyword_mode_prefilters_before_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            embedder = DictEmbedder(
                {
                    "adder circuit": [1.0, 0.0],
                    "fifo buffer": [0.0, 1.0],
                    "fifo request": [1.0, 0.0],
                }
            )
            cache_path = Path(tmp) / "cache.json"
            keyword_cache = HistorySemanticCache(embedder, cache_path, mode="keywords")
            keyword_cache.put("adder circuit", "module adder; endmodule")
            keyword_cache.put("fifo buffer", "module fifo; endmodule")

            lookup = keyword_cache.lookup("fifo request")
            self.assertEqual(lookup.candidate_count, 1)
            self.assertEqual(lookup.decision, "miss")
            self.assertEqual(lookup.best_history_match["query"], "fifo buffer")

            direct_cache = HistorySemanticCache(embedder, cache_path, mode="direct")
            direct_lookup = direct_cache.lookup("fifo request")
            self.assertEqual(direct_lookup.decision, "reuse")
            self.assertEqual(direct_lookup.reusable_entry.query, "adder circuit")

    def test_keyword_mode_does_not_embed_or_score_without_keyword_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            embedder = DictEmbedder(
                {
                    "adder circuit": [1.0, 0.0],
                    "fifo buffer": [0.0, 1.0],
                }
            )
            cache = HistorySemanticCache(embedder, Path(tmp) / "cache.json", mode="keywords")
            cache.put("adder circuit", "module adder; endmodule")
            cache.put("fifo buffer", "module fifo; endmodule")

            lookup = cache.lookup("uart transmitter")

            self.assertEqual(lookup.decision, "miss")
            self.assertEqual(lookup.candidate_count, 0)
            self.assertIsNone(lookup.best_history_match)

    def test_none_mode_never_embeds_loads_or_saves_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            cache_path.write_text("not valid json", encoding="utf-8")

            def fail_keyword_extractor(_query):
                raise AssertionError("cache mode none must not extract keywords")

            cache = HistorySemanticCache(
                ExplodingEmbedder(),
                cache_path,
                mode="none",
                keyword_extractor=fail_keyword_extractor,
            )

            lookup = cache.lookup("adder request")
            cache.put("adder request", "module adder; endmodule")

            self.assertEqual(lookup.decision, "disabled")
            self.assertIsNone(lookup.reusable_entry)
            self.assertIsNone(lookup.evidence_entry)
            self.assertEqual(lookup.candidate_count, 0)
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "not valid json")


if __name__ == "__main__":
    unittest.main()
