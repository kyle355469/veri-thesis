import tempfile
import unittest
from pathlib import Path

from agentic_ip_reuse.repository import JsonIpRepository


CATALOG = """{
  "ips": [
    {
      "ip_id": "fifo",
      "name": "FIFO",
      "summary": "valid-ready streaming buffer fifo",
      "category": "buffer",
      "interfaces": ["valid-ready"],
      "parameters": {"DATA_WIDTH": "32", "DEPTH": "16", "MODE": "fallthrough"},
      "license": "MIT",
      "verification": ["testbench", "formal"],
      "synthesis": "Yosys synthesis supported",
      "documentation": "complete integration examples",
      "tags": ["fifo", "buffer"],
      "behavior": "buffers stream data"
    }
  ]
}"""


class JsonIpRepositoryTests(unittest.TestCase):
    def test_search_inspect_and_score_catalog_ip(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.json"
            catalog.write_text(CATALOG, encoding="utf-8")
            repo = JsonIpRepository(catalog)

            hits = repo.search("need valid ready fifo buffer", top_k=3)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].ip_id, "fifo")
            self.assertGreater(hits[0].score, 0.0)

            description = repo.inspect("fifo")
            self.assertEqual(description.behavior, "buffers stream data")

            assessment = repo.score(
                description.candidate,
                {
                    "module_name": "Buffer / FIFO",
                    "role": "streaming buffer",
                    "interfaces": ["valid-ready"],
                    "requirements": ["configurable depth"],
                },
            )
            self.assertEqual(assessment.recommendation, "reuse")
            self.assertIn("function_match", assessment.criteria_scores)


if __name__ == "__main__":
    unittest.main()
