import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "run_agentic_plan_legacy_realbench.py"
    spec = importlib.util.spec_from_file_location("run_agentic_plan_legacy_realbench_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_agentic_plan_legacy_realbench_test"] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = load_script()

from rag_rtl.embeddings import HashingEmbedder  # noqa: E402


class CountingEmbedder:
    def __init__(self, dim=64):
        self.inner = HashingEmbedder(dim=dim)
        self.dim = dim
        self.encode_calls = 0

    def encode(self, texts):
        texts = list(texts)
        self.encode_calls += 1
        return self.inner.encode(texts)


CATALOG = {
    "ips": [
        {
            "ip_id": "sync_fifo",
            "name": "sync_fifo",
            "summary": "Synchronous FIFO",
            "interfaces": ["valid/ready"],
            "tags": ["fifo"],
            "behavior": "module sync_fifo; endmodule",
        }
    ]
}


class ParserDefaultTests(unittest.TestCase):
    def test_rag_flags_default_to_legacy_behavior(self):
        args = SCRIPT.build_parser().parse_args([])
        self.assertEqual(args.planner_search_mode, "token")
        self.assertEqual(args.repair_cache, "off")
        self.assertEqual(args.embedder, "auto")
        self.assertEqual(args.planner_retrieval_below_threshold, "flag")
        self.assertFalse(args.reindex)


class TaskIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.tmp.name) / "catalogs" / "module.aes.task.json"
        self.catalog_path.parent.mkdir(parents=True)
        self.catalog_path.write_text(json.dumps(CATALOG), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_build_persists_then_second_loads_without_encoding(self):
        embedder = CountingEmbedder()
        store, meta = SCRIPT.build_or_load_task_index(self.catalog_path, embedder, "hash", reindex=False)
        self.assertTrue(meta["index_built"])
        self.assertEqual(meta["doc_count"], 1)
        index_dir = self.catalog_path.with_suffix(".index")
        self.assertTrue((index_dir / "vectors.npy").exists())
        self.assertTrue((index_dir / "documents.jsonl").exists())
        self.assertTrue((index_dir / "index_meta.json").exists())

        fresh = CountingEmbedder()
        store2, meta2 = SCRIPT.build_or_load_task_index(self.catalog_path, fresh, "hash", reindex=False)
        self.assertFalse(meta2["index_built"])
        self.assertEqual(fresh.encode_calls, 0)
        self.assertEqual(len(store2.documents), len(store.documents))

    def test_catalog_change_triggers_rebuild(self):
        embedder = CountingEmbedder()
        SCRIPT.build_or_load_task_index(self.catalog_path, embedder, "hash", reindex=False)
        changed = dict(CATALOG)
        changed["ips"] = CATALOG["ips"] + [
            {"ip_id": "uart", "name": "uart", "summary": "UART", "interfaces": [], "tags": [], "behavior": ""}
        ]
        self.catalog_path.write_text(json.dumps(changed), encoding="utf-8")
        fresh = CountingEmbedder()
        _, meta = SCRIPT.build_or_load_task_index(self.catalog_path, fresh, "hash", reindex=False)
        self.assertTrue(meta["index_built"])
        self.assertEqual(meta["doc_count"], 2)
        self.assertGreater(fresh.encode_calls, 0)

    def test_embedder_change_and_reindex_trigger_rebuild(self):
        embedder = CountingEmbedder()
        SCRIPT.build_or_load_task_index(self.catalog_path, embedder, "hash", reindex=False)
        fresh = CountingEmbedder()
        _, meta = SCRIPT.build_or_load_task_index(self.catalog_path, fresh, "st-minilm", reindex=False)
        self.assertTrue(meta["index_built"])
        forced = CountingEmbedder()
        _, meta = SCRIPT.build_or_load_task_index(self.catalog_path, forced, "st-minilm", reindex=True)
        self.assertTrue(meta["index_built"])
        self.assertGreater(forced.encode_calls, 0)


class AggregateTests(unittest.TestCase):
    def test_rag_aggregates_over_records(self):
        records = [
            {
                "repair_cache_metrics": {"enabled": True, "lookups": 2, "hits": 1},
                "retrieval_metrics": {"searches": 3, "low_confidence_searches": 1},
                "legacy_repair_attempts": 2,
                "wall_s": 10.0,
                "llm_token_estimate": {"prompt_tokens": 100, "response_tokens": 50},
            },
            {
                "repair_cache_metrics": {"enabled": False},
                "retrieval_metrics": None,
                "legacy_repair_attempts": 0,
                "wall_s": 4.0,
                "llm_token_estimate": None,
            },
        ]
        aggregates = SCRIPT.rag_aggregates(records)
        self.assertEqual(aggregates["repair_cache_lookups"], 2)
        self.assertEqual(aggregates["repair_cache_hits"], 1)
        self.assertEqual(aggregates["repair_cache_hit_rate"], 0.5)
        self.assertEqual(aggregates["mean_repair_attempts"], 1.0)
        self.assertEqual(aggregates["mean_wall_s"], 7.0)
        self.assertEqual(aggregates["total_estimated_tokens"], 150)
        self.assertEqual(aggregates["retrieval_searches"], 3)
        self.assertAlmostEqual(aggregates["low_confidence_search_rate"], 1 / 3)


class RagRuntimeTests(unittest.TestCase):
    def test_defaults_build_inert_runtime(self):
        args = SCRIPT.build_parser().parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            rag = SCRIPT.build_rag_runtime(args, Path(tmp), [], {})
        self.assertIsNone(rag.embedder)
        self.assertEqual(rag.embedder_name, "none")
        self.assertEqual(rag.stores, {})
        self.assertIsNone(rag.repair_cache_for("anything"))


if __name__ == "__main__":
    unittest.main()
