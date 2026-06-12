import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = REPO_ROOT / "agentic_ip_reuse"
for path in (str(PLANNER_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentic_ip_reuse.repository import JsonIpRepository
from agentic_ip_reuse.semantic_repository import SemanticIpRepository
from agentic_ip_reuse.tools import AgentToolExecutor
from rag_rtl.embeddings import HashingEmbedder, make_embedder_with_fallback
from rag_rtl.retrieval import LexicalReranker, Retriever
from rag_rtl.types import RtlDocument
from rag_rtl.vector_store import VectorStore

CATALOG = {
    "ips": [
        {
            "ip_id": "sync_fifo",
            "name": "sync_fifo",
            "summary": "Synchronous FIFO buffer with configurable depth and width",
            "category": "memory",
            "interfaces": ["valid/ready"],
            "tags": ["fifo", "buffer", "queue"],
            "behavior": "module sync_fifo #(parameter DEPTH=16, WIDTH=8) (...);",
        },
        {
            "ip_id": "uart_tx",
            "name": "uart_tx",
            "summary": "UART transmitter with programmable baud rate",
            "category": "io",
            "interfaces": ["uart"],
            "tags": ["uart", "serial", "transmit"],
            "behavior": "module uart_tx (...);",
        },
    ]
}


def build_repository(tmp_dir: Path, mode: str = "semantic", min_score: float = 0.05, below_threshold: str = "flag"):
    catalog_path = tmp_dir / "catalog.json"
    catalog_path.write_text(json.dumps(CATALOG), encoding="utf-8")
    inner = JsonIpRepository(catalog_path)
    embedder = HashingEmbedder(dim=256)
    documents = []
    for item in CATALOG["ips"]:
        documents.append(
            RtlDocument(
                doc_id=item["ip_id"],
                problem=f"{item['name']}: {item['summary']}\n{' '.join(item['interfaces'])}",
                solution=item["behavior"],
                tags=item["tags"],
            )
        )
    vectors = embedder.encode([doc.retrieval_text for doc in documents])
    store = VectorStore(documents, vectors)
    return SemanticIpRepository(
        inner=inner,
        retriever=Retriever(store, embedder),
        reranker=LexicalReranker(),
        mode=mode,
        min_score=min_score,
        below_threshold=below_threshold,
    )


class SemanticIpRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_relevant_document_ranks_first(self):
        repository = build_repository(self.tmp_dir)
        candidates = repository.search("synchronous fifo buffer queue", top_k=2)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].ip_id, "sync_fifo")
        trace = repository.pop_last_trace()
        self.assertEqual(trace["mode"], "semantic")
        self.assertFalse(trace["low_confidence"])

    def test_drop_mode_removes_below_threshold(self):
        repository = build_repository(self.tmp_dir, min_score=0.99, below_threshold="drop")
        candidates = repository.search("synchronous fifo buffer", top_k=2)
        self.assertEqual(candidates, [])
        trace = repository.pop_last_trace()
        self.assertTrue(trace["low_confidence"])
        self.assertGreater(trace["filtered_below_threshold"], 0)

    def test_flag_mode_marks_low_confidence(self):
        repository = build_repository(self.tmp_dir, min_score=0.99, below_threshold="flag")
        candidates = repository.search("synchronous fifo buffer", top_k=2)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].criteria.get("retrieval_confidence"), "low")
        self.assertTrue(repository.pop_last_trace()["low_confidence"])

    def test_hybrid_mode_appends_token_results(self):
        repository = build_repository(self.tmp_dir, mode="hybrid", min_score=0.99, below_threshold="drop")
        candidates = repository.search("uart serial transmit", top_k=2)
        ids = [candidate.ip_id for candidate in candidates]
        self.assertIn("uart_tx", ids)
        self.assertEqual(candidates[0].criteria.get("retrieval_source"), "token")

    def test_filters_still_apply(self):
        repository = build_repository(self.tmp_dir)
        candidates = repository.search("fifo buffer", filters={"category": "io"}, top_k=5)
        self.assertTrue(all(candidate.category == "io" for candidate in candidates))

    def test_tool_payload_carries_trace_and_low_confidence_note(self):
        repository = build_repository(self.tmp_dir, min_score=0.99, below_threshold="flag")
        executor = AgentToolExecutor(repository, self.tmp_dir / "out")
        payload = json.loads(executor.execute("search_reuse_ip", {"query": "fifo buffer", "top_k": 2}))
        self.assertIn("retrieval", payload)
        self.assertIn("note", payload)
        self.assertIn("new RTL", payload["note"])

    def test_token_repository_payload_unchanged(self):
        catalog_path = self.tmp_dir / "catalog.json"
        catalog_path.write_text(json.dumps(CATALOG), encoding="utf-8")
        executor = AgentToolExecutor(JsonIpRepository(catalog_path), self.tmp_dir / "out")
        payload = json.loads(executor.execute("search_reuse_ip", {"query": "fifo buffer", "top_k": 2}))
        self.assertNotIn("retrieval", payload)
        self.assertNotIn("note", payload)
        self.assertTrue(payload["candidates"])


class EmbedderFallbackTests(unittest.TestCase):
    def test_hash_resolves_directly(self):
        embedder, name = make_embedder_with_fallback("hash")
        self.assertEqual(name, "hash")
        self.assertEqual(embedder.encode(["a b c"]).shape[1], embedder.dim)

    def test_auto_falls_back_without_sentence_transformers(self):
        warnings = []
        embedder, name = make_embedder_with_fallback("auto", warn=warnings.append)
        if name == "hash":
            self.assertTrue(warnings and "falling back" in warnings[0])
            self.assertIsInstance(embedder, HashingEmbedder)
        else:
            self.assertEqual(name, "sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
