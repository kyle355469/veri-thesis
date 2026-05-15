import unittest

from rag_rtl.cli import build_parser
from rag_rtl.embeddings import HashingEmbedder, encode_texts
from rag_rtl.retrieval import LexicalReranker, Retriever
from rag_rtl.types import RtlDocument
from rag_rtl.vector_store import build_vector_store


class RetrievalTests(unittest.TestCase):
    def test_hash_embedder_parallel_encoding_matches_serial_order(self):
        texts = [
            "module invert input output not",
            "module add input input output plus",
            "module register clock reset data",
        ]
        embedder = HashingEmbedder(dim=128)

        serial = encode_texts(embedder, texts, jobs=1)
        parallel = encode_texts(embedder, texts, jobs=2)

        self.assertEqual(serial.shape, parallel.shape)
        self.assertTrue((serial == parallel).all())

    def test_index_parser_accepts_jobs(self):
        args = build_parser().parse_args(["index", "--jobs", "3"])

        self.assertEqual(args.jobs, 3)

    def test_generate_parser_accepts_generation_budget_knobs(self):
        args = build_parser().parse_args(
            [
                "generate",
                "--prompt",
                "Design an inverter",
                "--max-tokens",
                "8192",
                "--generation-temperature",
                "0.1",
            ]
        )

        self.assertEqual(args.max_tokens, 8192)
        self.assertEqual(args.generation_temperature, 0.1)

    def test_retriever_finds_related_document(self):
        docs = [
            RtlDocument("a", "Design an inverter module", "module invert(input i, output o); assign o = ~i; endmodule"),
            RtlDocument("b", "Design an adder module", "module add(input a, input b, output y); assign y = a + b; endmodule"),
        ]
        embedder = HashingEmbedder(dim=128)
        store = build_vector_store(docs, embedder.encode([doc.retrieval_text for doc in docs]))
        hits = Retriever(store, embedder).retrieve("invert signal i to output o", top_k=2)
        reranked = LexicalReranker().rerank("invert signal i to output o", hits, top_k=1)
        self.assertEqual(reranked[0].document.doc_id, "a")


if __name__ == "__main__":
    unittest.main()
