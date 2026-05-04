import unittest

from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.retrieval import LexicalReranker, Retriever
from rag_rtl.types import RtlDocument
from rag_rtl.vector_store import build_vector_store


class RetrievalTests(unittest.TestCase):
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
