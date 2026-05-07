import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.llm import VllmClient
from rag_rtl.retrieval import LexicalReranker, Retriever
from rag_rtl.summarizer import ContextSummarizer
from rag_rtl.tool_calling import RtlToolExecutor
from rag_rtl.types import RtlDocument
from rag_rtl.vector_store import build_vector_store
from rag_rtl.verifier import RtlVerifier


class ToolLoopClient(VllmClient):
    def __init__(self):
        super().__init__(base_url="http://unused", model="stub")
        self.messages_seen = []

    def chat(self, messages, temperature=0.1, max_tokens=2048, tools=None, tool_choice=None, parallel_tool_calls=None):
        self.messages_seen.append(list(messages))
        if len(messages) == 1:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_yosys",
                            "arguments": json.dumps({"rtl": "module dut; endmodule"}),
                        },
                    }
                ],
            }
        return {"content": "```verilog\nmodule dut; endmodule\n```"}


class ToolCallingTests(unittest.TestCase):
    def test_vllm_http_error_includes_response_body(self):
        client = VllmClient(base_url="http://unused", model="siliconmind-server")
        error = urllib.error.HTTPError(
            url="http://unused/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesBody(b'{"error":{"message":"model not found"}}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "model not found"):
                client.complete("hello")

    def test_vllm_tool_loop_executes_and_returns_tool_result_message(self):
        client = ToolLoopClient()
        calls = []

        def execute(name, arguments):
            calls.append((name, arguments))
            return json.dumps({"ok": True, "tool": name})

        result = client.complete_with_tools(
            "Generate RTL",
            tools=[],
            tool_executor=execute,
            max_tool_rounds=1,
        )

        self.assertIn("module dut", result)
        self.assertEqual(calls, [("run_yosys", {"rtl": "module dut; endmodule"})])
        self.assertEqual(client.messages_seen[1][-1]["role"], "tool")
        self.assertEqual(client.messages_seen[1][-1]["tool_call_id"], "call_1")

    def test_rtl_tool_executor_retrieves_and_runs_configured_verifier_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = [
                RtlDocument(
                    "inv",
                    "Design an inverter",
                    "module invert(input i, output o); assign o = ~i; endmodule",
                )
            ]
            embedder = HashingEmbedder(dim=128)
            store = build_vector_store(docs, embedder.encode([doc.retrieval_text for doc in docs]))
            executor = RtlToolExecutor(
                retriever=Retriever(store, embedder),
                reranker=LexicalReranker(),
                summarizer=ContextSummarizer(),
                verifier=RtlVerifier(yosys_bin="/bin/true", verilator_bin="/bin/true"),
                default_top_module="invert",
            )

            retrieved = json.loads(executor.execute("retrieve_rtl_context", {"query": "invert signal"}))
            self.assertTrue(retrieved["ok"])
            self.assertEqual(retrieved["hits"][0]["doc_id"], "inv")

            yosys = json.loads(
                executor.execute(
                    "run_yosys",
                    {"rtl": "module invert(input i, output o); assign o = ~i; endmodule"},
                )
            )
            self.assertTrue(yosys["diagnostic"]["passed"])

            verilator = json.loads(
                executor.execute(
                    "run_verilator",
                    {"rtl": "module invert(input i, output o); assign o = ~i; endmodule"},
                )
            )
            self.assertTrue(verilator["diagnostic"]["passed"])

            report = json.loads(
                executor.execute(
                    "verify_rtl",
                    {"rtl": "module invert(input i, output o); assign o = ~i; endmodule"},
                )
            )
            self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()


class BytesBody:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def close(self):
        pass
