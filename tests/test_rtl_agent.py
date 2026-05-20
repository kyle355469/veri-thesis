import json
import tempfile
import unittest
from pathlib import Path

from rtl_agent.agent import AgentConfig, AgenticRtlAgent
from rtl_agent.cli import build_parser
from rtl_agent.harness import CompositeToolExecutor, WORKSPACE_TOOL_SCHEMAS, WorkspaceToolExecutor
from rag_rtl.types import RtlTask
from rag_rtl.verifier import RtlVerifier


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.1, max_tokens=2048, tools=None, tool_choice=None, parallel_tool_calls=None):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
            }
        )
        if not self.responses:
            raise AssertionError("no fake chat response left")
        return self.responses.pop(0)


class FakeToolExecutor:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps(self.results.get(name, {"ok": True, "tool": name}))


def passing_verifier():
    return RtlVerifier(yosys_bin="/bin/true", verilator_bin="/bin/true")


class AgenticRtlAgentTests(unittest.TestCase):
    def test_agent_executes_model_chosen_retrieval_then_final_verifies(self):
        client = FakeChatClient(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "retrieve_rtl_context",
                                "arguments": json.dumps({"query": "inverter", "top_k": 4}),
                            },
                        }
                    ],
                },
                {"content": "```verilog\nmodule inv(input i, output o); assign o = ~i; endmodule\n```"},
            ]
        )
        executor = FakeToolExecutor({"retrieve_rtl_context": {"ok": True, "tool": "retrieve_rtl_context", "hits": []}})
        agent = AgenticRtlAgent(
            client,
            executor,
            passing_verifier(),
            AgentConfig(max_steps=4),
        )

        result = agent.run(RtlTask(prompt="Design an inverter", top_module="inv"))

        self.assertTrue(result.used_tools)
        self.assertTrue(result.verification.passed)
        self.assertIn("module inv", result.rtl)
        self.assertEqual(executor.calls, [("retrieve_rtl_context", {"query": "inverter", "top_k": 4})])
        self.assertEqual(client.calls[0]["tool_choice"], "auto")
        self.assertEqual(client.calls[1]["messages"][-1]["role"], "tool")

    def test_agent_can_call_workspace_file_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "spec.txt").write_text("module should invert input\n", encoding="utf-8")
            client = FakeChatClient(
                [
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "spec.txt"}),
                                },
                            }
                        ],
                    },
                    {"content": "```verilog\nmodule dut(input i, output o); assign o = ~i; endmodule\n```"},
                ]
            )
            agent = AgenticRtlAgent(
                client,
                CompositeToolExecutor(WorkspaceToolExecutor(tmp)),
                passing_verifier(),
                AgentConfig(max_steps=3),
                tool_schemas=WORKSPACE_TOOL_SCHEMAS,
            )

            result = agent.run(RtlTask(prompt="Read spec and implement", top_module="dut"))

            self.assertTrue(result.used_tools)
            self.assertTrue(result.verification.passed)
            self.assertIn("read_file", [schema["function"]["name"] for schema in client.calls[0]["tools"]])

    def test_agent_final_verifies_even_when_model_uses_no_tools(self):
        client = FakeChatClient(
            [{"content": "```verilog\nmodule dut; endmodule\n```"}]
        )
        executor = FakeToolExecutor()
        agent = AgenticRtlAgent(client, executor, passing_verifier(), AgentConfig(max_steps=2))

        result = agent.run(RtlTask(prompt="Create empty module", top_module="dut"))

        self.assertFalse(result.used_tools)
        self.assertEqual(executor.calls, [])
        self.assertTrue(result.verification.passed)
        self.assertEqual(result.stopped_reason, "final")

    def test_agent_forces_final_after_tool_budget(self):
        client = FakeChatClient(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_yosys", "arguments": json.dumps({"rtl": "module dut; endmodule"})},
                        }
                    ],
                },
                {"content": "```verilog\nmodule dut; endmodule\n```"},
            ]
        )
        executor = FakeToolExecutor({"run_yosys": {"ok": True, "tool": "run_yosys", "diagnostic": {"passed": True}}})
        agent = AgenticRtlAgent(client, executor, passing_verifier(), AgentConfig(max_steps=1))

        result = agent.run(RtlTask(prompt="Create empty module", top_module="dut"))

        self.assertEqual(result.stopped_reason, "forced_final")
        self.assertTrue(result.verification.passed)
        self.assertEqual(client.calls[-1]["tool_choice"], "none")

    def test_cli_parser_accepts_agent_run_args(self):
        args = build_parser().parse_args(
            [
                "run",
                "--prompt",
                "Design mux",
                "--base-url",
                "http://localhost:18000/v1",
                "--tool-choice",
                "auto",
                "--max-steps",
                "3",
                "--show-final-code",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.prompt, "Design mux")
        self.assertEqual(args.base_url, "http://localhost:18000/v1")
        self.assertEqual(args.max_steps, 3)
        self.assertTrue(args.show_final_code)


class WorkspaceToolExecutorTests(unittest.TestCase):
    def test_workspace_tools_read_write_and_run_allowed_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = WorkspaceToolExecutor(tmp)
            write = json.loads(
                executor.execute(
                    "write_file",
                    {"path": "out/dut.v", "content": "module dut; endmodule\n"},
                )
            )
            self.assertTrue(write["ok"])
            self.assertTrue(Path(tmp, "out/dut.v").exists())

            read = json.loads(executor.execute("read_file", {"path": "out/dut.v"}))
            self.assertTrue(read["ok"])
            self.assertIn("module dut", read["content"])

            command = json.loads(executor.execute("run_command", {"argv": ["grep", "module", "out/dut.v"]}))
            self.assertTrue(command["ok"])
            self.assertEqual(command["returncode"], 0)
            self.assertIn("module dut", command["stdout"])

    def test_workspace_tools_block_path_escape_and_unlisted_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = WorkspaceToolExecutor(tmp)

            read = json.loads(executor.execute("read_file", {"path": "../outside.txt"}))
            self.assertFalse(read["ok"])
            self.assertIn("escapes workspace", read["error"])

            command = json.loads(executor.execute("run_command", {"argv": ["python3", "-c", "print(1)"]}))
            self.assertFalse(command["ok"])
            self.assertIn("not allowed", command["error"])

            absolute = json.loads(executor.execute("run_command", {"argv": ["grep", "x", "/etc/passwd"]}))
            self.assertFalse(absolute["ok"])
            self.assertIn("escapes workspace", absolute["error"])


if __name__ == "__main__":
    unittest.main()
