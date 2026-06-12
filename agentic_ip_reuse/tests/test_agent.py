import json
import tempfile
import unittest

from agentic_ip_reuse.agent import AgentConfig, AgenticIpReuseAgent
from agentic_ip_reuse.repository import JsonIpRepository
from agentic_ip_reuse.tools import AgentToolExecutor
from agentic_ip_reuse.types import DesignTask


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=8192, tools=None, tool_choice=None, parallel_tool_calls=None):
        self.calls.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        if not self.responses:
            raise AssertionError("no fake chat response left")
        return self.responses.pop(0)


class AgentTests(unittest.TestCase):
    def test_agent_executes_search_tool_then_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = f"{tmp}/catalog.json"
            with open(catalog, "w", encoding="utf-8") as handle:
                handle.write(
                    """{"ips":[{"ip_id":"fifo","name":"FIFO","summary":"stream fifo","category":"buffer","interfaces":["valid-ready"],"parameters":{"DEPTH":"16"},"license":"MIT","verification":["testbench"],"synthesis":"synthesis supported","documentation":"complete examples","tags":["fifo"]}]}"""
                )
            client = FakeChatClient(
                [
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_reuse_ip",
                                    "arguments": json.dumps({"query": "fifo stream", "top_k": 2}),
                                },
                            }
                        ],
                    },
                    {
                        "content": json.dumps(
                            {
                                "requirements": {"functionality": ["stream data"]},
                                "modules": [{"name": "Buffer / FIFO", "role": "buffer", "interfaces": ["valid-ready"]}],
                                "reuse_decisions": [{"module_name": "Buffer / FIFO", "selected_ip": "fifo", "new_rtl_required": False}],
                                "integration_plan": ["instantiate fifo"],
                                "verification_plan": ["run fifo tests"],
                                "debug_plan": ["trace ready valid"],
                                "unresolved_assumptions": [],
                            }
                        )
                    },
                ]
            )
            agent = AgenticIpReuseAgent(
                client,
                AgentToolExecutor(JsonIpRepository(catalog), tmp),
                AgentConfig(max_steps=3),
            )

            result = agent.run(DesignTask(prompt="Build stream design"))

            self.assertTrue(result.used_tools)
            self.assertEqual(result.stopped_reason, "final")
            self.assertIn("requirements", result.artifact_paths)
            self.assertTrue(result.artifact_paths["result"].endswith("result.json"))
            self.assertEqual(client.calls[0]["tool_choice"], "auto")

    def test_agent_forces_final_after_tool_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = f"{tmp}/catalog.json"
            with open(catalog, "w", encoding="utf-8") as handle:
                handle.write("""{"ips":[]}""")
            client = FakeChatClient(
                [
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search_reuse_ip", "arguments": json.dumps({"query": "fifo"})},
                            }
                        ],
                    },
                    {"content": json.dumps({"requirements": {"functionality": ["fallback"]}})},
                ]
            )
            agent = AgenticIpReuseAgent(
                client,
                AgentToolExecutor(JsonIpRepository(catalog), tmp),
                AgentConfig(max_steps=1),
            )

            result = agent.run(DesignTask(prompt="Build stream design"))

            self.assertEqual(result.stopped_reason, "forced_final")
            # The forced-final request must not attach tools: the deployed
            # reasoning parser returns empty content when tools are present.
            self.assertIsNone(client.calls[-1]["tools"])
            self.assertIsNone(client.calls[-1]["tool_choice"])
            self.assertIn("requirements", result.structured_plan)


if __name__ == "__main__":
    unittest.main()
