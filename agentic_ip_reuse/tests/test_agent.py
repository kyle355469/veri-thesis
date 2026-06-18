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


_E203_CATALOG = (
    '{"ips":['
    '{"ip_id":"e203_exu","name":"e203_exu","summary":"execution unit","category":"core","interfaces":["clk","rst"]},'
    '{"ip_id":"e203_ifu","name":"e203_ifu","summary":"fetch unit","category":"core","interfaces":["clk"]}'
    "]}"
)


class CatalogGroundingTests(unittest.TestCase):
    def test_catalog_is_injected_into_user_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = f"{tmp}/catalog.json"
            with open(catalog, "w", encoding="utf-8") as handle:
                handle.write(_E203_CATALOG)
            client = FakeChatClient(
                [{"content": json.dumps({"requirements": {"functionality": ["x"]}, "modules": [], "reuse_decisions": [{"module_name": "m", "selected_ip": "e203_exu"}], "integration_plan": ["wire it"]})}]
            )
            agent = AgenticIpReuseAgent(client, AgentToolExecutor(JsonIpRepository(catalog), tmp), AgentConfig(max_steps=1))
            agent.run(DesignTask(prompt="Build e203 core"))
            user_msg = client.calls[0]["messages"][1]["content"]
            self.assertIn("Reusable IP catalog", user_msg)
            self.assertIn("e203_exu", user_msg)
            self.assertIn("e203_ifu", user_msg)

    def test_hallucinated_ip_name_is_remapped_to_real_catalog_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = f"{tmp}/catalog.json"
            with open(catalog, "w", encoding="utf-8") as handle:
                handle.write(_E203_CATALOG)
            # Planner invents "e203_exu_core" (does not exist) and "made_up_ip".
            client = FakeChatClient(
                [{"content": json.dumps({
                    "requirements": {"functionality": ["x"]},
                    "modules": [],
                    "reuse_decisions": [
                        {"module_name": "exu", "ip": "e203_exu_core"},
                        {"module_name": "ghost", "ip": "totally_made_up_ip"},
                    ],
                    "integration_plan": ["wire it"],
                })}]
            )
            agent = AgenticIpReuseAgent(client, AgentToolExecutor(JsonIpRepository(catalog), tmp), AgentConfig(max_steps=1))
            result = agent.run(DesignTask(prompt="Build e203 core"))
            decisions = {d["module_name"]: d for d in result.structured_plan["reuse_decisions"]}
            self.assertEqual(decisions["exu"]["selected_ip"], "e203_exu")  # remapped
            self.assertIsNone(decisions["ghost"]["selected_ip"])  # dropped
            self.assertTrue(decisions["ghost"]["new_rtl_required"])
            self.assertEqual(result.grounding["remapped"], 1)
            self.assertEqual(result.grounding["dropped"], 1)

    def test_completeness_gate_reprompts_when_reuse_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = f"{tmp}/catalog.json"
            with open(catalog, "w", encoding="utf-8") as handle:
                handle.write(_E203_CATALOG)
            # First reply: no reuse_decisions / integration_plan. Retry: complete.
            client = FakeChatClient([
                {"content": json.dumps({"requirements": {"functionality": ["x"]}, "modules": [{"name": "m"}]})},
                {"content": json.dumps({
                    "requirements": {"functionality": ["x"]},
                    "modules": [{"name": "m"}],
                    "reuse_decisions": [{"module_name": "m", "selected_ip": "e203_exu"}],
                    "integration_plan": ["instantiate e203_exu"],
                })},
            ])
            agent = AgenticIpReuseAgent(client, AgentToolExecutor(JsonIpRepository(catalog), tmp), AgentConfig(max_steps=1))
            result = agent.run(DesignTask(prompt="Build e203 core"))
            self.assertTrue(result.structured_plan["reuse_decisions"])
            self.assertTrue(result.structured_plan["integration_plan"])
            # The completion request must not attach tools.
            self.assertIsNone(client.calls[-1]["tools"])


if __name__ == "__main__":
    unittest.main()
