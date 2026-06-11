import json
import tempfile
import unittest
from pathlib import Path

from agentic_ip_reuse.agent import AgentConfig
from agentic_ip_reuse.hierarchical import HierarchicalAgent, HierarchicalConfig
from agentic_ip_reuse.repository import JsonIpRepository
from agentic_ip_reuse.tools import AgentToolExecutor
from agentic_ip_reuse.types import DesignTask


CATALOG = '{"ips": []}'

FLAT_FINAL = json.dumps({
    "requirements": {"functionality": ["stream"]},
    "modules": [
        {"name": "Buffer", "role": "buffer", "interfaces": ["valid-ready"], "needs_decomposition": False},
        {"name": "Core", "role": "compute", "interfaces": ["valid-ready"], "needs_decomposition": False},
    ],
    "reuse_decisions": [],
    "integration_plan": [],
    "verification_plan": [],
    "debug_plan": [],
    "unresolved_assumptions": [],
})

COMPLEX_FINAL = json.dumps({
    "requirements": {"functionality": ["complex pipeline"]},
    "modules": [
        {
            "name": "Pipeline",
            "role": "multi-stage pipeline",
            "interfaces": ["valid-ready"],
            "needs_decomposition": True,
            "sub_spec": "4-stage arithmetic pipeline with bypass logic",
        },
        {"name": "Buffer", "role": "input buffer", "interfaces": ["valid-ready"], "needs_decomposition": False},
    ],
    "reuse_decisions": [],
    "integration_plan": [],
    "verification_plan": [],
    "debug_plan": [],
    "unresolved_assumptions": [],
})

SUB_FINAL = json.dumps({
    "requirements": {"functionality": ["pipeline stages"]},
    "modules": [
        {"name": "Stage1", "role": "fetch", "interfaces": ["valid-ready"]},
        {"name": "Stage2", "role": "decode", "interfaces": ["valid-ready"]},
    ],
    "reuse_decisions": [],
    "integration_plan": [],
    "verification_plan": [],
    "debug_plan": [],
    "unresolved_assumptions": [],
})


class SequentialFakeClient:
    """Pops responses in order; returns a tool-call-free final on last call."""

    def __init__(self, *response_sequences):
        self._sequences = [list(seq) for seq in response_sequences]
        self._call_count = 0

    def chat(self, messages, temperature=0.2, max_tokens=8192, tools=None, tool_choice=None, parallel_tool_calls=None):
        seq_idx = min(self._call_count, len(self._sequences) - 1)
        seq = self._sequences[seq_idx]
        self._call_count += 1
        if seq:
            return seq.pop(0)
        return {"content": json.dumps({"requirements": {}, "modules": [], "reuse_decisions": [],
                                        "integration_plan": [], "verification_plan": [], "debug_plan": [],
                                        "unresolved_assumptions": []})}


class HierarchicalAgentTests(unittest.TestCase):
    def _make_agent(self, llm, tmp, max_depth=2):
        catalog = Path(tmp) / "catalog.json"
        catalog.write_text(CATALOG, encoding="utf-8")
        repo = JsonIpRepository(catalog)
        executor = AgentToolExecutor(repo, tmp)
        config = AgentConfig(max_steps=4)
        return HierarchicalAgent(
            llm_client=llm,
            base_executor=executor,
            agent_config=config,
            h_config=HierarchicalConfig(max_depth=max_depth),
        )

    def test_flat_plan_no_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = SequentialFakeClient([{"content": FLAT_FINAL}])
            agent = self._make_agent(llm, tmp)
            plan = agent.run(DesignTask(prompt="Build stream design"))
            self.assertEqual(plan.depth, 0)
            self.assertEqual(len(plan.children), 0)
            self.assertEqual(len(plan.result.structured_plan["modules"]), 2)

    def test_complex_module_triggers_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = SequentialFakeClient(
                [{"content": COMPLEX_FINAL}],   # depth-0 agent
                [{"content": SUB_FINAL}],        # depth-1 agent for Pipeline
            )
            agent = self._make_agent(llm, tmp)
            plan = agent.run(DesignTask(prompt="Build complex pipeline"))
            self.assertEqual(len(plan.children), 1)
            self.assertIn("Pipeline", plan.children)
            child = plan.children["Pipeline"]
            self.assertEqual(child.depth, 1)
            self.assertEqual(len(child.result.structured_plan["modules"]), 2)

    def test_max_depth_prevents_infinite_recursion(self):
        with tempfile.TemporaryDirectory() as tmp:
            deeply_nested = json.dumps({
                "requirements": {},
                "modules": [{
                    "name": "DeepModule",
                    "role": "complex",
                    "interfaces": [],
                    "needs_decomposition": True,
                    "sub_spec": "still complex",
                }],
                "reuse_decisions": [],
                "integration_plan": [],
                "verification_plan": [],
                "debug_plan": [],
                "unresolved_assumptions": [],
            })
            llm = SequentialFakeClient(
                [{"content": deeply_nested}],
                [{"content": deeply_nested}],
                [{"content": deeply_nested}],
                [{"content": deeply_nested}],
            )
            agent = self._make_agent(llm, tmp, max_depth=1)
            plan = agent.run(DesignTask(prompt="Deeply nested"))
            # max_depth=1: depth-0 spawns depth-1, depth-1 does NOT spawn depth-2
            self.assertEqual(plan.depth, 0)
            self.assertEqual(len(plan.children), 1)
            child = plan.children["DeepModule"]
            self.assertEqual(child.depth, 1)
            self.assertEqual(len(child.children), 0)  # depth limit hit

    def test_hierarchical_summary_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = SequentialFakeClient([{"content": FLAT_FINAL}])
            agent = self._make_agent(llm, tmp)
            plan = agent.run(DesignTask(prompt="Build stream design"))
            summary_path = plan.write_hierarchical_summary(Path(tmp))
            self.assertTrue(summary_path.exists())
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["depth"], 0)
            self.assertIn("modules", data)

    def test_summary_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = SequentialFakeClient(
                [{"content": COMPLEX_FINAL}],
                [{"content": SUB_FINAL}],
            )
            agent = self._make_agent(llm, tmp)
            plan = agent.run(DesignTask(prompt="Build complex pipeline"))
            summary = plan.summary()
            self.assertIn("children", summary)
            self.assertIn("Pipeline", summary["children"])
            self.assertEqual(summary["children"]["Pipeline"]["depth"], 1)


if __name__ == "__main__":
    unittest.main()
