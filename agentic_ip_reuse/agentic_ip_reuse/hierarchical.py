from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import AgentConfig, AgenticIpReuseAgent
from .json_utils import json_default, preview_text
from .tools import AgentToolExecutor
from .types import AgentResult, DesignTask


@dataclass
class HierarchicalConfig:
    max_depth: int = 2
    decompose_key: str = "needs_decomposition"


@dataclass
class HierarchicalPlan:
    depth: int
    task: DesignTask
    result: AgentResult
    children: Dict[str, "HierarchicalPlan"] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "prompt_preview": preview_text(self.task.prompt, 120),
            "modules": len(self.result.structured_plan.get("modules", [])),
            "steps": self.result.steps,
            "stopped_reason": self.result.stopped_reason,
            "children": {name: child.summary() for name, child in self.children.items()},
        }

    def all_artifact_paths(self) -> Dict[str, Any]:
        paths: Dict[str, Any] = dict(self.result.artifact_paths)
        for name, child in self.children.items():
            paths[f"child/{name}"] = child.all_artifact_paths()
        return paths

    def write_hierarchical_summary(self, output_dir: Path) -> Path:
        target = output_dir / "hierarchical_summary.json"
        target.write_text(
            json.dumps(self.summary(), default=json_default, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target


class HierarchicalAgent:
    def __init__(
        self,
        llm_client: Any,
        base_executor: AgentToolExecutor,
        agent_config: AgentConfig,
        h_config: Optional[HierarchicalConfig] = None,
    ) -> None:
        self.llm_client = llm_client
        self.base_executor = base_executor
        self.agent_config = agent_config
        self.h_config = h_config or HierarchicalConfig()

    def run(self, task: DesignTask, depth: int = 0) -> HierarchicalPlan:
        executor = self._make_executor(task, depth)
        agent = AgenticIpReuseAgent(
            llm_client=self.llm_client,
            tool_executor=executor,
            config=self.agent_config,
        )
        result = agent.run(task)

        children: Dict[str, HierarchicalPlan] = {}
        if depth < self.h_config.max_depth:
            for module in result.structured_plan.get("modules", []):
                if not isinstance(module, dict):
                    continue
                if not module.get(self.h_config.decompose_key, False):
                    continue
                module_name = str(module.get("name", f"module_{len(children)}"))
                sub_task = _build_sub_task(module, task)
                children[module_name] = self.run(sub_task, depth + 1)

        return HierarchicalPlan(depth=depth, task=task, result=result, children=children)

    def _make_executor(self, task: DesignTask, depth: int) -> AgentToolExecutor:
        if depth == 0:
            return self.base_executor
        safe_name = _safe_dir_name(task.prompt)
        sub_dir = self.base_executor.output_dir / f"depth_{depth}" / safe_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        return AgentToolExecutor(self.base_executor.repository, sub_dir)


def _build_sub_task(module: Dict[str, Any], parent: DesignTask) -> DesignTask:
    name = module.get("name", "SubModule")
    role = module.get("role", "")
    sub_spec = module.get("sub_spec", "")

    parts = [f"Design the '{name}' sub-module. Role: {role}."]
    if sub_spec:
        parts.append(f"Detailed specification: {sub_spec}")
    interfaces = _str_list(module.get("interfaces"))
    if interfaces:
        parts.append(f"Required interfaces: {', '.join(interfaces)}.")
    parts.append(f"Parent context: {preview_text(parent.prompt, 200)}")

    constraints = list(parent.constraints) + _extract_constraints(module)

    return DesignTask(
        prompt=" ".join(parts),
        target_hdl=parent.target_hdl,
        constraints=constraints,
        known_interfaces=_str_list(module.get("interfaces")) or list(parent.known_interfaces),
        ppa_targets=list(parent.ppa_targets),
    )


def _extract_constraints(module: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("constraints", "requirements", "ppa_constraints", "verification_needs"):
        val = module.get(key)
        if isinstance(val, list):
            out.extend(str(x) for x in val)
        elif isinstance(val, str) and val:
            out.append(val)
    return out


def _str_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str) and val:
        return [val]
    return []


def _safe_dir_name(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]", "_", text[:max_len])
    return slug.strip("_") or "module"
