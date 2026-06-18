from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


CRITERIA = [
    "function_match",
    "interface_compatibility",
    "configurability",
    "verification_status",
    "license",
    "synthesis_support",
    "documentation_quality",
]


@dataclass
class DesignTask:
    prompt: str
    target_hdl: str = "systemverilog"
    constraints: List[str] = field(default_factory=list)
    known_interfaces: List[str] = field(default_factory=list)
    ppa_targets: List[str] = field(default_factory=list)


@dataclass
class SystemRequirements:
    functionality: List[str] = field(default_factory=list)
    performance: List[str] = field(default_factory=list)
    io_interfaces: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    ppa_constraints: List[str] = field(default_factory=list)
    clock_reset: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


@dataclass
class ModulePlan:
    name: str
    role: str
    interfaces: List[str] = field(default_factory=list)
    reuse_preference: str = "prefer reusable IP when criteria pass"
    verification_needs: List[str] = field(default_factory=list)


@dataclass
class IpCandidate:
    ip_id: str
    name: str
    summary: str
    category: str
    interfaces: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    license: str = "unknown"
    verification: List[str] = field(default_factory=list)
    synthesis: str = "unknown"
    documentation: str = "unknown"
    tags: List[str] = field(default_factory=list)
    score: float = 0.0
    criteria: Dict[str, str] = field(default_factory=dict)


@dataclass
class IpDescription:
    candidate: IpCandidate
    behavior: str = ""
    integration_notes: List[str] = field(default_factory=list)
    known_limits: List[str] = field(default_factory=list)


@dataclass
class IpAssessment:
    ip_id: str
    module_name: str
    total_score: float
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    criteria_notes: Dict[str, str] = field(default_factory=dict)
    recommendation: str = "review"


@dataclass
class ReuseDecision:
    module_name: str
    selected_ip: Optional[str] = None
    rejected_ips: List[str] = field(default_factory=list)
    required_adapters: List[str] = field(default_factory=list)
    parameterization: Dict[str, Any] = field(default_factory=dict)
    risk_notes: List[str] = field(default_factory=list)
    new_rtl_required: bool = False


@dataclass
class AgentEvent:
    event: str
    step: int
    message: str
    tool: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"event": self.event, "step": self.step, "message": self.message}
        if self.tool:
            payload["tool"] = self.tool
        if self.data:
            payload["data"] = self.data
        return payload


@dataclass
class AgentResult:
    final_text: str
    structured_plan: Dict[str, Any]
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    events: List[AgentEvent] = field(default_factory=list)
    steps: int = 0
    used_tools: bool = False
    stopped_reason: str = "final"
    grounding: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_text": self.final_text,
            "structured_plan": self.structured_plan,
            "artifact_paths": self.artifact_paths,
            "events": [event.to_dict() for event in self.events],
            "steps": self.steps,
            "used_tools": self.used_tools,
            "stopped_reason": self.stopped_reason,
            "grounding": self.grounding,
        }


class IpRepository(Protocol):
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[IpCandidate]:
        ...

    def inspect(self, ip_id: str) -> IpDescription:
        ...

    def score(self, candidate: IpCandidate, module_requirements: Dict[str, Any]) -> IpAssessment:
        ...
