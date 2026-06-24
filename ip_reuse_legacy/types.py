from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag_rtl.types import VerificationReport


@dataclass
class SystemRequirements:
    functionality: str = "unknown"
    performance_target: str = "unknown"
    io_interface: str = "unknown"
    ppa_constraints: List[str] = field(default_factory=list)
    clock_reset: str = "unknown"
    assumptions: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)


@dataclass
class ModuleSpec:
    category: str
    name: str
    purpose: str
    required_interface: str = "unknown"
    performance_target: str = "unknown"
    ppa_constraints: List[str] = field(default_factory=list)
    reuse_query: str = ""
    omitted_reason: Optional[str] = None


@dataclass
class IpCandidate:
    doc_id: str
    score: float
    rerank_score: Optional[float]
    tags: List[str]
    problem: str
    solution: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    criteria: Dict[str, str] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class ModuleReuseDecision:
    module: ModuleSpec
    candidates: List[IpCandidate] = field(default_factory=list)
    selected_doc_id: Optional[str] = None
    action: str = "new"
    parameterization: Dict[str, Any] = field(default_factory=dict)
    integration_notes: str = ""
    rationale: str = ""


@dataclass
class IpReusePlan:
    requirements: SystemRequirements
    modules: List[ModuleSpec]
    decisions: List[ModuleReuseDecision]


@dataclass
class LlmTrace:
    stage: str
    prompt_preview: str
    response_preview: str
    parsed: bool
    prompt_chars: int = 0
    response_chars: int = 0


@dataclass
class AgenticIpReuseResult:
    plan: IpReusePlan
    rtl: str
    final_text: str
    verification: VerificationReport
    repair_attempts: int = 0
    functional_repair_attempts: int = 0
    function_info: str = ""
    llm_traces: List[LlmTrace] = field(default_factory=list)
    retrieval_traces: List[Dict[str, Any]] = field(default_factory=list)
    repair_cache_events: List[Dict[str, Any]] = field(default_factory=list)
    functional_repair_events: List[Dict[str, Any]] = field(default_factory=list)
    large_spec_manifest: Optional[Dict[str, Any]] = None
    decomposition_tree: Optional[Dict[str, Any]] = None
    module_generation: List[Dict[str, Any]] = field(default_factory=list)
    workspace_dir: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirements": self.plan.requirements,
            "modules": self.plan.modules,
            "ip_reuse_decisions": self.plan.decisions,
            "rtl": self.rtl,
            "final_text": self.final_text,
            "verification": self.verification,
            "repair_attempts": self.repair_attempts,
            "functional_repair_attempts": self.functional_repair_attempts,
            "function_info": self.function_info,
            "llm_traces": self.llm_traces,
            "retrieval_traces": self.retrieval_traces,
            "repair_cache_events": self.repair_cache_events,
            "functional_repair_events": self.functional_repair_events,
            "large_spec_manifest": self.large_spec_manifest,
            "decomposition_tree": self.decomposition_tree,
            "module_generation": self.module_generation,
            "workspace_dir": self.workspace_dir,
            "artifacts": self.artifacts,
        }
