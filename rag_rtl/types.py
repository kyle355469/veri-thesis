from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RtlDocument:
    doc_id: str
    problem: str
    solution: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_text(self) -> str:
        tag_text = " ".join(self.tags)
        return f"{tag_text}\n{self.problem}\n{self.solution}".strip()


@dataclass
class RetrievalHit:
    document: RtlDocument
    score: float
    rerank_score: Optional[float] = None


@dataclass
class RtlTask:
    prompt: str
    target_hdl: str = "verilog"
    module_signature: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    max_repair_attempts: int = 1
    top_module: Optional[str] = None


@dataclass
class Diagnostic:
    tool: str
    passed: bool
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    missing: bool = False


@dataclass
class VerificationReport:
    syntax_passed: bool
    lint_passed: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        external_results = [item.passed for item in self.diagnostics if item.tool == "external_testbench"]
        external_passed = all(external_results) if external_results else True
        return self.syntax_passed and self.lint_passed and external_passed


@dataclass
class PipelineResponse:
    rtl: str
    verification: VerificationReport
    retrieved_doc_ids: List[str]
    cache_source: str
    repair_attempts: int
    llm_actions: List[Dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
