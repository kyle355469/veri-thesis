from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List

from rag_rtl.json_utils import json_default, preview_text

from .types import IpCandidate, IpReusePlan, ModuleSpec, SystemRequirements


MODULE_CATEGORIES = [
    "Input Interface",
    "Buffer / FIFO",
    "Processing Core",
    "Memory Controller",
    "Output Interface",
]


CRITERIA = [
    "function_match",
    "interface_compatibility",
    "configurability",
    "verification_status",
    "license",
    "synthesis_support",
    "documentation_quality",
]


def build_requirements_prompt(prompt: str, target_hdl: str, constraints: Iterable[str]) -> str:
    constraint_text = "\n".join(f"- {item}" for item in constraints) if constraints else "- none"
    return f"""You are planning an IC design that may reuse existing IP.
Extract system-level requirements from the user request.

Target HDL: {target_hdl}
Extra constraints:
{constraint_text}

User request:
{prompt}

Return only JSON with this shape:
{{
  "functionality": "required behavior",
  "performance_target": "latency/throughput/frequency target or unknown",
  "io_interface": "AXI/APB/Wishbone/valid-ready/plain ports/etc. or unknown",
  "ppa_constraints": ["power/performance/area constraints"],
  "clock_reset": "clock/reset assumptions or unknown",
  "assumptions": ["explicit assumptions needed"],
  "unknowns": ["important missing fields"]
}}"""


def build_decomposition_prompt(requirements: SystemRequirements) -> str:
    requirements_json = json.dumps(asdict(requirements), default=json_default, indent=2)
    categories = "\n".join(f"- {item}" for item in MODULE_CATEGORIES)
    return f"""Decompose the IC system into reusable modules.

System requirements:
{requirements_json}

Use these categories when applicable:
{categories}

Return only JSON:
{{
  "modules": [
    {{
      "category": "one category above",
      "name": "module name",
      "purpose": "what this module does",
      "required_interface": "required interface or unknown",
      "performance_target": "target or unknown",
      "ppa_constraints": ["constraints"],
      "reuse_query": "semantic retrieval query for reusable IP",
      "omitted_reason": null
    }}
  ]
}}

If a category is irrelevant, omit it only when the module list or assumptions make that clear."""


def build_candidate_evaluation_prompt(module: ModuleSpec, candidates: List[IpCandidate]) -> str:
    compact_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        compact_candidates.append(
            {
                "doc_id": candidate.doc_id,
                "score": candidate.score,
                "rerank_score": candidate.rerank_score,
                "tags": candidate.tags,
                "problem": preview_text(candidate.problem, 700),
                "solution": preview_text(candidate.solution, 1600),
                "metadata": candidate.metadata,
                "known_criteria": candidate.criteria,
            }
        )
    criteria = "\n".join(f"- {item}" for item in CRITERIA)
    return f"""Evaluate reusable IP candidates for one IC module.
Do not invent license, verification, synthesis, or documentation facts. Use "unknown" when metadata does not state them.

Module:
{json.dumps(asdict(module), default=json_default, indent=2)}

Criteria:
{criteria}

Candidates:
{json.dumps(compact_candidates, default=json_default, indent=2)}

Choose one action:
- reuse: candidate is directly usable
- configure: candidate is reusable with parameter changes or wrappers
- adapt: candidate is partially reusable but needs RTL adaptation
- new: no acceptable candidate was found

Return only JSON:
{{
  "candidate_evaluations": [
    {{
      "doc_id": "candidate id",
      "criteria": {{
        "function_match": "assessment or unknown",
        "interface_compatibility": "assessment or unknown",
        "configurability": "assessment or unknown",
        "verification_status": "assessment or unknown",
        "license": "assessment or unknown",
        "synthesis_support": "assessment or unknown",
        "documentation_quality": "assessment or unknown"
      }},
      "rationale": "short reason"
    }}
  ],
  "selected_doc_id": "candidate id or null",
  "action": "reuse|configure|adapt|new",
  "parameterization": {{}},
  "integration_notes": "how to integrate or why new RTL is needed",
  "rationale": "selection rationale"
}}"""


def build_rtl_generation_prompt(plan: IpReusePlan, target_hdl: str, top_module: str | None) -> str:
    plan_json = json.dumps(asdict(plan), default=json_default, indent=2)
    top = top_module or "choose a suitable top module name from the requirements"
    return f"""Generate integrated {target_hdl} RTL from this IP reuse plan.
Use selected reusable IP behavior where appropriate, configure or adapt candidates as described, and create new RTL for modules marked new.
Return exactly one fenced {target_hdl} code block and no extra prose.

Top module: {top}

IP reuse plan:
{plan_json}"""


def build_repair_prompt(
    plan: IpReusePlan,
    rtl: str,
    diagnostics: List[Dict[str, Any]],
    target_hdl: str,
    top_module: str | None,
) -> str:
    plan_json = json.dumps(asdict(plan), default=json_default, indent=2)
    return f"""Repair this integrated {target_hdl} RTL so it passes syntax and lint checks.
Keep the same IP reuse intent and public top-level behavior.
Return exactly one fenced {target_hdl} code block and no extra prose.

Top module: {top_module or "unknown"}

IP reuse plan:
{plan_json}

Diagnostics:
{json.dumps(diagnostics, default=json_default, indent=2)}

Current RTL:
```{target_hdl}
{rtl}
```"""
