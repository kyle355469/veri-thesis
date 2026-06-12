from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from rag_rtl.json_utils import json_default, preview_text

from .constants import CRITERIA, MODULE_CATEGORIES
from .types import IpCandidate, IpReusePlan, ModuleSpec, SystemRequirements


def _generation_context_sections(
    original_spec: Optional[str],
    reuse_modules: Optional[Dict[str, str]],
    environment_notes: Optional[Iterable[str]],
) -> str:
    sections: List[str] = []
    if reuse_modules:
        signatures = json.dumps(reuse_modules, indent=2, ensure_ascii=False)
        sections.append(
            "Provided reusable modules (each is supplied as a separate source file in the compile "
            "environment; instantiate them using these exact module names and port declarations; "
            "do NOT re-declare or re-implement any of them):\n"
            f"{signatures}"
        )
    if environment_notes:
        notes = "\n".join(f"- {note}" for note in environment_notes)
        sections.append(f"Compile environment notes:\n{notes}")
    if original_spec:
        sections.append(
            "Original design specification (authoritative for the top module name, exact port "
            "names, directions, widths, parameters, reset polarity, and behavior — follow it "
            "even where the plan disagrees):\n"
            f"{original_spec}"
        )
    return ("\n\n".join(sections) + "\n\n") if sections else ""


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


def build_rtl_generation_prompt(
    plan: IpReusePlan,
    target_hdl: str,
    top_module: str | None,
    *,
    original_spec: Optional[str] = None,
    reuse_modules: Optional[Dict[str, str]] = None,
    environment_notes: Optional[Iterable[str]] = None,
) -> str:
    plan_json = json.dumps(asdict(plan), default=json_default, indent=2)
    top = top_module or "choose a suitable top module name from the requirements"
    context = _generation_context_sections(original_spec, reuse_modules, environment_notes)
    return f"""Generate integrated {target_hdl} RTL from this IP reuse plan.
Use selected reusable IP behavior where appropriate, configure or adapt candidates as described, and create new RTL for modules marked new.
Reused modules listed as provided source files must only be instantiated, never re-declared in your output.
Return exactly one fenced {target_hdl} code block and no extra prose.

Top module: {top}

{context}IP reuse plan:
{plan_json}"""


def build_repair_prompt(
    plan: IpReusePlan,
    rtl: str,
    diagnostics: List[Dict[str, Any]],
    target_hdl: str,
    top_module: str | None,
    *,
    original_spec: Optional[str] = None,
    reuse_modules: Optional[Dict[str, str]] = None,
    environment_notes: Optional[Iterable[str]] = None,
    repair_hints: Optional[List[str]] = None,
) -> str:
    plan_json = json.dumps(asdict(plan), default=json_default, indent=2)
    context = _generation_context_sections(original_spec, reuse_modules, environment_notes)
    hints_section = ""
    if repair_hints:
        hint_text = "\n\n".join(hint.strip() for hint in repair_hints if hint.strip())
        if hint_text:
            hints_section = (
                "Guidance from previously verified fixes of similar diagnostics "
                "(advisory only; adapt the pattern, do not copy unrelated code or module names):\n"
                f"{hint_text}\n\n"
            )
    return f"""Repair this integrated {target_hdl} RTL so it passes syntax and lint checks.
Keep the same IP reuse intent and public top-level behavior.
Reused modules listed as provided source files must only be instantiated, never re-declared in your output.
Return exactly one fenced {target_hdl} code block and no extra prose.

Top module: {top_module or "unknown"}

{context}IP reuse plan:
{plan_json}

{hints_section}Diagnostics:
{json.dumps(diagnostics, default=json_default, indent=2)}

Current RTL:
```{target_hdl}
{rtl}
```"""


def build_spec_partition_prompt(
    prompt: str,
    target_hdl: str,
    top_module: str | None,
    constraints: Iterable[str],
) -> str:
    constraint_text = "\n".join(f"- {item}" for item in constraints) if constraints else "- none"
    return f"""Partition a large hardware specification into a precise one-layer implementation manifest.
Do not generate RTL. Do not invent ports, widths, protocols, timing, reset behavior, parameters, or module relationships.
Preserve every stated numerical, timing, protocol, clock, reset, and conditional-compilation requirement.
Use "unknown" or an unknowns entry whenever the source does not specify a detail.
The caller-required top module is {top_module or "not provided"}; use it exactly when provided.
List only the top module's immediate child modules. Do not recursively list grandchildren or deeper descendants.
An immediate child may later be decomposed by another planning call.

Target HDL: {target_hdl}
Extra constraints:
{constraint_text}

Full hardware specification:
{prompt}

Return only JSON with this exact shape:
{{
  "system_summary": "concise complete summary",
  "clocks": ["clock requirements"],
  "resets": ["reset requirements"],
  "parameters": ["parameter and macro requirements"],
  "shared_constraints": ["requirements applying across modules"],
  "assumptions": ["only explicit or unavoidable assumptions"],
  "unknowns": ["important unspecified details"],
  "top_module": {{
    "name": "exact top module name",
    "ports": [
      {{"name": "port", "direction": "input|output|inout", "width": "width or unknown", "description": "behavior"}}
    ],
    "instances": [
      {{
        "module": "submodule name",
        "instance_name": "instance name",
        "connections": {{"submodule_port": "top-level signal or expression"}}
      }}
    ]
  }},
  "modules": [
    {{
      "name": "HDL-safe submodule name",
      "category": "module category",
      "purpose": "module purpose",
      "ports": [
        {{"name": "port", "direction": "input|output|inout", "width": "width or unknown", "description": "behavior"}}
      ],
      "behavioral_requirements": ["complete module-scoped requirements"],
      "dependencies": ["direct submodule dependencies"],
      "reuse_query": "semantic retrieval query"
    }}
  ]
}}"""


def build_chunk_partition_prompt(
    chunk: str,
    chunk_index: int,
    chunk_count: int,
    target_hdl: str,
    top_module: str | None,
) -> str:
    return f"""Extract one-layer implementation facts from chunk {chunk_index} of {chunk_count} of a large hardware specification.
Do not generate RTL and do not invent missing facts. Preserve exact names, ports, widths, connections, protocols, clocks,
resets, parameters, conditional requirements, and module dependencies. Use "unknown" for unspecified values.
The expected top module is {top_module or "not provided"}. Target HDL is {target_hdl}.
Extract only immediate children of the top module; do not flatten grandchildren into the same module list.

Specification chunk:
{chunk}

Return only JSON. Use the same manifest fields when present: system_summary, clocks, resets, parameters,
shared_constraints, assumptions, unknowns, top_module, and modules. Omit fields that this chunk does not describe."""


def build_manifest_merge_prompt(
    partial_manifests: List[Dict[str, Any]],
    target_hdl: str,
    top_module: str | None,
    constraints: Iterable[str],
) -> str:
    constraint_text = "\n".join(f"- {item}" for item in constraints) if constraints else "- none"
    return f"""Merge partial large-hardware-specification extracts into one canonical one-layer implementation manifest.
Do not generate RTL. Do not invent facts. Deduplicate compatible facts and preserve all exact requirements.
Resolve conflicts only when one extract is clearly more specific; otherwise record the conflict in unknowns.
The caller-required top module is {top_module or "not provided"} and must be used exactly when provided.
Keep only the top module's immediate children in modules. Do not flatten deeper descendants into this manifest.

Target HDL: {target_hdl}
Extra constraints:
{constraint_text}

Partial extracts:
{json.dumps(partial_manifests, default=json_default, indent=2)}

Return only JSON using this complete shape:
{{
  "system_summary": "summary",
  "clocks": ["requirements"],
  "resets": ["requirements"],
  "parameters": ["requirements"],
  "shared_constraints": ["requirements"],
  "assumptions": ["assumptions"],
  "unknowns": ["unknowns"],
  "top_module": {{
    "name": "exact top name",
    "ports": [{{"name": "port", "direction": "input|output|inout", "width": "width or unknown", "description": "behavior"}}],
    "instances": [{{"module": "submodule", "instance_name": "instance", "connections": {{"submodule_port": "signal"}}}}]
  }},
  "modules": [{{
    "name": "submodule",
    "category": "category",
    "purpose": "purpose",
    "ports": [{{"name": "port", "direction": "input|output|inout", "width": "width or unknown", "description": "behavior"}}],
    "behavioral_requirements": ["requirements"],
    "dependencies": ["direct dependencies"],
    "reuse_query": "retrieval query"
  }}]
}}"""


def build_manifest_correction_prompt(
    manifest: Dict[str, Any],
    errors: List[str],
    target_hdl: str,
    top_module: str | None,
) -> str:
    return f"""Correct this large hardware specification manifest so it satisfies the listed validation errors.
Do not generate RTL and do not invent missing hardware facts. Preserve all valid requirements.
Use "unknown" or unknowns entries where the source facts are incomplete.
Target HDL: {target_hdl}
Required top module: {top_module or "not provided"}

Validation errors:
{json.dumps(errors, indent=2)}

Manifest:
{json.dumps(manifest, default=json_default, indent=2)}

Return only the corrected complete manifest JSON."""


def build_recursive_decomposition_prompt(
    module_spec_text: str,
    target_hdl: str,
    depth: int,
    max_depth: int,
    known_module_names: List[str],
) -> str:
    return f"""Decide whether one hardware module should be implemented as a leaf or decomposed by exactly one hierarchy layer.
Do not generate RTL. Do not describe grandchildren or deeper descendants.
Preserve the parent module's exact name and public port interface.
Choose "leaf" when one focused RTL generation call can implement the module reliably, or further splitting would only
create trivial gates/wires. Choose "decompose" when the module contains meaningful independently implementable blocks.
When decomposing, child module names must be HDL-safe, globally unique, and preferably prefixed with the parent name.
Do not reuse any of these existing module names: {json.dumps(known_module_names)}.
Existing direct dependencies listed in the parent specification remain external dependencies; do not repeat them as children.

Target HDL: {target_hdl}
Current depth: {depth}
Maximum recursive depth: {max_depth}

Parent module specification:
{module_spec_text}

Return only JSON:
{{
  "decision": "leaf|decompose",
  "reason": "short reason",
  "parent_module": {{
    "name": "exact unchanged parent module name",
    "ports": [
      {{"name": "port", "direction": "input|output|inout", "width": "width or unknown", "description": "behavior"}}
    ],
    "instances": [
      {{
        "module": "immediate child module",
        "instance_name": "instance name",
        "connections": {{"child_port": "parent signal or expression"}}
      }}
    ]
  }},
  "children": [
    {{
      "name": "globally unique immediate child module name",
      "category": "module category",
      "purpose": "module purpose",
      "ports": [
        {{"name": "port", "direction": "input|output|inout", "width": "width or unknown", "description": "behavior"}}
      ],
      "behavioral_requirements": ["complete child-scoped requirements"],
      "dependencies": ["direct dependencies among these immediate children"],
      "reuse_query": "semantic retrieval query"
    }}
  ]
}}

For "leaf", children and parent_module.instances must both be empty."""


def build_recursive_decomposition_correction_prompt(
    decomposition: Dict[str, Any],
    errors: List[str],
    module_spec_text: str,
    target_hdl: str,
) -> str:
    return f"""Correct this one-layer module decomposition so it satisfies the validation errors.
Do not generate RTL, invent hardware facts, change the parent public interface, or add grandchildren.
Return only the corrected decomposition JSON.

Target HDL: {target_hdl}
Parent module specification:
{module_spec_text}

Validation errors:
{json.dumps(errors, indent=2)}

Current decomposition:
{json.dumps(decomposition, default=json_default, indent=2)}"""


def build_scoped_module_generation_prompt(
    module_spec_text: str,
    dependency_interfaces: Dict[str, Any],
    decision: Dict[str, Any],
    target_hdl: str,
) -> str:
    return f"""Generate exactly one self-contained {target_hdl} module from this module-scoped specification.
Return exactly one fenced {target_hdl} code block and no extra prose.
The code block must declare only the requested module. It may instantiate only the listed direct dependencies.
Preserve every specified port name, direction, width, behavior, clock, reset, protocol, and parameter requirement.
Do not implement the top-level wrapper or unrelated modules.

Module specification:
{module_spec_text}

Direct dependency interfaces:
{json.dumps(dependency_interfaces, default=json_default, indent=2)}

IP reuse decision:
{json.dumps(decision, default=json_default, indent=2)}"""


def build_scoped_module_repair_prompt(
    module_spec_text: str,
    dependency_interfaces: Dict[str, Any],
    rtl: str,
    diagnostics: List[Dict[str, Any]],
    target_hdl: str,
) -> str:
    return f"""Repair exactly one {target_hdl} module so it passes syntax and lint checks.
Keep its exact public interface and module-scoped behavior. Do not emit dependency implementations or a top wrapper.
Return exactly one fenced {target_hdl} code block and no extra prose.

Module specification:
{module_spec_text}

Direct dependency interfaces:
{json.dumps(dependency_interfaces, default=json_default, indent=2)}

Diagnostics:
{json.dumps(diagnostics, default=json_default, indent=2)}

Current module:
```{target_hdl}
{rtl}
```"""


def build_top_wrapper_generation_prompt(
    index_text: str,
    module_signatures: Dict[str, str],
    target_hdl: str,
    top_module: str,
) -> str:
    return f"""Generate exactly one {target_hdl} top-level wrapper module named {top_module}.
Return exactly one fenced {target_hdl} code block and no extra prose.
Declare only {top_module}; do not repeat submodule implementations.
Instantiate and connect the supplied submodules according to the implementation index.
Preserve the exact public top-level interface and all stated connection, clock, reset, and parameter requirements.

Implementation index:
{index_text}

Available submodule signatures:
{json.dumps(module_signatures, indent=2)}"""


def build_testbench_generation_prompt(module_spec_text: str, rtl: str, target_hdl: str) -> str:
    return f"""Write a self-checking {target_hdl} testbench for the module below.
Requirements:
- Instantiate the module under test exactly once.
- Apply representative input vectors covering normal and edge cases.
- Use $display to log intermediate results.
- On any output mismatch call $fatal(1, "FAIL: <reason>").
- At the end of a successful run call $finish.
- Return ONLY a single fenced {target_hdl} code block containing the testbench. No prose.

Module specification:
{module_spec_text}

Module implementation:
```{target_hdl}
{rtl}
```"""


def build_top_wrapper_repair_prompt(
    index_text: str,
    module_signatures: Dict[str, str],
    wrapper_rtl: str,
    diagnostics: List[Dict[str, Any]],
    target_hdl: str,
    top_module: str,
) -> str:
    return f"""Repair exactly one {target_hdl} top-level wrapper module named {top_module}.
Return exactly one fenced {target_hdl} code block and no extra prose.
Do not emit or alter submodule implementations. Preserve the exact public top-level interface and intended connections.

Implementation index:
{index_text}

Available submodule signatures:
{json.dumps(module_signatures, indent=2)}

Combined-design diagnostics:
{json.dumps(diagnostics, default=json_default, indent=2)}

Current top wrapper:
```{target_hdl}
{wrapper_rtl}
```"""
