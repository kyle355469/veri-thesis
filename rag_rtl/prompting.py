from __future__ import annotations

from typing import Iterable, List, Optional

from .history_cache import CacheLookup
from .siliconmind_utils import SYS_PROMPT_INTERNAL_WORKFLOW, wrap_code, wrap_text
from .types import Diagnostic, RetrievalHit, RtlTask

SYSTEM_PROMPT = SYS_PROMPT_INTERNAL_WORKFLOW

TOOL_CALL_GUIDE = """If tool calling is available, use tools before the final answer when they help:
1. Call retrieve_rtl_context only when the provided context is insufficient or diagnostics suggest a missing pattern.
2. Draft the candidate RTL internally.
3. Call run_yosys and run_verilator, or call verify_rtl when a full configured check is needed.
4. If a tool reports an error, repair the candidate and re-check before final output when tool rounds remain.
5. After emitting the final fenced code block, stop."""


def _return_format(target_hdl: str) -> str:
    return f"""### Output Format
Return only one fenced {target_hdl} code block containing the complete RTL.
Do not include explanations, diagnostics, markdown outside the code block, or extra text."""


def _format_hit(hit: RetrievalHit, index: int) -> str:
    doc = hit.document
    score = hit.rerank_score if hit.rerank_score is not None else hit.score
    return f"""### Retrieved Example {index}
Doc ID: {doc.doc_id}
Score: {score:.4f}
Tags: {", ".join(doc.tags)}

Problem:
{doc.problem}

Solution:
{wrap_code(doc.solution)}
"""


def format_diagnostics(diagnostics: Iterable[Diagnostic]) -> str:
    parts: List[str] = []
    for diagnostic in diagnostics:
        if diagnostic.passed:
            continue
        missing = " missing=true" if diagnostic.missing else ""
        parts.append(
            f"""Tool: {diagnostic.tool}
Return code: {diagnostic.returncode}{missing}
STDOUT:
{diagnostic.stdout[-3000:]}
STDERR:
{diagnostic.stderr[-3000:]}"""
        )
    return "\n".join(parts)


def _format_history_evidence(history_lookup: Optional[CacheLookup]) -> str:
    if not history_lookup or not history_lookup.evidence_entry:
        return "No semantic history evidence available."
    entry = history_lookup.evidence_entry
    return f"""Matched query: {entry.query}
Score: {history_lookup.score:.4f}
{wrap_code(entry.result)}
"""


def build_generation_prompt(
    task: RtlTask,
    hits: List[RetrievalHit],
    diagnostics: Optional[List[Diagnostic]] = None,
    history_lookup: Optional[CacheLookup] = None,
) -> str:
    constraints = "\n".join(f"- {item}" for item in task.constraints) if task.constraints else "- Follow the user prompt exactly."
    signature = task.module_signature or "Not provided."
    retrieved = "\n\n".join(_format_hit(hit, index + 1) for index, hit in enumerate(hits))
    history_evidence = _format_history_evidence(history_lookup)
    diagnostic_text = format_diagnostics(diagnostics or [])
    repair_instruction = (
        "\nRepair the previous RTL using the diagnostics. Preserve the requested interface."
        if diagnostic_text
        else ""
    )
    return f"""{SYSTEM_PROMPT}

{TOOL_CALL_GUIDE}

### Verilog Coding Problem
Target HDL: {task.target_hdl}
Module signature: {signature}
Constraints:
{constraints}

User request:
{wrap_text(task.prompt)}

### Retrieved Context
{retrieved or "No retrieved documents available."}

### Semantic History Evidence
{history_evidence}

### Verification Diagnostics
{diagnostic_text or "No verification diagnostics available."}
{repair_instruction}

{_return_format(task.target_hdl)}"""


def build_second_edition_prompt(
    task: RtlTask,
    first_edition_rtl: str,
    first_edition_datapath: str,
    structure_hits: List[RetrievalHit],
    diagnostics: Optional[List[Diagnostic]] = None,
) -> str:
    constraints = "\n".join(f"- {item}" for item in task.constraints) if task.constraints else "- Follow the user prompt exactly."
    signature = task.module_signature or "Not provided."
    retrieved = "\n\n".join(_format_hit(hit, index + 1) for index, hit in enumerate(structure_hits))
    diagnostic_text = format_diagnostics(diagnostics or [])
    repair_instruction = (
        "\nRepair the second-edition RTL using the diagnostics. Preserve the requested interface and verified behavior."
        if diagnostic_text
        else ""
    )
    return f"""{SYSTEM_PROMPT}

{TOOL_CALL_GUIDE}

### Verilog Coding Problem
Target HDL: {task.target_hdl}
Module signature: {signature}
Constraints:
{constraints}

User request:
{wrap_text(task.prompt)}

### First-Edition Verified RTL
{wrap_code(first_edition_rtl, task.target_hdl)}

### First-Edition Datapath
{first_edition_datapath or "No datapath graph was available."}

### Retrieved Code-Structure Context
{retrieved or "No code-structure documents available."}

### Verification Diagnostics
{diagnostic_text or "No verification diagnostics available."}
{repair_instruction}

Produce a second-edition RTL implementation. Keep the same external behavior and interface, but use the datapath and code-structure context to improve structural alignment.

{_return_format(task.target_hdl)}"""
