from __future__ import annotations

from typing import Iterable, List, Optional

from .generation import AttemptFeedback
from .history_cache import CacheLookup
from .siliconmind_utils import SYS_PROMPT_INTERNAL_WORKFLOW, wrap_code, wrap_text
from .types import Diagnostic, RetrievalHit, RtlTask

SYSTEM_PROMPT = SYS_PROMPT_INTERNAL_WORKFLOW

MODEL_ONLY_PROMPT = """Generate the requested Verilog/SystemVerilog implementation.
Think internally if needed, but return only the final code."""

TOOL_CALL_GUIDE = """If tool calling is available, use tools before the final answer when they help:
1. Call retrieve_rtl_context only when the provided context is insufficient or diagnostics suggest a missing pattern.
2. Draft the candidate RTL internally.
3. Call run_yosys and run_verilator, or call verify_rtl when a full configured check is needed.
4. If a tool reports an error, repair the candidate and re-check before final output when tool rounds remain.
5. After emitting the final fenced code block, stop."""


def _return_format(target_hdl: str) -> str:
    return f"""### Output Format
Return only one fenced {target_hdl} code block containing the complete RTL.
Start the response immediately with ```{target_hdl}.
Do not include explanations, diagnostics, markdown outside the code block, or extra text.
If the problem is difficult, still output the simplest complete compilable RTL that matches the requested interface."""


def _compact_task_text(task: RtlTask) -> str:
    constraints = "\n".join(f"- {item}" for item in task.constraints) if task.constraints else "- Follow the user prompt exactly."
    signature = task.module_signature or "Not provided."
    return f"""Target HDL: {task.target_hdl}
Module signature: {signature}
Constraints:
{constraints}

User request:
{wrap_text(task.prompt)}"""


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
    feedback: Optional[AttemptFeedback] = None,
    history_lookup: Optional[CacheLookup] = None,
) -> str:
    profile = task.prompt_profile
    constraints = "\n".join(f"- {item}" for item in task.constraints) if task.constraints else "- Follow the user prompt exactly."
    signature = task.module_signature or "Not provided."
    retrieved = "\n\n".join(_format_hit(hit, index + 1) for index, hit in enumerate(hits))
    history_evidence = _format_history_evidence(history_lookup)
    retry_text = "" if profile == "model" and feedback is None else _format_generation_retry_feedback(feedback, task.target_hdl)
    header_parts = [MODEL_ONLY_PROMPT if profile == "model" else SYSTEM_PROMPT]
    if profile in {"tool", "full"}:
        header_parts.append(TOOL_CALL_GUIDE)

    problem_title = "### Verilog Coding Problem" if profile != "model" else "### Coding Problem"
    problem_section = f"""{problem_title}
Target HDL: {task.target_hdl}
Module signature: {signature}
Constraints:
{constraints}

User request:
{wrap_text(task.prompt)}"""

    context_sections = []
    if profile in {"rag", "full"}:
        context_sections.extend(
            [
                f"""### Retrieved Context
{retrieved or "No retrieved documents available."}""",
                f"""### Semantic History Evidence
{history_evidence}""",
            ]
        )

    return "\n\n".join(
        [
            *header_parts,
            problem_section,
            *context_sections,
            retry_text,
            _return_format(task.target_hdl),
        ]
    )


def build_emergency_generation_prompt(
    task: RtlTask,
    previous_model_text: str = "",
) -> str:
    previous = previous_model_text.strip()
    previous_note = (
        f"\nPrevious non-code response preview:\n{wrap_text(previous[-1200:])}\n"
        if previous
        else ""
    )
    return f"""The previous response did not contain a parsable fenced HDL code block, or it spent too much output budget before producing RTL.

Output a minimal complete Verilog/SystemVerilog implementation now.
No reasoning. No explanation. No diagnostics. No markdown except the single fenced code block.
Start immediately with ```{task.target_hdl} and end with ```.
Prefer a simple compilable implementation over an optimized one.

### Verilog Coding Problem
{_compact_task_text(task)}
{previous_note}
{_return_format(task.target_hdl)}"""


def build_second_edition_prompt(
    task: RtlTask,
    first_edition_rtl: str,
    first_edition_datapath: str,
    structure_hits: List[RetrievalHit],
    feedback: Optional[AttemptFeedback] = None,
) -> str:
    constraints = "\n".join(f"- {item}" for item in task.constraints) if task.constraints else "- Follow the user prompt exactly."
    signature = task.module_signature or "Not provided."
    retrieved = "\n\n".join(_format_hit(hit, index + 1) for index, hit in enumerate(structure_hits))
    retry_text = _format_second_edition_retry_feedback(feedback, task.target_hdl)
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

{retry_text}

Produce a second-edition RTL implementation. Keep the same external behavior and interface, but use the datapath and code-structure context to improve structural alignment.

{_return_format(task.target_hdl)}"""


def build_emergency_second_edition_prompt(
    task: RtlTask,
    first_edition_rtl: str,
    previous_model_text: str = "",
) -> str:
    previous = previous_model_text.strip()
    previous_note = (
        f"\nPrevious non-code response preview:\n{wrap_text(previous[-1200:])}\n"
        if previous
        else ""
    )
    return f"""The previous second-edition response did not contain a parsable fenced HDL code block, or it spent too much output budget before producing RTL.

Output a complete second-edition RTL implementation now.
No reasoning. No explanation. No diagnostics. No markdown except the single fenced code block.
Start immediately with ```{task.target_hdl} and end with ```.
If uncertain, preserve the verified first-edition RTL behavior and interface.

### Verilog Coding Problem
{_compact_task_text(task)}

### Verified First-Edition RTL
{wrap_code(first_edition_rtl, task.target_hdl)}
{previous_note}
{_return_format(task.target_hdl)}"""


def _format_generation_retry_feedback(
    feedback: Optional[AttemptFeedback],
    target_hdl: str,
) -> str:
    if feedback is None:
        return """### Verification Diagnostics
No verification diagnostics available."""
    if feedback.kind == "extraction":
        return f"""### Retry Instruction
The previous response did not contain a parsable fenced HDL code block. Do not include reasoning, analysis, explanations, or extra markdown. Return exactly one fenced {target_hdl} code block containing the complete final RTL."""

    diagnostic_text = format_diagnostics(feedback.diagnostics)
    return f"""### Previous RTL
{wrap_code(feedback.previous_rtl, target_hdl)}

### Verification Diagnostics
{diagnostic_text or "No verification diagnostics available."}

Repair the previous RTL using the diagnostics. Preserve the requested interface and return only the corrected complete RTL."""


def _format_second_edition_retry_feedback(
    feedback: Optional[AttemptFeedback],
    target_hdl: str,
) -> str:
    if feedback is None:
        return """### Verification Diagnostics
No verification diagnostics available."""
    if feedback.kind == "extraction":
        return f"""### Retry Instruction
The previous response did not contain a parsable fenced HDL code block. Do not include reasoning, analysis, explanations, or extra markdown. Return exactly one fenced {target_hdl} code block containing the complete final RTL."""

    diagnostic_text = format_diagnostics(feedback.diagnostics)
    return f"""### Previous Second-Edition RTL
{wrap_code(feedback.previous_rtl, target_hdl)}

### Verification Diagnostics
{diagnostic_text or "No verification diagnostics available."}

Repair the second-edition RTL using the diagnostics. Preserve the requested interface and verified behavior, and return only the corrected complete RTL."""
