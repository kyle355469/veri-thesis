from __future__ import annotations

from typing import Iterable, List, Optional

from .history_cache import CacheLookup
from .types import Diagnostic, RetrievalHit, RtlTask

SYSTEM_PROMPT = """You generate synthesizable RTL.
Do not include private reasoning, chain-of-thought, analysis prose, or explanations in the final answer.
When you are ready to answer, output one complete RTL implementation and stop immediately."""

TOOL_CALL_POLICY = """<tool_call_policy>
If tool calling is available in this generation step, use the tools before final output:
1. Call retrieve_rtl_context only when the provided context is insufficient or diagnostics suggest a missing pattern.
2. Draft the candidate RTL privately.
3. Call run_yosys and run_verilator, or call verify_rtl when a full configured check is needed.
4. If a tool reports an error, repair the candidate and re-check before final output when tool rounds remain.
5. After the final RTL is emitted, do not call any more tools and do not continue reasoning.
</tool_call_policy>"""

FINAL_RTL_POLICY = """<final_output_policy>
The final answer for this generation step must contain exactly one <final_rtl> block.
Do not print, quote, demonstrate, or repeat any extra <final_rtl> blocks.
Place the complete HDL code inside that single block.
After writing the closing </final_rtl> tag, stop immediately. Do not add prose, diagnostics, notes, markdown outside the block, or further reasoning.
</final_output_policy>"""


def _return_format(target_hdl: str) -> str:
    return f"""Return format:
Open exactly one <final_rtl> block, put one fenced {target_hdl} code block containing the complete RTL inside it, close </final_rtl>, then stop."""


def _format_hit(hit: RetrievalHit, index: int) -> str:
    doc = hit.document
    score = hit.rerank_score if hit.rerank_score is not None else hit.score
    return f"""<retrieved_document index="{index}" doc_id="{doc.doc_id}" score="{score:.4f}" tags="{','.join(doc.tags)}">
<problem>
{doc.problem}
</problem>
<solution>
```verilog
{doc.solution}
```
</solution>
</retrieved_document>"""


def format_diagnostics(diagnostics: Iterable[Diagnostic]) -> str:
    parts: List[str] = []
    for diagnostic in diagnostics:
        if diagnostic.passed:
            continue
        missing = " missing=true" if diagnostic.missing else ""
        parts.append(
            f"""<diagnostic tool="{diagnostic.tool}" returncode="{diagnostic.returncode}"{missing}>
STDOUT:
{diagnostic.stdout[-3000:]}
STDERR:
{diagnostic.stderr[-3000:]}
</diagnostic>"""
        )
    return "\n".join(parts)


def _format_history_evidence(history_lookup: Optional[CacheLookup]) -> str:
    if not history_lookup or not history_lookup.evidence_entry:
        return "No semantic history evidence available."
    entry = history_lookup.evidence_entry
    return f"""<history_example score="{history_lookup.score:.4f}" matched_query="{entry.query}">
```verilog
{entry.result}
```
</history_example>"""


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

{TOOL_CALL_POLICY}

{FINAL_RTL_POLICY}

<task>
Target HDL: {task.target_hdl}
Module signature: {signature}
Constraints:
{constraints}

User request:
{task.prompt}
</task>

<retrieved_context>
{retrieved or "No retrieved documents available."}
</retrieved_context>

<semantic_history_evidence>
{history_evidence}
</semantic_history_evidence>

{diagnostic_text}
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

{TOOL_CALL_POLICY}

{FINAL_RTL_POLICY}

<task>
Target HDL: {task.target_hdl}
Module signature: {signature}
Constraints:
{constraints}

User request:
{task.prompt}
</task>

<first_edition_verified_rtl>
```{task.target_hdl}
{first_edition_rtl}
```
</first_edition_verified_rtl>

<first_edition_datapath>
{first_edition_datapath or "No datapath graph was available."}
</first_edition_datapath>

<retrieved_code_structure_context>
{retrieved or "No code-structure documents available."}
</retrieved_code_structure_context>

{diagnostic_text}
{repair_instruction}

Produce a second-edition RTL implementation. Keep the same external behavior and interface, but use the datapath and code-structure context to improve structural alignment.

{_return_format(task.target_hdl)}"""
