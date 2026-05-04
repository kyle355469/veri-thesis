from __future__ import annotations

from typing import Iterable, List, Optional

from .types import Diagnostic, RetrievalHit, RtlTask

SYSTEM_PROMPT = """You generate synthesizable RTL.
Return only the final HDL code in one fenced code block.
Do not include private reasoning, chain-of-thought, prose explanations, or hidden analysis."""


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


def build_generation_prompt(task: RtlTask, hits: List[RetrievalHit], diagnostics: Optional[List[Diagnostic]] = None) -> str:
    constraints = "\n".join(f"- {item}" for item in task.constraints) if task.constraints else "- Follow the user prompt exactly."
    signature = task.module_signature or "Not provided."
    retrieved = "\n\n".join(_format_hit(hit, index + 1) for index, hit in enumerate(hits))
    diagnostic_text = format_diagnostics(diagnostics or [])
    repair_instruction = (
        "\nRepair the previous RTL using the diagnostics. Preserve the requested interface."
        if diagnostic_text
        else ""
    )
    return f"""{SYSTEM_PROMPT}

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

{diagnostic_text}
{repair_instruction}

Return only:
```{task.target_hdl}
...code...
```"""
