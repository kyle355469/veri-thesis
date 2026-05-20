from __future__ import annotations

from typing import Iterable, Optional

from rag_rtl.siliconmind_utils import wrap_text


def build_system_prompt(target_hdl: str = "verilog") -> str:
    return f"""You are an agentic RTL implementation assistant.
You can decide when local tools are useful:
- retrieve_rtl_context searches a local RTL corpus for examples.
- run_yosys checks syntax/elaboration.
- run_verilator checks lint.
- verify_rtl runs the complete configured verifier.
- read_file, write_file, and list_dir inspect or update files inside the configured workspace.
- run_command runs allowed non-shell inspection commands such as rg, grep, ls, cat, sed, head, tail, and wc.

Use tools only when they help you make progress. If diagnostics fail, repair the RTL and call a verifier tool again when tool rounds remain.
Use workspace writes when the user asks for an artifact on disk, or when saving a useful candidate file helps the task.
When you are ready to finish, return exactly one fenced {target_hdl} code block containing the complete final RTL.
Do not include explanations, diagnostics, or markdown outside that final code block."""


def build_user_prompt(
    prompt: str,
    target_hdl: str = "verilog",
    module_signature: Optional[str] = None,
    constraints: Optional[Iterable[str]] = None,
) -> str:
    constraint_items = list(constraints or [])
    constraint_text = "\n".join(f"- {item}" for item in constraint_items) if constraint_items else "- Follow the user request exactly."
    signature = module_signature or "Not provided."
    return f"""Target HDL: {target_hdl}
Module signature: {signature}
Constraints:
{constraint_text}

User request:
{wrap_text(prompt)}

Return format:
```{target_hdl}
// complete RTL here
```"""
