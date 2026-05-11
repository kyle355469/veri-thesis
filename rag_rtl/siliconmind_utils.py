from __future__ import annotations

import re
from typing import Dict, List

SILICONMIND_WORKFLOW_GUIDE = """SiliconMind-style internal workflow:
1. Draft a complete Verilog/SystemVerilog solution.
2. Privately self-check it against representative scenarios and the requested interface.
3. Repair any discovered issue before emitting the final code.

Keep the workflow internal. The visible answer must be code only in the requested wrapper."""

SYS_PROMPT_ANSWER_GUIDE = """Please solve the following Verilog coding problem.
Think internally about the implementation and any self-checks.
Then output only the Verilog code in a fenced code block."""

SYS_PROMPT_INTERNAL_WORKFLOW = """Please solve the following Verilog coding problem.
First draft a solution, then internally check it with representative scenarios.
If the attempted design is faulty, fix it before responding.
Then output only the corrected Verilog code in a fenced code block."""

SYS_PROMPT_QUANT_TEST_PT1 = """Please check whether the following Verilog design is syntactically correct and satisfies the problem requirements.
Derive representative test scenarios internally.
If the design is faulty, write [DESIGN NEEDS FIXING].
Otherwise, write [DESIGN IS CORRECT]."""

SYS_PROMPT_QUANT_TEST_PT2 = """Fix the attempted Verilog design using the provided error analysis.
Think internally about the correction, then output only the corrected Verilog code in a fenced code block."""

CODE_BLOCK_RE = re.compile(
    r"```(?:verilog|systemverilog|sv)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def parse_text(text: str) -> str:
    return text.strip()


def parse_code(text: str) -> str:
    matches = list(CODE_BLOCK_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return ""


def wrap_text(text: str) -> str:
    return '"""\n' + text.strip() + '\n"""'


def wrap_prompt(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def wrap_code(code: str, language: str = "verilog") -> str:
    return f"```{language}\n{code.strip()}\n```"


def get_attempt_prompts(problems: List[str], internal_workflow: bool) -> List[List[Dict[str, str]]]:
    prefix = SYS_PROMPT_INTERNAL_WORKFLOW if internal_workflow else SYS_PROMPT_ANSWER_GUIDE
    return [
        wrap_prompt(prefix + "\n\n### Verilog Coding Problem\n\n" + wrap_text(problem))
        for problem in problems
    ]


def get_test_prompts(problems: List[str], attempts: List[str]) -> List[List[Dict[str, str]]]:
    assert len(problems) == len(attempts)
    return [
        wrap_prompt(
            SYS_PROMPT_QUANT_TEST_PT1
            + "\n\n### Problem\n\n"
            + wrap_text(problem)
            + "\n\n### Verilog Design\n\n"
            + wrap_code(attempt)
        )
        for problem, attempt in zip(problems, attempts)
    ]


def get_debug_prompts(
    problems: List[str],
    attempts: List[str],
    error_analysis: List[str],
) -> List[List[Dict[str, str]]]:
    assert len(problems) == len(attempts) and len(problems) == len(error_analysis)
    return [
        wrap_prompt(
            SYS_PROMPT_QUANT_TEST_PT2
            + "\n\n### Verilog Design Problem\n\n"
            + wrap_text(problem)
            + "\n\n### Attempted Solution\n\n"
            + wrap_code(attempt)
            + "\n\n### Error Analysis\n\n"
            + wrap_text(error)
        )
        for problem, attempt, error in zip(problems, attempts, error_analysis)
    ]
