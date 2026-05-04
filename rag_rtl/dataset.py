from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .types import RtlDocument

THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ANSWER_CODE_RE = re.compile(
    r"<answer>\s*```(?:verilog|systemverilog|sv)?\s*(.*?)```\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
FENCED_CODE_RE = re.compile(r"```(?:verilog|systemverilog|sv)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return ""


def strip_private_reasoning(text: str) -> str:
    """Remove private reasoning traces while preserving final answers."""
    return THINK_RE.sub("", text).strip()


def extract_rtl_code(text: str) -> str:
    cleaned = strip_private_reasoning(text)
    answer_match = ANSWER_CODE_RE.search(cleaned)
    if answer_match:
        return answer_match.group(1).strip()
    code_match = FENCED_CODE_RE.search(cleaned)
    if code_match:
        return code_match.group(1).strip()
    return cleaned.strip()


def infer_tags(problem: str, solution: str) -> List[str]:
    text = f"{problem}\n{solution}".lower()
    tags: List[str] = []
    for tag, needles in {
        "combinational": ["always @(*)", "assign ", "case "],
        "sequential": ["posedge", "negedge", "always @("],
        "fsm": ["fsm", "state", "case"],
        "memory": ["ram", "memory", "array", "addr"],
        "interface": ["axi", "valid", "ready", "interface"],
        "parameterized": ["parameter", "localparam"],
    }.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    return tags


def normalize_record(record: Dict[str, object], index: int) -> Optional[RtlDocument]:
    prompt = _message_text(record.get("prompt", ""))
    completion = _message_text(record.get("completion", ""))
    if not prompt or not completion:
        return None

    solution = extract_rtl_code(completion)
    if not solution:
        return None

    digest = hashlib.sha1(f"{index}:{prompt[:256]}".encode("utf-8")).hexdigest()[:16]
    return RtlDocument(
        doc_id=f"merged-{index}-{digest}",
        problem=prompt.strip(),
        solution=solution,
        tags=infer_tags(prompt, solution),
        metadata={
            "source": "merged.jsonl",
            "row": index,
            "private_reasoning_removed": completion != strip_private_reasoning(completion),
        },
    )


def iter_jsonl_documents(path: str | Path, limit: Optional[int] = None) -> Iterator[RtlDocument]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            document = normalize_record(record, index)
            if document is not None:
                yield document
