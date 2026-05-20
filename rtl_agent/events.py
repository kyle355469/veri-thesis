from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rag_rtl.json_utils import preview_text


@dataclass
class AgentEvent:
    event: str
    step: int
    message: str
    tool: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "event": self.event,
            "step": self.step,
            "message": self.message,
        }
        if self.tool:
            payload["tool"] = self.tool
        if self.data:
            payload["data"] = self.data
        return payload

    def render(self) -> str:
        prefix = f"step {self.step}" if self.step >= 0 else "final"
        return f"[{prefix}] {self.message}"


def summarize_tool_result(tool: str, result: Dict[str, Any]) -> str:
    if not result.get("ok", False):
        return f"{tool} failed: {preview_text(str(result.get('error', 'unknown error')), 180)}"
    if tool == "retrieve_rtl_context":
        hits = result.get("hits") or []
        doc_ids = [str(hit.get("doc_id")) for hit in hits[:3] if isinstance(hit, dict)]
        suffix = f" ({', '.join(doc_ids)})" if doc_ids else ""
        return f"{tool} returned {len(hits)} hits{suffix}"
    if tool in {"run_yosys", "run_verilator"}:
        diagnostic = result.get("diagnostic") or {}
        status = "passed" if diagnostic.get("passed") else "failed"
        detail = diagnostic.get("stderr") or diagnostic.get("stdout") or ""
        return f"{tool} {status}: {preview_text(detail, 180)}" if detail else f"{tool} {status}"
    if tool == "verify_rtl":
        status = "passed" if result.get("passed") else "failed"
        verification = result.get("verification") or {}
        diagnostics = verification.get("diagnostics") or []
        failed = [
            item.get("tool", "unknown")
            for item in diagnostics
            if isinstance(item, dict) and not item.get("passed", False)
        ]
        suffix = f" failed tools: {', '.join(failed)}" if failed else ""
        return f"{tool} {status}{suffix}"
    if tool == "read_file":
        path = result.get("path", "")
        start = result.get("start_line", "?")
        end = result.get("end_line", "?")
        truncated = " truncated" if result.get("truncated") else ""
        return f"read {path}:{start}-{end}{truncated}"
    if tool == "write_file":
        mode = "appended" if result.get("append") else "wrote"
        return f"{mode} {result.get('path', '')} ({result.get('bytes', 0)} bytes)"
    if tool == "list_dir":
        entries = result.get("entries") or []
        return f"listed {result.get('path', '')}: {len(entries)} entries"
    if tool == "run_command":
        status = "passed" if result.get("passed") else f"exit {result.get('returncode')}"
        argv = " ".join(str(item) for item in result.get("argv", []))
        stdout = result.get("stdout") or result.get("stderr") or ""
        suffix = f": {preview_text(stdout, 180)}" if stdout else ""
        return f"command {status}: {argv}{suffix}"
    return f"{tool} returned result"
