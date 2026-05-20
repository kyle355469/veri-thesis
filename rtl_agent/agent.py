from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from rag_rtl.json_utils import json_default, preview_text
from rag_rtl.llm import extract_code
from rag_rtl.tool_calling import RTL_TOOL_SCHEMAS
from rag_rtl.types import RtlTask, VerificationReport
from rag_rtl.verifier import RtlVerifier

from .events import AgentEvent, summarize_tool_result
from .prompts import build_system_prompt, build_user_prompt


EventSink = Callable[[AgentEvent], None]


@dataclass(frozen=True)
class AgentConfig:
    temperature: float = 0.2
    max_tokens: int = 32768
    tool_choice: Any = "auto"
    max_steps: int = 8
    target_hdl: str = "verilog"
    final_verify: bool = True


@dataclass
class AgentResult:
    rtl: str
    final_text: str
    verification: VerificationReport
    events: List[AgentEvent] = field(default_factory=list)
    steps: int = 0
    used_tools: bool = False
    stopped_reason: str = "final"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rtl": self.rtl,
            "final_text": self.final_text,
            "verification": self.verification,
            "events": [event.to_dict() for event in self.events],
            "steps": self.steps,
            "used_tools": self.used_tools,
            "stopped_reason": self.stopped_reason,
        }


class AgenticRtlAgent:
    def __init__(
        self,
        llm_client: Any,
        tool_executor: Any,
        verifier: RtlVerifier,
        config: Optional[AgentConfig] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.llm = llm_client
        self.tool_executor = tool_executor
        self.verifier = verifier
        self.config = config or AgentConfig()
        self.tool_schemas = tool_schemas or RTL_TOOL_SCHEMAS

    def run(self, task: RtlTask, event_sink: Optional[EventSink] = None) -> AgentResult:
        events: List[AgentEvent] = []

        def record(event: AgentEvent) -> None:
            events.append(event)
            if event_sink:
                event_sink(event)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(task.target_hdl or self.config.target_hdl)},
            {
                "role": "user",
                "content": build_user_prompt(
                    prompt=task.prompt,
                    target_hdl=task.target_hdl or self.config.target_hdl,
                    module_signature=task.module_signature,
                    constraints=task.constraints,
                ),
            },
        ]
        final_text = ""
        stopped_reason = "max_steps"
        used_tools = False
        step_count = 0

        record(AgentEvent("agent_start", 0, "agent started"))
        for step in range(1, self.config.max_steps + 1):
            step_count = step
            message = self.llm.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=self.tool_schemas,
                tool_choice=self.config.tool_choice,
                parallel_tool_calls=False,
            )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_text = message.get("content") or ""
                stopped_reason = "final"
                record(
                    AgentEvent(
                        "model_final",
                        step,
                        f"model returned final response: {preview_text(final_text, 180)}",
                        data={"content_preview": preview_text(final_text)},
                    )
                )
                break

            used_tools = True
            messages.append(_assistant_tool_call_message(message))
            for index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = _parse_tool_arguments(function.get("arguments", "{}"))
                record(
                    AgentEvent(
                        "tool_call",
                        step,
                        f"model chose tool {name}",
                        tool=name,
                        data={"arguments": _preview_arguments(arguments)},
                    )
                )
                result_text = self.tool_executor.execute(name, arguments)
                result_payload = _parse_tool_result(result_text)
                record(
                    AgentEvent(
                        "tool_result",
                        step,
                        summarize_tool_result(name, result_payload),
                        tool=name,
                        data={"result_preview": preview_text(result_text)},
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or f"tool_call_{step}_{index}",
                        "name": name,
                        "content": result_text,
                    }
                )

        if not final_text:
            final_text = self._force_final(messages)
            stopped_reason = "forced_final"
            record(
                AgentEvent(
                    "forced_final",
                    step_count,
                    f"tool budget exhausted; requested final response: {preview_text(final_text, 180)}",
                    data={"content_preview": preview_text(final_text)},
                )
            )

        rtl = extract_code(final_text)
        verification = self.verifier.verify(rtl, top_module=task.top_module) if rtl else _empty_extraction_report()
        record(
            AgentEvent(
                "final_verification",
                -1,
                _verification_summary(verification),
                data={"passed": verification.passed},
            )
        )
        return AgentResult(
            rtl=rtl,
            final_text=final_text,
            verification=verification,
            events=events,
            steps=step_count,
            used_tools=used_tools,
            stopped_reason=stopped_reason,
        )

    def _force_final(self, messages: Sequence[Dict[str, Any]]) -> str:
        final_messages = list(messages) + [
            {
                "role": "user",
                "content": "Tool-call budget is exhausted. Return exactly one final fenced RTL code block now.",
            }
        ]
        message = self.llm.chat(
            final_messages,
            temperature=min(self.config.temperature, 0.1),
            max_tokens=self.config.max_tokens,
            tools=self.tool_schemas,
            tool_choice="none",
            parallel_tool_calls=False,
        )
        return message.get("content") or ""


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw_arguments": str(arguments)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _parse_tool_result(result_text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(result_text)
    except json.JSONDecodeError:
        return {"ok": False, "error": result_text}
    return payload if isinstance(payload, dict) else {"ok": True, "value": payload}


def _assistant_tool_call_message(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
    }


def _preview_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    preview: Dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            preview[key] = preview_text(value, 240)
        else:
            preview[key] = value
    return preview


def _empty_extraction_report() -> VerificationReport:
    from rag_rtl.types import Diagnostic

    return VerificationReport(
        syntax_passed=False,
        lint_passed=False,
        diagnostics=[
            Diagnostic(
                tool="rtl_extraction",
                passed=False,
                stderr="final model response did not contain parsable RTL",
            )
        ],
    )


def _verification_summary(verification: VerificationReport) -> str:
    status = "passed" if verification.passed else "failed"
    failed_tools = [item.tool for item in verification.diagnostics if not item.passed]
    suffix = f"; failed tools: {', '.join(failed_tools)}" if failed_tools else ""
    return f"final verification {status}{suffix}"


def dumps_result(result: AgentResult, *, indent: Optional[int] = 2) -> str:
    return json.dumps(result.to_dict(), default=json_default, ensure_ascii=False, indent=indent)
