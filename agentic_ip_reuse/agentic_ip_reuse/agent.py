from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .artifacts import write_standard_artifacts
from .grounding import completeness_gaps, completeness_score, ground_reuse_decisions
from .json_utils import json_default, preview_text
from .prompts import build_catalog_digest, build_system_prompt, build_user_prompt, catalog_identifiers
from .tools import AgentToolExecutor, TOOL_SCHEMAS
from .types import AgentEvent, AgentResult, DesignTask


@dataclass(frozen=True)
class AgentConfig:
    temperature: float = 0.2
    max_tokens: int = 8192
    use_tools: bool = False
    tool_choice: Any = "auto"
    max_steps: int = 16
    max_final_nudges: int = 2
    inject_catalog: bool = True
    ground_reuse_decisions: bool = True
    completeness_gate: bool = True
    max_catalog_entries: int = 60


_FINAL_NUDGE_PROMPT = (
    "Your previous reply was working notes, not a result. Continue now and do one of these two things: "
    "either call one of the available tools, or return the complete final plan as a single JSON object "
    "with keys requirements, modules, reuse_decisions, integration_plan, verification_plan, debug_plan, "
    "unresolved_assumptions. Do not reply with notes or partial reasoning again."
)


def _completeness_prompt(gaps: Sequence[str], catalog_digest: str) -> str:
    parts = [
        "Your plan is missing required sections: " + ", ".join(gaps) + ".",
        "Return the COMPLETE plan again as one JSON object (all keys), this time filling those sections.",
    ]
    if "reuse_decisions" in gaps or "integration_plan" in gaps:
        parts.append(
            "For every catalog IP that fits a module, add a reuse_decisions entry "
            '{"module_name": ..., "selected_ip": <exact ip_id from the catalog>, "new_rtl_required": false}, '
            "and add an integration_plan step describing how each selected IP is instantiated and connected."
        )
        if catalog_digest:
            parts.append(catalog_digest)
    return "\n\n".join(parts)


@dataclass
class AgenticIpReuseAgent:
    llm_client: Any
    tool_executor: AgentToolExecutor
    config: AgentConfig = field(default_factory=AgentConfig)
    tool_schemas: List[Dict[str, Any]] = field(default_factory=lambda: list(TOOL_SCHEMAS))

    def run(self, task: DesignTask) -> AgentResult:
        events: List[AgentEvent] = []
        catalog_candidates = self._catalog_candidates()
        catalog_ids = catalog_identifiers(catalog_candidates)
        catalog_digest = (
            build_catalog_digest(catalog_candidates, self.config.max_catalog_entries)
            if self.config.inject_catalog
            else ""
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(task, catalog_digest)},
        ]
        final_text = ""
        stopped_reason = "max_steps"
        used_tools = False
        step_count = 0
        nudges_used = 0
        events.append(AgentEvent("agent_start", 0, "agent started"))

        for step in range(1, self.config.max_steps + 1):
            step_count = step
            chat_kwargs: Dict[str, Any] = {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            # Agentic tool calling is opt-in: the deployed reasoning parser returns
            # empty content (answer trapped in reasoning_content) whenever tools are
            # attached, and across every run the planner calls 0 tools anyway --
            # catalog injection + grounding replace it. Only attach tools when the
            # caller explicitly enables them (e.g. for a tool-capable model).
            if self.config.use_tools:
                chat_kwargs["tools"] = self.tool_schemas
                chat_kwargs["tool_choice"] = self.config.tool_choice
                chat_kwargs["parallel_tool_calls"] = False
            message = self.llm_client.chat(messages, **chat_kwargs)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                candidate_text = message.get("content") or ""
                if _has_plan_payload(candidate_text):
                    final_text = candidate_text
                    stopped_reason = "final"
                    events.append(AgentEvent("model_final", step, f"model returned final plan: {preview_text(final_text)}"))
                    break
                # Reasoning-model deployments often stop after emitting working
                # notes (empty content promoted from reasoning_content) without
                # calling a tool or producing the plan. Push back instead of
                # accepting the notes as a final answer.
                if nudges_used < self.config.max_final_nudges and step < self.config.max_steps:
                    nudges_used += 1
                    events.append(
                        AgentEvent(
                            "final_nudge",
                            step,
                            f"reply had no tool calls and no parseable plan; nudging ({nudges_used}/{self.config.max_final_nudges}): {preview_text(candidate_text)}",
                        )
                    )
                    if candidate_text.strip():
                        messages.append({"role": "assistant", "content": candidate_text})
                    messages.append({"role": "user", "content": _FINAL_NUDGE_PROMPT})
                    continue
                final_text = candidate_text
                stopped_reason = "final_unparsed"
                events.append(AgentEvent("model_final", step, f"model returned unparsed final text: {preview_text(final_text)}"))
                break

            used_tools = True
            messages.append(_assistant_tool_call_message(message))
            for index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = _parse_tool_arguments(function.get("arguments", "{}"))
                events.append(
                    AgentEvent(
                        "tool_call",
                        step,
                        f"model chose tool {name}",
                        tool=name,
                        data={"arguments": arguments},
                    )
                )
                result_text = self.tool_executor.execute(name, arguments)
                result_payload = _parse_json_object(result_text)
                events.append(
                    AgentEvent(
                        "tool_result",
                        step,
                        _summarize_tool_result(name, result_payload),
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

        if not final_text or not _has_plan_payload(final_text):
            forced_text = self._force_final(messages)
            if _has_plan_payload(forced_text) or not final_text:
                final_text = forced_text
                stopped_reason = "forced_final"
                events.append(
                    AgentEvent(
                        "forced_final",
                        step_count,
                        f"forced final plan (unparsed response or tool budget exhausted): {preview_text(final_text)}",
                    )
                )

        structured_plan = _parse_final_plan(final_text)

        # Completeness gate (#3): if the task has reuse candidates but the plan
        # came back missing reuse/integration sections, re-prompt once and keep
        # whichever attempt is more complete.
        has_catalog = bool(catalog_ids)
        if self.config.completeness_gate:
            gaps = completeness_gaps(structured_plan, has_catalog)
            if gaps:
                events.append(AgentEvent("completeness_gap", step_count, f"plan missing: {', '.join(gaps)}; re-prompting once"))
                retry_text = self._request_completion(messages, gaps, catalog_digest)
                retry_plan = _parse_final_plan(retry_text)
                if completeness_score(retry_plan) > completeness_score(structured_plan):
                    structured_plan = retry_plan
                    final_text = retry_text
                    stopped_reason = "completed"
                    events.append(AgentEvent("completeness_filled", step_count, "re-prompt produced a more complete plan"))

        # Closed-vocabulary grounding (#2): force every selected_ip to a real
        # catalog ip_id (remap near-misses, drop hallucinations).
        grounding: Dict[str, Any] = {}
        if self.config.ground_reuse_decisions and has_catalog:
            structured_plan, grounding = ground_reuse_decisions(structured_plan, catalog_ids)
            if grounding.get("remapped") or grounding.get("dropped"):
                events.append(
                    AgentEvent(
                        "reuse_grounded",
                        -1,
                        f"grounded reuse decisions: {grounding['exact']} exact, "
                        f"{grounding['remapped']} remapped, {grounding['dropped']} dropped",
                        data=grounding,
                    )
                )

        artifact_paths = write_standard_artifacts(structured_plan, self.tool_executor.output_dir)
        events.append(AgentEvent("artifacts_written", -1, f"wrote {len(artifact_paths)} standard artifacts", data=artifact_paths))
        return AgentResult(
            final_text=final_text,
            structured_plan=structured_plan,
            artifact_paths=artifact_paths,
            events=events,
            steps=step_count,
            used_tools=used_tools,
            stopped_reason=stopped_reason,
            grounding=grounding,
        )

    def _catalog_candidates(self) -> List[Any]:
        lister = getattr(self.tool_executor.repository, "list_candidates", None)
        if not callable(lister):
            return []
        try:
            return list(lister())
        except Exception:  # noqa: BLE001 - catalog injection is best-effort.
            return []

    def _request_completion(self, messages: List[Dict[str, Any]], gaps: Sequence[str], catalog_digest: str) -> str:
        # No tools attached: the deployed reasoning parser returns empty content
        # when tools are present, and we only need a JSON plan back here.
        retry_messages = list(messages) + [
            {"role": "user", "content": _completeness_prompt(gaps, catalog_digest)}
        ]
        message = self.llm_client.chat(
            retry_messages,
            temperature=min(self.config.temperature, 0.1),
            max_tokens=self.config.max_tokens,
        )
        return message.get("content") or "{}"

    def _force_final(self, messages: Sequence[Dict[str, Any]]) -> str:
        forced_messages = list(messages) + [
            {
                "role": "user",
                "content": "Tool-call budget is exhausted. Return the best complete IP-reuse design plan as one JSON object now.",
            }
        ]
        # No tools attached: the deployed reasoning parser tends to return empty
        # content (answer trapped in reasoning_content) when tools are present.
        message = self.llm_client.chat(
            forced_messages,
            temperature=min(self.config.temperature, 0.1),
            max_tokens=self.config.max_tokens,
        )
        return message.get("content") or "{}"


def dumps_result(result: AgentResult) -> str:
    return json.dumps(result.to_dict(), default=json_default, indent=2, ensure_ascii=False) + "\n"


def _assistant_tool_call_message(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
    }


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


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


_PLAN_KEYS = {
    "requirements",
    "modules",
    "reuse_decisions",
    "integration_plan",
    "verification_plan",
    "debug_plan",
    "unresolved_assumptions",
}


def _has_plan_payload(text: str) -> bool:
    """True when the text contains a JSON object carrying at least one plan key,
    i.e. it would parse as a real plan rather than the unparsed-text fallback."""
    if not text.strip():
        return False
    for candidate in _plan_json_candidates(text):
        parsed = _parse_json_object(candidate)
        if not parsed or "error" in parsed or "value" in parsed:
            continue
        if _PLAN_KEYS & set(parsed):
            return True
    return False


def _parse_final_plan(text: str) -> Dict[str, Any]:
    fallback: Optional[Dict[str, Any]] = None
    for candidate in _plan_json_candidates(text):
        parsed = _parse_json_object(candidate)
        if not parsed or "error" in parsed or "value" in parsed:
            continue
        if _PLAN_KEYS & set(parsed):
            return _normalize_plan(parsed)
        if fallback is None:
            fallback = parsed
    if fallback is not None:
        return _normalize_plan(fallback)
    return _normalize_plan({"unparsed_final_text": text})


def _plan_json_candidates(text: str) -> List[str]:
    """Candidate JSON payloads: the fence-stripped text, fenced code blocks
    anywhere in the response, then balanced {...} spans (longest first)."""
    candidates = [_strip_fence(text)]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json|JSON)?\s*\n(.*?)```", text, re.DOTALL)
    )
    spans: List[str] = []
    for start in [match.start() for match in re.finditer(r"\{", text)][:16]:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[start : index + 1])
                    break
    candidates.extend(sorted(spans, key=len, reverse=True))
    seen: set = set()
    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan.setdefault("requirements", {})
    plan.setdefault("modules", [])
    plan.setdefault("reuse_decisions", [])
    plan.setdefault("integration_plan", [])
    plan.setdefault("verification_plan", [])
    plan.setdefault("debug_plan", [])
    plan.setdefault("unresolved_assumptions", [])
    return plan


def _summarize_tool_result(name: str, result: Dict[str, Any]) -> str:
    if not result.get("ok", False):
        return f"{name} failed: {preview_text(str(result.get('error', 'unknown error')))}"
    if name == "search_reuse_ip":
        return f"search_reuse_ip returned {len(result.get('candidates') or [])} candidates"
    if name == "inspect_reuse_ip":
        description = result.get("description") or {}
        candidate = description.get("candidate") or {}
        return f"inspect_reuse_ip returned {candidate.get('ip_id', 'unknown')}"
    if name == "evaluate_ip_candidate":
        assessment = result.get("assessment") or {}
        return f"evaluate_ip_candidate scored {assessment.get('ip_id', 'unknown')} as {assessment.get('recommendation', 'review')}"
    if name == "write_artifact":
        return f"write_artifact wrote {result.get('path', '')}"
    if name == "generate_rtl_module":
        n_ports = len(result.get("ports_detected") or [])
        return f"generate_rtl_module wrote {result.get('module_name', 'module')} ({n_ports} ports detected) → {result.get('path', '')}"
    if name == "validate_verilog":
        n_err = len(result.get("errors") or [])
        linter = result.get("linter", "unknown")
        status = "clean" if result.get("ok") else f"{n_err} error(s)"
        return f"validate_verilog [{linter}]: {status}"
    if name == "check_port_compatibility":
        n_issues = len(result.get("issues") or [])
        n_ok = len(result.get("matched") or [])
        return f"check_port_compatibility: {n_ok} matched, {n_issues} issue(s)"
    return f"{name} returned result"
