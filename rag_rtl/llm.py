from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .json_utils import preview_text
from .siliconmind_utils import parse_code as parse_siliconmind_code, wrap_code

HDL_SOURCE_RE = re.compile(r"(?m)^\s*(module|interface|package|primitive|program)\b", re.IGNORECASE)
KEYWORD_PROMPT_MARKER = "You are a Verilog specification keyword extraction assistant."
KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "build",
    "code",
    "create",
    "for",
    "from",
    "hdl",
    "in",
    "input",
    "make",
    "module",
    "of",
    "output",
    "rtl",
    "that",
    "the",
    "to",
    "verilog",
    "with",
}

@dataclass
class VllmClient:
    base_url: str = "http://localhost:8000/v1"
    model: str = "siliconmind-server"
    timeout_s: int = 1200
    api_key: str = "EMPTY"

    @classmethod
    def from_env(cls) -> "VllmClient":
        return cls(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("VLLM_MODEL", "siliconmind-server"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        )

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        temperature: float = 0.4,
        max_tokens: int = 65536,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        body = self._post_chat_completion(payload)
        return body["choices"][0]["message"]

    def complete(self, prompt: str, temperature: float = 0.4, max_tokens: int = 2048) -> str:
        message = self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return message.get("content") or ""

    def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], str],
        temperature: float = 0.4,
        max_tokens: int = 2048,
        tool_choice: Any = "auto",
        max_tool_rounds: int = 4,
        action_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        for round_index in range(max_tool_rounds):
            message = self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
            )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                if action_recorder:
                    content = message.get("content") or ""
                    action_recorder(
                        {
                            "action": "llm_final_response",
                            "round": round_index,
                            "used_tools": any(item.get("role") == "tool" for item in messages),
                            "content": content,
                            "content_preview": preview_text(content),
                        }
                    )
                return message.get("content") or ""
            messages.append(_assistant_tool_call_message(message))
            for index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function") or {}
                name = function.get("name", "")
                arguments = _parse_tool_arguments(function.get("arguments", "{}"))
                if action_recorder:
                    action_recorder(
                        {
                            "action": "llm_tool_call",
                            "round": round_index,
                            "tool": name,
                            "arguments": arguments,
                        }
                    )
                result = tool_executor(name, arguments)
                if action_recorder:
                    action_recorder(
                        {
                            "action": "tool_result",
                            "round": round_index,
                            "tool": name,
                            "result_preview": preview_text(result),
                        }
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or f"tool_call_{index}",
                        "name": name,
                        "content": result,
                    }
                )
        message = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="none",
            parallel_tool_calls=False,
        )
        if action_recorder:
            content = message.get("content") or ""
            action_recorder(
                {
                    "action": "llm_final_response",
                    "round": max_tool_rounds,
                    "used_tools": True,
                    "content": content,
                    "content_preview": preview_text(content),
                }
            )
        return message.get("content") or ""

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM request failed: HTTP {exc.code} {exc.reason}: {_compact_error_body(body)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        return body


class StubLlmClient:
    """Deterministic client for tests and dry runs."""

    def __init__(self, rtl: Optional[str] = None):
        self.rtl = rtl or "module stub();\nendmodule"
        self.prompts: List[str] = []
        self.keyword_prompts: List[str] = []
        self.tool_prompts: List[str] = []

    def complete(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        if KEYWORD_PROMPT_MARKER in prompt:
            self.keyword_prompts.append(prompt)
            text = prompt.rsplit("Specification:", 1)[-1]
            keywords: List[str] = []
            seen = set()
            for token in KEYWORD_TOKEN_RE.findall(text.lower()):
                if len(token) < 2 or token in KEYWORD_STOPWORDS or token in seen:
                    continue
                seen.add(token)
                keywords.append(token)
            return json.dumps(
                {
                    "direction": "design",
                    "module_name": [],
                    "type": "unknown",
                    "gate_usage": [],
                    "signals": {"input": [], "output": []},
                    "keywords": keywords[:12],
                }
            )
        self.prompts.append(prompt)
        return wrap_code(self.rtl)

    def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], str],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        tool_choice: Any = "auto",
        max_tool_rounds: int = 4,
        action_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        self.tool_prompts.append(prompt)
        if action_recorder:
            content = wrap_code(self.rtl)
            action_recorder(
                {
                    "action": "llm_final_response",
                    "round": 0,
                    "used_tools": False,
                    "content": content,
                    "content_preview": preview_text(content),
                }
            )
        return self.complete(prompt, temperature=temperature, max_tokens=max_tokens)


def extract_code(model_text: str) -> str:
    siliconmind_code = parse_siliconmind_code(model_text)
    if siliconmind_code:
        return siliconmind_code
    source = model_text.strip()
    return source if _looks_like_hdl_source(source) else ""


def _looks_like_hdl_source(source: str) -> bool:
    return bool(HDL_SOURCE_RE.search(source))


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


def _assistant_tool_call_message(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
    }


def _compact_error_body(body: str) -> str:
    if not body:
        return "<empty response body>"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:2000]
    return json.dumps(payload, ensure_ascii=False)[:2000]
