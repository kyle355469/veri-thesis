from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class VllmClient:
    base_url: str = "http://localhost:8000/v1"
    model: str = "siliconmind-server"
    api_key: str = "EMPTY"
    timeout_s: int = 1200

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
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
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
        choice = body["choices"][0]
        message = choice["message"]
        # Reasoning-parser deployments can leave "content" empty while the actual
        # answer sits in "reasoning_content"; without this fallback the agent sees
        # an empty final response and needs a wasted forced-final retry. The
        # promoted text is usually working notes, not an answer, so it is marked
        # for callers that must not accept it as final output.
        if not (message.get("content") or "").strip() and not message.get("tool_calls"):
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if str(reasoning).strip():
                message = dict(message)
                message["content"] = str(reasoning)
                message["_content_from_reasoning"] = True
        message["_finish_reason"] = choice.get("finish_reason")
        return message

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed: HTTP {exc.code} {exc.reason}: {_compact_error_body(body)}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc


class MockLlmClient:
    """Deterministic tool-calling client for tests and local demos."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self.calls.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if not tool_messages:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_search_fifo",
                        "type": "function",
                        "function": {
                            "name": "search_reuse_ip",
                            "arguments": json.dumps(
                                {
                                    "query": "streaming fifo axi valid ready buffer",
                                    "top_k": 3,
                                    "filters": {},
                                }
                            ),
                        },
                    }
                ],
            }
        if len(tool_messages) == 1:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_inspect_fifo",
                        "type": "function",
                        "function": {
                            "name": "inspect_reuse_ip",
                            "arguments": json.dumps({"ip_id": "sync_fifo"}),
                        },
                    },
                    {
                        "id": "call_eval_fifo",
                        "type": "function",
                        "function": {
                            "name": "evaluate_ip_candidate",
                            "arguments": json.dumps(
                                {
                                    "ip_id": "sync_fifo",
                                    "module_requirements": {
                                        "module_name": "Buffer / FIFO",
                                        "role": "elastic streaming data buffering",
                                        "interfaces": ["valid-ready"],
                                        "requirements": ["parameterized depth", "synthesis-ready"],
                                    },
                                }
                            ),
                        },
                    },
                ],
            }
        return {"content": _mock_final_text()}


def _mock_final_text() -> str:
    return """{
  "requirements": {
    "functionality": ["Streaming data path with reusable FIFO and AXI-lite control"],
    "performance": ["Throughput target to be refined from system constraints"],
    "io_interfaces": ["AXI-lite control", "valid-ready streaming data"],
    "protocols": ["AXI-lite", "valid-ready"],
    "ppa_constraints": ["Prefer existing verified IP to reduce area and verification schedule risk"],
    "clock_reset": ["Single clock assumed for v1 unless CDC is requested"],
    "assumptions": ["Exact sample rate, data width, and timing target are unresolved"]
  },
  "modules": [
    {"name": "Input Interface", "role": "Accept streaming input and enforce protocol", "interfaces": ["valid-ready"], "reuse_preference": "adapter or existing stream endpoint", "verification_needs": ["protocol assertions"]},
    {"name": "Buffer / FIFO", "role": "Elastic buffering between interface and core", "interfaces": ["valid-ready"], "reuse_preference": "reuse sync_fifo", "verification_needs": ["overflow/underflow tests"]},
    {"name": "Processing Core", "role": "Perform requested datapath computation", "interfaces": ["valid-ready"], "reuse_preference": "search datapath IP, otherwise new RTL", "verification_needs": ["golden model comparison"]},
    {"name": "Memory Controller", "role": "Coordinate coefficient/state memory if needed", "interfaces": ["AXI-lite", "SRAM"], "reuse_preference": "reuse register bank or SRAM adapter", "verification_needs": ["register and memory access tests"]},
    {"name": "Output Interface", "role": "Present processed stream downstream", "interfaces": ["valid-ready"], "reuse_preference": "adapter or existing stream endpoint", "verification_needs": ["backpressure tests"]}
  ],
  "reuse_decisions": [
    {"module_name": "Buffer / FIFO", "selected_ip": "sync_fifo", "rejected_ips": [], "required_adapters": ["Map data valid-ready to FIFO push/pop if native ports differ"], "parameterization": {"DATA_WIDTH": "match stream width", "DEPTH": "size from burst/backpressure target"}, "risk_notes": ["Confirm almost-full/almost-empty semantics"], "new_rtl_required": false},
    {"module_name": "Processing Core", "selected_ip": null, "rejected_ips": [], "required_adapters": [], "parameterization": {}, "risk_notes": ["No computation-specific IP selected in mock run"], "new_rtl_required": true}
  ],
  "integration_plan": ["Instantiate selected IPs behind thin wrappers", "Normalize resets and handshakes", "Document parameter choices before RTL integration"],
  "verification_plan": ["Run IP unit testbenches", "Add protocol assertions for AXI-lite and valid-ready", "Run lint, synthesis elaboration, and timing smoke checks"],
  "debug_plan": ["Start with module-level simulation", "Trace handshakes across adapters", "Promote passing configurations into reusable catalog metadata"],
  "unresolved_assumptions": ["Concrete PPA target", "Exact data width", "Clock-domain requirements"]
}"""


def _compact_error_body(body: str, max_chars: int = 800) -> str:
    text = " ".join(body.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
