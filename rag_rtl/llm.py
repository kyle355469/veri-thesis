from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

CODE_RE = re.compile(r"```(?:verilog|systemverilog|sv)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass
class VllmClient:
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-rtl-model"
    timeout_s: int = 120
    api_key: str = "EMPTY"

    @classmethod
    def from_env(cls) -> "VllmClient":
        return cls(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("VLLM_MODEL", "local-rtl-model"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        )

    def complete(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
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
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        return body["choices"][0]["message"]["content"]


class StubLlmClient:
    """Deterministic client for tests and dry runs."""

    def __init__(self, rtl: Optional[str] = None):
        self.rtl = rtl or "module stub();\nendmodule"
        self.prompts: List[str] = []

    def complete(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        self.prompts.append(prompt)
        return f"```verilog\n{self.rtl}\n```"


def extract_code(model_text: str) -> str:
    match = CODE_RE.search(model_text)
    if match:
        return match.group(1).strip()
    return model_text.strip()
