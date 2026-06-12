from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_rtl.json_utils import preview_text
from rag_rtl.llm import extract_code

from ..rtl import validate_single_module_rtl as _validate_single_module_rtl
from ..serialization import parse_json_object as _parse_json_object
from ..types import LlmTrace


class LlmStagesMixin:
    def _complete_json(self, stage: str, prompt: str, traces: List[LlmTrace]) -> Dict[str, Any]:
        for attempt in range(self.config.max_generation_retries + 1):
            label = stage if attempt == 0 else f"{stage}:retry{attempt}"
            response = self._complete_text(label, prompt, traces)
            parsed = _parse_json_object(response)
            traces[-1].parsed = parsed is not None
            if parsed is not None:
                return parsed
            if attempt < self.config.max_generation_retries:
                self._stage(f"llm:{stage}", "retry", attempt=attempt + 1, reason="json_parse_failed")
        return {}

    def _generate_module_rtl(
        self, stage: str, prompt: str, module_name: str, traces: List[LlmTrace]
    ) -> str:
        """Complete an LLM call and extract RTL, retrying on format failures."""
        for attempt in range(self.config.max_generation_retries + 1):
            label = stage if attempt == 0 else f"{stage}:retry{attempt}"
            final_text = self._complete_text(label, prompt, traces)
            rtl = extract_code(final_text).strip()
            try:
                _validate_single_module_rtl(rtl, module_name)
                return rtl
            except RuntimeError as exc:
                if attempt < self.config.max_generation_retries:
                    self._stage(f"llm:{stage}", "retry", attempt=attempt + 1, reason=str(exc))
                    continue
                raise

    def _complete_text(self, stage: str, prompt: str, traces: List[LlmTrace]) -> str:
        self._stage(f"llm:{stage}", "running", prompt_chars=len(prompt))
        last_exc: Optional[Exception] = None
        response = ""
        for attempt in range(self.config.max_generation_retries + 1):
            try:
                response = self.llm.complete(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.config.max_generation_retries:
                    self._stage(f"llm:{stage}", "retry", attempt=attempt + 1, reason=str(exc))
        if last_exc is not None:
            raise last_exc
        traces.append(
            LlmTrace(
                stage=stage,
                prompt_preview=preview_text(prompt),
                response_preview=preview_text(response),
                parsed=False,
                prompt_chars=len(prompt),
                response_chars=len(response),
            )
        )
        self._stage(f"llm:{stage}", "complete", response_chars=len(response))
        return response

    def _stage(self, stage: str, status: str, **payload: Any) -> None:
        if self.stage_callback is None:
            return
        event = {"stage": stage, "status": status, **payload}
        self.stage_callback(event)

