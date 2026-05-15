from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .config import RuntimeConfig, ToolCallingConfig
from .json_utils import preview_text
from .llm import extract_code
from .retrieval_context import RetrievalContext
from .tool_calling import RTL_TOOL_SCHEMAS
from .types import Diagnostic, RtlTask, VerificationReport


@dataclass(frozen=True)
class AttemptFeedback:
    kind: str
    diagnostics: List[Diagnostic]
    previous_rtl: str
    previous_model_text_preview: str


PromptBuilder = Callable[[Optional[AttemptFeedback], int], str]
EmergencyPromptBuilder = Callable[[str, int, Optional[AttemptFeedback]], str]
ActionMetadata = Callable[[int, Optional[AttemptFeedback]], Dict[str, Any]]
FailureRecorder = Callable[[str, VerificationReport, int, bool], None]
VerboseLogger = Callable[[str, Dict[str, Any]], None]


@dataclass(frozen=True)
class StageActionConfig:
    generation_action: str
    generation_description: str
    extraction_action: str
    extraction_description: str
    verification_action: str
    verification_description: str
    prompt_event: str
    raw_event: str
    extracted_event: str
    llm_timing_key: str
    verify_timing_key: str
    raw_action: Optional[str] = None
    raw_description: str = ""
    final_response_action: Optional[str] = "llm_final_response"
    tool_stage: Optional[str] = None


@dataclass
class StageRunResult:
    rtl: str
    verification: VerificationReport
    diagnostics: List[Diagnostic]
    repair_attempts: int


FIRST_STAGE_ACTIONS = StageActionConfig(
    generation_action="llm_generation_attempt",
    generation_description="Asked the LLM to produce final RTL as one SiliconMind-style fenced code block.",
    extraction_action="rtl_extracted",
    extraction_description="Extracted RTL from the LLM response, preferring the final fenced code block.",
    verification_action="verification_result",
    verification_description="Pipeline verified the RTL produced by this LLM attempt.",
    prompt_event="generation_prompt",
    raw_event="raw_model_text",
    extracted_event="extracted_rtl",
    llm_timing_key="llm_attempt_{attempt}_s",
    verify_timing_key="verify_attempt_{attempt}_s",
)


SECOND_STAGE_ACTIONS = StageActionConfig(
    generation_action="second_edition_generation_attempt",
    generation_description=(
        "Asked the LLM for second-edition RTL using first-edition code, "
        "Yosys graph, and code-structure VectorDB context."
    ),
    raw_action="second_edition_raw_model_text",
    raw_description="Received raw model text for this second-edition generation attempt.",
    extraction_action="second_edition_rtl_extracted",
    extraction_description="Extracted second-edition RTL from the model response.",
    verification_action="second_edition_verification_result",
    verification_description="Verified the second-edition RTL.",
    prompt_event="second_edition_prompt",
    raw_event="second_edition_raw_model_text",
    extracted_event="second_edition_extracted_rtl",
    llm_timing_key="second_edition_llm_attempt_{attempt}_s",
    verify_timing_key="second_edition_verify_attempt_{attempt}_s",
    final_response_action=None,
    tool_stage="second_edition",
)


class RtlGenerationStage:
    """Run one prompt/generate/extract/verify repair loop."""

    def __init__(
        self,
        llm_client: Any,
        verifier: Any,
        retrieval_context: RetrievalContext,
        runtime_config: RuntimeConfig,
        tool_config: ToolCallingConfig,
        verbose: VerboseLogger,
        actions: StageActionConfig,
    ):
        self.llm = llm_client
        self.verifier = verifier
        self.retrieval_context = retrieval_context
        self.runtime_config = runtime_config
        self.tool_config = tool_config
        self.verbose = verbose
        self.actions = actions

    def run(
        self,
        task: RtlTask,
        max_attempts: int,
        build_prompt: PromptBuilder,
        llm_actions: List[Dict[str, Any]],
        timings: Dict[str, float],
        action_metadata: Optional[ActionMetadata] = None,
        build_emergency_prompt: Optional[EmergencyPromptBuilder] = None,
        on_failed_attempt: Optional[FailureRecorder] = None,
    ) -> StageRunResult:
        feedback: Optional[AttemptFeedback] = None
        diagnostics: List[Diagnostic] = []
        rtl = ""
        verification = VerificationReport(False, False, [])
        repair_attempts = 0

        for attempt in range(max_attempts + 1):
            attempt_feedback = feedback if attempt else None
            prompt = build_prompt(attempt_feedback, attempt)
            self._record_generation_attempt(llm_actions, attempt, attempt_feedback, action_metadata)
            self.verbose(self.actions.prompt_event, {"attempt": attempt, "prompt": prompt})

            t0 = time.perf_counter()
            model_text = self._complete(prompt, task, llm_actions, attempt)
            self.verbose(self.actions.raw_event, {"attempt": attempt, "text": model_text})
            self._record_raw_model_text(llm_actions, attempt, model_text)
            rtl = extract_code(model_text)
            self.verbose(self.actions.extracted_event, {"attempt": attempt, "rtl": rtl})
            self._record_extracted_rtl(llm_actions, attempt, rtl)

            if not rtl and build_emergency_prompt is not None and attempt >= max_attempts:
                emergency_prompt = build_emergency_prompt(model_text, attempt, attempt_feedback)
                self._record_emergency_retry(llm_actions, attempt, model_text)
                self.verbose(
                    self.actions.prompt_event,
                    {"attempt": attempt, "emergency": True, "prompt": emergency_prompt},
                )
                emergency_text = self._complete_without_tools(
                    emergency_prompt,
                    llm_actions,
                    attempt,
                )
                self.verbose(
                    self.actions.raw_event,
                    {"attempt": attempt, "emergency": True, "text": emergency_text},
                )
                self._record_raw_model_text(llm_actions, attempt, emergency_text)
                emergency_rtl = extract_code(emergency_text)
                self.verbose(
                    self.actions.extracted_event,
                    {"attempt": attempt, "emergency": True, "rtl": emergency_rtl},
                )
                self._record_extracted_rtl(llm_actions, attempt, emergency_rtl)
                model_text = emergency_text
                if emergency_rtl:
                    rtl = emergency_rtl

            timings[self.actions.llm_timing_key.format(attempt=attempt)] = time.perf_counter() - t0

            t0 = time.perf_counter()
            verification = (
                extraction_failure_report(model_text)
                if not rtl
                else self.verifier.verify(rtl, top_module=task.top_module)
            )
            timings[self.actions.verify_timing_key.format(attempt=attempt)] = time.perf_counter() - t0

            repair_attempts = attempt
            diagnostics = verification.diagnostics
            self._record_verification(llm_actions, attempt, verification)
            if verification.passed:
                break
            feedback = AttemptFeedback(
                kind="extraction" if not rtl else "verification",
                diagnostics=diagnostics,
                previous_rtl=rtl,
                previous_model_text_preview=preview_text(model_text, limit=1200),
            )
            if on_failed_attempt:
                on_failed_attempt(rtl, verification, attempt, attempt >= max_attempts)

        return StageRunResult(
            rtl=rtl,
            verification=verification,
            diagnostics=diagnostics,
            repair_attempts=repair_attempts,
        )

    def _complete(
        self,
        prompt: str,
        task: RtlTask,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
    ) -> str:
        if not self.tool_config.enabled or not hasattr(self.llm, "complete_with_tools"):
            model_text = self.llm.complete(
                prompt,
                temperature=self.runtime_config.generation_temperature,
                max_tokens=self.runtime_config.max_tokens,
            )
            if self.actions.final_response_action:
                llm_actions.append(
                    {
                        "action": self.actions.final_response_action,
                        "attempt": attempt,
                        "used_tools": False,
                        "content": model_text,
                        "content_preview": preview_text(model_text),
                    }
                )
            return model_text

        executor = self.retrieval_context.tool_executor(
            verifier=self.verifier,
            default_top_module=task.top_module,
        )
        return self.llm.complete_with_tools(
            prompt,
            tools=RTL_TOOL_SCHEMAS,
            tool_executor=executor.execute,
            temperature=self.runtime_config.generation_temperature,
            max_tokens=self.runtime_config.max_tokens,
            tool_choice=self.tool_config.choice,
            max_tool_rounds=self.tool_config.max_rounds,
            action_recorder=lambda action: llm_actions.append(
                self._tool_action_record(attempt, action)
            ),
        )

    def _complete_without_tools(
        self,
        prompt: str,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
    ) -> str:
        model_text = self.llm.complete(
            prompt,
            temperature=min(self.runtime_config.generation_temperature, 0.1),
            max_tokens=self.runtime_config.max_tokens,
        )
        if self.actions.final_response_action:
            llm_actions.append(
                {
                    "action": "emergency_llm_final_response",
                    "attempt": attempt,
                    "used_tools": False,
                    "content": model_text,
                    "content_preview": preview_text(model_text),
                }
            )
        return model_text

    def _record_generation_attempt(
        self,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
        feedback: Optional[AttemptFeedback],
        action_metadata: Optional[ActionMetadata],
    ) -> None:
        metadata = {
            "with_repair_diagnostics": bool(feedback and feedback.diagnostics),
            "retry_kind": feedback.kind if feedback else None,
        }
        if action_metadata:
            metadata.update(action_metadata(attempt, feedback))
        llm_actions.append(
            {
                "action": self.actions.generation_action,
                "attempt": attempt,
                "description": self.actions.generation_description,
                **metadata,
            }
        )

    def _record_raw_model_text(
        self,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
        model_text: str,
    ) -> None:
        if not self.actions.raw_action:
            return
        llm_actions.append(
            {
                "action": self.actions.raw_action,
                "attempt": attempt,
                "description": self.actions.raw_description,
                "content": model_text,
                "content_preview": preview_text(model_text),
            }
        )

    def _record_extracted_rtl(
        self,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
        rtl: str,
    ) -> None:
        llm_actions.append(
            {
                "action": self.actions.extraction_action,
                "attempt": attempt,
                "description": self.actions.extraction_description,
                "rtl": rtl,
                "rtl_preview": preview_text(rtl),
            }
        )

    def _record_emergency_retry(
        self,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
        model_text: str,
    ) -> None:
        llm_actions.append(
            {
                "action": "emergency_extraction_retry",
                "attempt": attempt,
                "description": (
                    "The first response in this attempt contained no extractable RTL, "
                    "so the pipeline asked for a compact code-only answer before verification."
                ),
                "previous_content_preview": preview_text(model_text),
            }
        )

    def _record_verification(
        self,
        llm_actions: List[Dict[str, Any]],
        attempt: int,
        verification: VerificationReport,
    ) -> None:
        llm_actions.append(
            {
                "action": self.actions.verification_action,
                "attempt": attempt,
                "description": self.actions.verification_description,
                "passed": verification.passed,
                "syntax_passed": verification.syntax_passed,
                "lint_passed": verification.lint_passed,
                "failed_tools": [
                    diagnostic.tool for diagnostic in verification.diagnostics if not diagnostic.passed
                ],
            }
        )

    def _tool_action_record(self, attempt: int, action: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {"attempt": attempt}
        if self.actions.tool_stage:
            record["stage"] = self.actions.tool_stage
        record.update(action)
        return record


def extraction_failure_report(model_text: str) -> VerificationReport:
    return VerificationReport(
        syntax_passed=False,
        lint_passed=False,
        diagnostics=[
            Diagnostic(
                tool="rtl_extraction",
                passed=False,
                stderr=(
                    "No RTL code was extracted from the model output. "
                    "Return exactly one fenced HDL code block containing complete RTL code."
                ),
                stdout=preview_text(model_text, limit=1200),
            )
        ],
    )
