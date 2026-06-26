from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rag_rtl.llm import extract_code
from rag_rtl.types import Diagnostic, VerificationReport

from ..prompts import build_functional_repair_prompt
from ..spec_slice import slice_spec_for_diagnostics
from ..types import IpReusePlan, LlmTrace

# Reference testbenches print "Output X has N mismatches. First at time T"; the
# total mismatch count is the score we minimise when no attempt fully passes.
_MISMATCH_RE = re.compile(r"(\d+)\s+mismatch", re.IGNORECASE)
_NO_IMPROVEMENT = 1 << 30


def _mismatch_count(function_info: str) -> int:
    counts = [int(value) for value in _MISMATCH_RE.findall(function_info or "")]
    if counts:
        return sum(counts)
    return 0 if not function_info else _NO_IMPROVEMENT


class FunctionalRepairMixin:
    """Testbench-driven (functional) repair turns appended after the syntax/lint
    loop, gated on the design already compiling. Reuses the injected
    ``functional_verifier`` (duck-typed: ``verify_functional(rtl, top_module)``
    returning an object with ``function_passed``/``function_info``/``syntax_ok``)."""

    def _attach_functional_diagnostic(
        self, verification: VerificationReport, report: Any
    ) -> VerificationReport:
        diagnostic = Diagnostic(
            tool="external_testbench",
            passed=bool(report.function_passed),
            stdout=(report.function_info or "")[:8000],
            stderr="" if report.syntax_ok else "functional repair candidate failed to compile",
            returncode=0 if report.function_passed else 1,
        )
        return VerificationReport(
            syntax_passed=verification.syntax_passed,
            lint_passed=verification.lint_passed,
            diagnostics=[*verification.diagnostics, diagnostic],
        )

    def _run_functional_repair(
        self,
        rtl: str,
        plan: IpReusePlan,
        target: str,
        top_module: Optional[str],
        llm_traces: List[LlmTrace],
        original_spec: Optional[str],
        reuse_modules: Optional[Dict[str, str]],
        environment_notes: Optional[List[str]],
        verification: VerificationReport,
    ) -> Tuple[str, VerificationReport, int, List[Dict[str, Any]], str]:
        events: List[Dict[str, Any]] = []
        report = self.functional_verifier.verify_functional(rtl, top_module)
        self._stage(
            "functional_verification",
            "complete",
            function_passed=report.function_passed,
            syntax_ok=report.syntax_ok,
        )
        events.append(
            {
                "event": "verify",
                "attempt": 0,
                "function_passed": report.function_passed,
                "syntax_ok": report.syntax_ok,
                "mismatches": _mismatch_count(report.function_info),
                "function_info": (report.function_info or "")[:500],
            }
        )

        # Track the best COMPILING candidate (lowest mismatch count); the working
        # design fed back into the prompt only ever advances to a compiling one, so
        # a syntax regression never poisons the next turn's localisation signal.
        best_rtl, best_report = rtl, report
        best_mismatches = _mismatch_count(report.function_info)
        working_rtl, working_info = rtl, report.function_info
        attempts = 0
        while not report.function_passed and attempts < self.config.max_functional_repair_attempts:
            attempts += 1
            self._stage("functional_repair", "running", attempt=attempts)
            # Focus on the spec section(s) describing the mismatching output(s);
            # an empty slice falls back to the full spec inside the prompt builder.
            behavioral_slice = None
            if getattr(self.config, "enable_repair_spec_slice", False) and original_spec:
                behavioral_slice = (
                    slice_spec_for_diagnostics(
                        original_spec,
                        function_info=working_info,
                        max_chars=self.config.repair_spec_slice_max_chars,
                    )
                    or None
                )
            final_text = self._complete_text(
                f"functional_repair_{attempts}",
                build_functional_repair_prompt(
                    plan,
                    working_rtl,
                    working_info,
                    target,
                    top_module,
                    original_spec=original_spec,
                    behavioral_slice=behavioral_slice,
                    reuse_modules=reuse_modules,
                    environment_notes=environment_notes,
                ),
                llm_traces,
            )
            candidate = extract_code(final_text)
            self._stage("functional_repair", "generated", attempt=attempts, rtl_chars=len(candidate))
            if not candidate:
                events.append({"event": "repair", "attempt": attempts, "status": "empty_code"})
                continue
            report = self.functional_verifier.verify_functional(candidate, top_module)
            # Mismatch trajectory: record this candidate's count against the best so
            # far, so analysis can see whether a non-passing turn still got closer.
            candidate_mismatches = _mismatch_count(report.function_info)
            accepted_as_best = report.function_passed or (
                report.syntax_ok and candidate_mismatches < best_mismatches
            )
            events.append(
                {
                    "event": "repair",
                    "attempt": attempts,
                    "function_passed": report.function_passed,
                    "syntax_ok": report.syntax_ok,
                    "mismatches": candidate_mismatches,
                    "prev_best_mismatches": best_mismatches,
                    "accepted_as_best": bool(accepted_as_best),
                    "function_info": (report.function_info or "")[:500],
                }
            )
            self._stage(
                "functional_repair",
                "verified",
                attempt=attempts,
                function_passed=report.function_passed,
                syntax_ok=report.syntax_ok,
            )
            if report.function_passed:
                best_rtl, best_report, best_mismatches = candidate, report, 0
                break
            if report.syntax_ok:
                # Compiling candidate: advance the working design and keep the best.
                working_rtl, working_info = candidate, report.function_info
                if candidate_mismatches < best_mismatches:
                    best_rtl, best_report, best_mismatches = candidate, report, candidate_mismatches
            # A non-compiling candidate is discarded: working_rtl/working_info stay
            # at the last compiling design so the next prompt keeps real mismatch info.

        updated = self._attach_functional_diagnostic(verification, best_report)
        return best_rtl, updated, attempts, events, best_report.function_info
