from __future__ import annotations

from typing import Optional

from rag_rtl.types import Diagnostic, VerificationReport


class VerificationCoreMixin:
    def _verify_or_empty(self, rtl: str, top_module: Optional[str]) -> VerificationReport:
        if rtl:
            return self.verifier.verify(rtl, top_module=top_module)
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

    def _verify_module(self, rtl: str, top_module: str) -> VerificationReport:
        if not rtl:
            return self._verify_or_empty(rtl, top_module)
        if not hasattr(self.verifier, "run_yosys") or not hasattr(self.verifier, "run_verilator"):
            return self.verifier.verify(rtl, top_module=top_module)
        diagnostics = [
            self.verifier.run_yosys(rtl, top_module=top_module),
            self.verifier.run_verilator(rtl),
        ]
        return VerificationReport(
            syntax_passed=diagnostics[0].passed,
            lint_passed=diagnostics[1].passed,
            diagnostics=diagnostics,
        )
