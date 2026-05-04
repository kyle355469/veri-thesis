from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from .types import Diagnostic, VerificationReport


class RtlVerifier:
    def __init__(self, yosys_bin: str = "yosys", verilator_bin: str = "verilator", timeout_s: int = 30):
        self.yosys_bin = yosys_bin
        self.verilator_bin = verilator_bin
        self.timeout_s = timeout_s

    def verify(self, rtl: str, top_module: str | None = None) -> VerificationReport:
        diagnostics: List[Diagnostic] = []
        with tempfile.TemporaryDirectory(prefix="rag_rtl_") as tempdir:
            rtl_path = Path(tempdir) / "candidate.v"
            rtl_path.write_text(rtl, encoding="utf-8")
            diagnostics.append(self._run_yosys(rtl_path, top_module))
            diagnostics.append(self._run_verilator(rtl_path))

        syntax_passed = diagnostics[0].passed
        lint_passed = diagnostics[1].passed
        return VerificationReport(syntax_passed=syntax_passed, lint_passed=lint_passed, diagnostics=diagnostics)

    def _run_yosys(self, rtl_path: Path, top_module: str | None) -> Diagnostic:
        if shutil.which(self.yosys_bin) is None:
            return Diagnostic(tool="yosys", passed=False, missing=True, stderr="yosys not found on PATH")
        script = f"read_verilog {rtl_path}; "
        if top_module:
            script += f"hierarchy -top {top_module}; "
        script += "proc; check"
        return self._run([self.yosys_bin, "-q", "-p", script], "yosys")

    def _run_verilator(self, rtl_path: Path) -> Diagnostic:
        if shutil.which(self.verilator_bin) is None:
            return Diagnostic(tool="verilator", passed=False, missing=True, stderr="verilator not found on PATH")
        return self._run([self.verilator_bin, "--lint-only", str(rtl_path)], "verilator")

    def _run(self, command: List[str], tool: str) -> Diagnostic:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return Diagnostic(tool=tool, passed=False, stderr=str(exc), returncode=None)
        return Diagnostic(
            tool=tool,
            passed=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
