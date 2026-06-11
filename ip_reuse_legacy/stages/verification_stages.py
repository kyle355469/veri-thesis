from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rag_rtl.llm import extract_code
from rag_rtl.types import Diagnostic, RtlDocument, VerificationReport
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

from ..constants import MODULE_DECL_RE
from ..manifest import write_text as _write_text
from ..prompts import build_testbench_generation_prompt


class VerificationStagesMixin:
    def _make_leaf_verifier(
        self,
        module_name: str,
        module_spec_text: str,
        rtl: str,
        state: Dict[str, Any],
    ) -> RtlVerifier:
        """Return a verifier for a leaf module, wiring in a testbench when available."""
        tb_path = self._find_testbench(module_name)
        if tb_path is None:
            # Ask LLM to generate a testbench and write it to the workspace.
            tb_text_raw = self._complete_text(
                f"testbench_generation:{module_name}",
                build_testbench_generation_prompt(module_spec_text, rtl, self.config.target_hdl),
                state["llm_traces"],
            )
            tb_rtl = extract_code(tb_text_raw).strip()
            if tb_rtl:
                tb_dir = state["workspace"] / "testbenches"
                tb_path = tb_dir / f"{module_name}_tb.v"
                _write_text(tb_path, tb_rtl + "\n")
                state["artifacts"][f"testbench:{module_name}"] = str(tb_path)
        if tb_path is not None and tb_path.exists():
            test_command = (
                f"iverilog -o {{top}}_sim {{testbench}} {{rtl}} && ./{{top}}_sim"
            )
            return RtlVerifier(
                yosys_bin=self.verifier.yosys_bin,
                verilator_bin=self.verifier.verilator_bin,
                timeout_s=self.verifier.timeout_s,
                testbench_path=tb_path,
                test_command=test_command,
            )
        return self.verifier

    def _verify_with(self, verifier: RtlVerifier, rtl: str, top_module: str) -> VerificationReport:
        """Run verification using the given verifier instance."""
        if verifier is self.verifier:
            return self._verify_module(rtl, top_module)
        if not rtl:
            return self._verify_or_empty(rtl, top_module)
        return verifier.verify(rtl, top_module=top_module)

    def _register_live_document(
        self, module_name: str, module_payload: Dict[str, Any], rtl: str
    ) -> None:
        """Add a successfully generated leaf RTL to the in-session live store."""
        doc = RtlDocument(
            doc_id=f"live:{module_name}",
            problem=(
                f"Previously generated leaf module {module_name}. "
                f"Purpose: {module_payload.get('purpose', 'unknown')}. "
                f"Category: {module_payload.get('category', 'unknown')}."
            ),
            solution=rtl,
            tags=["live", "generated", module_name, str(module_payload.get("category", ""))],
        )
        embedder = self.retrieval_context.retriever.embedder
        vector = embedder.encode([doc.retrieval_text])[0]
        if self._live_store is None:
            self._live_store = VectorStore([doc], vector.reshape(1, -1))
        else:
            self._live_store.add(doc, vector)

    def _find_testbench(self, module_name: str) -> Optional[Path]:
        """Return a testbench file for module_name from testbench_dir, or None."""
        tb_dir = self.config.testbench_dir
        if tb_dir is None or not tb_dir.exists():
            return None
        for suffix in ("_tb.v", "_tb.sv", "_top.sv", "_tb.vhd"):
            candidate = tb_dir / f"{module_name}{suffix}"
            if candidate.exists():
                return candidate
        for path in sorted([*tb_dir.glob("*.v"), *tb_dir.glob("*.sv")]):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if module_name in MODULE_DECL_RE.findall(text):
                return path
        return None

