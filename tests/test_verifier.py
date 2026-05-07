import tempfile
import unittest
from pathlib import Path

from rag_rtl.verifier import RtlVerifier


class VerifierTests(unittest.TestCase):
    def test_external_testbench_success_is_required_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            testbench = Path(tmp) / "tb.v"
            testbench.write_text("module tb; endmodule\n", encoding="utf-8")
            verifier = RtlVerifier(
                yosys_bin="/bin/true",
                verilator_bin="/bin/true",
                testbench_path=testbench,
                test_command="/bin/true {rtl} {testbench} {top}",
            )

            report = verifier.verify("module dut; endmodule", top_module="dut")

            self.assertTrue(report.passed)
            self.assertTrue(any(item.tool == "external_testbench" and item.passed for item in report.diagnostics))

    def test_external_testbench_failure_fails_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            testbench = Path(tmp) / "tb.v"
            testbench.write_text("module tb; endmodule\n", encoding="utf-8")
            verifier = RtlVerifier(
                yosys_bin="/bin/true",
                verilator_bin="/bin/true",
                testbench_path=testbench,
                test_command="/bin/false {rtl} {testbench} {top}",
            )

            report = verifier.verify("module dut; endmodule", top_module="dut")

            self.assertFalse(report.passed)
            self.assertTrue(any(item.tool == "external_testbench" and not item.passed for item in report.diagnostics))


if __name__ == "__main__":
    unittest.main()
