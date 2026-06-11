import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agentic_ip_reuse.cli import build_parser, cmd_run


class CliTests(unittest.TestCase):
    def test_parser_accepts_run_args(self):
        args = build_parser().parse_args(
            [
                "run",
                "--prompt",
                "Build a FIFO subsystem",
                "--mock-llm",
                "--max-steps",
                "4",
                "--known-interface",
                "AXI-lite",
                "--ppa-target",
                "low area",
            ]
        )
        self.assertEqual(args.command, "run")
        self.assertEqual(args.prompt, "Build a FIFO subsystem")
        self.assertTrue(args.mock_llm)
        self.assertEqual(args.max_steps, 4)
        self.assertEqual(args.known_interface, ["AXI-lite"])

    def test_cli_mock_run_writes_json_report_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            out = Path(tmp) / "out"
            args = build_parser().parse_args(
                [
                    "run",
                    "--prompt",
                    "Build a simple streaming FIR accelerator with reusable FIFO and AXI-lite control",
                    "--mock-llm",
                    "--output-dir",
                    str(out),
                    "--json-report",
                    str(report),
                ]
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = cmd_run(args)

            self.assertTrue(report.exists())
            self.assertTrue((out / "requirements.md").exists())
            self.assertTrue((out / "module_decomposition.md").exists())
            self.assertTrue((out / "ip_reuse_matrix.md").exists())
            self.assertTrue((out / "integration_plan.md").exists())
            self.assertTrue((out / "verification_plan.md").exists())
            self.assertTrue((out / "result.json").exists())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["stopped_reason"], "final")
            self.assertIn("Buffer / FIFO", (out / "module_decomposition.md").read_text(encoding="utf-8"))
            self.assertIn("requirements", result.artifact_paths)


if __name__ == "__main__":
    unittest.main()
