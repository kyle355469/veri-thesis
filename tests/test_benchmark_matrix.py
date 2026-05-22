import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "run_benchmark_matrix.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_benchmark_matrix"] = module
    spec.loader.exec_module(module)
    return module


class FailingToolClient:
    def __init__(self):
        self.chat_calls = 0
        self.reset_calls = 0

    def chat(self, *args, **kwargs):
        self.chat_calls += 1
        raise RuntimeError(
            'vLLM request failed: HTTP 400 Bad Request: {"error": {"message": '
            '"\\"auto\\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"}}'
        )

    def reset_usage(self):
        self.reset_calls += 1


class FakeBenchmarkModule:
    def __init__(self):
        self.run_one_calls = 0

    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pipeline", default="rag")
        parser.add_argument("--output-dir", default="")
        parser.add_argument("--rtllm-root", default="")
        parser.add_argument("--include", action="append", default=[])
        parser.add_argument("--limit", type=int)
        parser.add_argument("--samples", type=int, default=1)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--evaluate-only", action="store_true")
        parser.add_argument("--resume", action="store_true")
        return parser

    def discover_problems(self, rtllm_root, include, limit=None):
        return ["problem-a", "problem-b"]

    def iter_work_items(self, problems, samples):
        return [(problem, sample) for problem in problems for sample in range(1, samples + 1)]

    def run_one(self, item, pipeline, args, output_dir):
        self.run_one_calls += 1
        return {"problem": str(item), "sample": 1, "passed": False, "passfail": "G"}

    def write_csv_summary(self, path, records):
        path.write_text("problem,sample,passed,passfail\n", encoding="utf-8")


class BenchmarkMatrixTests(unittest.TestCase):
    def test_tool_preflight_error_includes_restart_hint(self):
        module = load_script()
        args = argparse.Namespace(enable_tool_calling=True, tool_choice="auto")

        with self.assertRaises(module.ToolPreflightError) as caught:
            module.preflight_tool_calling(FailingToolClient(), args, "verilog-eval", "tool")

        message = str(caught.exception)
        self.assertIn("ENABLE_TOOL_CALLING=1 TOOL_CALL_PARSER=qwen3_xml bash vllm_deploy.sh", message)
        self.assertIn("--enable-auto-tool-choice", message)
        self.assertIn("verilog-eval/tool", message)

    def test_tool_preflight_skips_mode_before_submitting_samples(self):
        module = load_script()
        fake_module = FakeBenchmarkModule()
        cli = module.build_parser().parse_args(
            [
                "--benchmark",
                "rtllm",
                "--mode",
                "tool",
                "--samples",
                "2",
                "--output-dir",
                "unused",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(module.TrackingVllmClient, "from_env", return_value=FailingToolClient()):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary, records = module.run_benchmark_mode(
                        fake_module,
                        "rtllm",
                        "tool",
                        cli,
                        Path(tmp),
                    )

        self.assertEqual(records, [])
        self.assertTrue(summary["preflight_failed"])
        self.assertEqual(summary["num_records"], 4)
        self.assertEqual(summary["passfail_counts"], {"preflight_error": 4})
        self.assertEqual(fake_module.run_one_calls, 0)

    def test_parallel_mode_jobs_keep_final_order(self):
        module = load_script()
        cli = module.build_parser().parse_args(
            [
                "--benchmark",
                "rtllm",
                "--mode",
                "model",
                "--mode",
                "rag",
                "--mode",
                "tool",
                "--mode-concurrency",
                "2",
            ]
        )
        modules = {"rtllm": object()}
        lock = threading.Lock()
        active = 0
        max_active = 0
        delays = {"model": 0.08, "rag": 0.02, "tool": 0.01}

        def fake_run_benchmark_mode(script_module, benchmark, mode, cli_args, output_dir):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(delays[mode])
                return (
                    {"benchmark": benchmark, "mode": mode},
                    [{"benchmark": benchmark, "mode": mode, "problem": mode, "sample": 1}],
                )
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(module, "run_benchmark_mode", side_effect=fake_run_benchmark_mode):
                with contextlib.redirect_stdout(io.StringIO()):
                    summaries, records = module.run_matrix_jobs(
                        modules,
                        ["rtllm"],
                        ["model", "rag", "tool"],
                        cli,
                        Path(tmp),
                    )

        self.assertGreater(max_active, 1)
        self.assertEqual([summary["mode"] for summary in summaries], ["model", "rag", "tool"])
        self.assertEqual([record["mode"] for record in records], ["model", "rag", "tool"])

    def test_token_and_repair_columns_are_exported(self):
        module = load_script()
        records = [
            {
                "benchmark": "rtllm",
                "mode": "rag",
                "category": "cat",
                "problem": "p",
                "sample": 1,
                "passed": True,
                "passfail": ".",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "tokens_used": 15,
                "repair_attempts": 2,
            },
            {
                "benchmark": "rtllm",
                "mode": "rag",
                "category": "cat",
                "problem": "p",
                "sample": 2,
                "passed": False,
                "passfail": "G",
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
                "tokens_used": 12,
                "repair_attempts": 0,
            },
        ]

        rows = module.build_question_rows(records)
        self.assertEqual(rows[0]["total_tokens_used"], 27)
        self.assertEqual(rows[0]["total_repair_attempts"], 2)
        self.assertEqual(rows[0]["avg_repair_attempts"], 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "records.csv"
            module.write_records_csv(csv_path, records)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")

        self.assertIn("tokens_used", header)
        self.assertIn("repair_attempts", header)


if __name__ == "__main__":
    unittest.main()
