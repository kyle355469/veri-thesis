import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "analyze_question_table.py"
    spec = importlib.util.spec_from_file_location("analyze_question_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["analyze_question_table"] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeQuestionTableTests(unittest.TestCase):
    def test_accumulates_error_reason_ratios_from_markdown_table(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "question_table.md"
            table.write_text(
                "\n".join(
                    [
                        "| benchmark | mode | category | problem | samples | passfail_counts |",
                        "| --- | --- | --- | --- | --- | --- |",
                        "| rtllm | model | Arithmetic | adder | 5 | .:3 G:1 R:1 |",
                        "| rtllm | model | Control | fsm | 5 | C:2 R:3 |",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            rows = module.read_markdown_table(table)
            grouped = module.accumulate_counts(rows)
            analysis = module.build_analysis_rows(grouped)

        by_reason = {row["reason"]: row for row in analysis}
        self.assertEqual(set(by_reason), {"R", "C", "G"})
        self.assertEqual(by_reason["R"]["count"], 4)
        self.assertEqual(by_reason["R"]["ratio_of_errors"], 4 / 7)
        self.assertEqual(by_reason["R"]["ratio_of_all"], 4 / 10)
        self.assertEqual(by_reason["C"]["total_errors"], 7)
        self.assertEqual(by_reason["C"]["total_samples"], 10)

    def test_can_group_by_mode_and_include_success(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "question_table.md"
            table.write_text(
                "\n".join(
                    [
                        "| benchmark | mode | category | problem | samples | passfail_counts |",
                        "| --- | --- | --- | --- | --- | --- |",
                        "| rtllm | model | Arithmetic | adder | 5 | .:4 G:1 |",
                        "| rtllm | rag | Arithmetic | adder | 5 | .:2 R:3 |",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            rows = module.read_markdown_table(table)
            grouped = module.accumulate_counts(rows, group_by="mode")
            analysis = module.build_analysis_rows(grouped, include_success=True, group_by="mode")

        by_group_reason = {(row["mode"], row["reason"]): row for row in analysis}
        self.assertEqual(by_group_reason[("model", ".")]["ratio_of_all"], 4 / 5)
        self.assertEqual(by_group_reason[("model", "G")]["ratio_of_errors"], 1.0)
        self.assertEqual(by_group_reason[("rag", "R")]["ratio_of_errors"], 1.0)

    def test_by_mode_is_shortcut_for_group_by_mode(self):
        module = load_script()
        cli = module.build_parser().parse_args(["question_table.md", "--by-mode"])

        self.assertEqual(module.resolve_group_by(cli), "mode")


if __name__ == "__main__":
    unittest.main()
