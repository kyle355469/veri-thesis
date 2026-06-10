import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "build_retrieval_similarity_report.py"
    spec = importlib.util.spec_from_file_location("build_retrieval_similarity_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_retrieval_similarity_report"] = module
    spec.loader.exec_module(module)
    return module


class RetrievalSimilarityReportTests(unittest.TestCase):
    def test_summary_records_use_retrieved_doc_ids_from_generation_log(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "rag"
            problem_dir = run_dir / "Prob001_zero"
            problem_dir.mkdir(parents=True)
            gen_log = problem_dir / "Prob001_zero_sample01-sv-generate.log"
            gen_log.write_text(
                "\n".join(
                    [
                        "problem = Prob001_zero",
                        "sample = 01",
                        'retrieved_doc_ids = ["from-log"]',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "pipeline": "rag",
                        "records": [
                            {
                                "problem": "Prob001_zero",
                                "sample": 1,
                                "generation_log_path": str(gen_log),
                                "retrieved_doc_ids": ["from-summary"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            attempts = module.load_attempts(run_dir)

        self.assertEqual(attempts[0].retrieved_doc_ids, ["from-log"])

    def test_scores_best_retrieved_document_against_reference(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "Prob001_zero_ref.sv"
            ref.write_text(
                "module TopModule(input a, output z); assign z = a; endmodule",
                encoding="utf-8",
            )
            attempt = module.Attempt(
                mode="rag",
                problem="Prob001_zero",
                sample=1,
                generation_log_path=Path(tmp) / "generate.log",
                compile_log_path=None,
                reference_path=ref,
                generated_code_path=None,
                retrieved_doc_ids=["far", "near", "missing"],
                passed=True,
                passfail=".",
                mismatches=0,
                syntax_passed=True,
                lint_passed=True,
                rag_generation_passed=True,
                cache_source="miss",
                repair_attempts=0,
            )
            docs = {
                "far": module.RtlDoc("far", "", "module other(input clk); always @(posedge clk); endmodule"),
                "near": module.RtlDoc(
                    "near",
                    "",
                    "module TopModule(input a, output z); assign z = a; endmodule",
                ),
            }

            scored = module.score_attempts([attempt], docs, "hash", "solution")

        self.assertEqual(scored[0].best_doc_id, "near")
        self.assertEqual(scored[0].missing_doc_count, 1)
        self.assertGreater(scored[0].best_score, 0.9)


if __name__ == "__main__":
    unittest.main()
