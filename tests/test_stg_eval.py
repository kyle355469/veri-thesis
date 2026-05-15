import json

from rag_rtl.stg_eval import infer_first_module_name, iter_dataset_records, run_stg_dataset_evaluation
from rag_rtl.types import PipelineResponse, VerificationReport


def test_infer_first_module_name():
    assert infer_first_module_name("module top(input a); endmodule") == "top"
    assert infer_first_module_name("// module ignored\nmodule real_top; endmodule") == "real_top"


def test_iter_dataset_records_jsonl(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"spec": "a", "golden_code": "module m; endmodule"}\n\n', encoding="utf-8")

    records = list(iter_dataset_records(dataset))

    assert records == [{"spec": "a", "golden_code": "module m; endmodule"}]


def test_stg_dataset_records_missing_golden_without_calling_stg(tmp_path):
    class FakePipeline:
        def run(self, *args, **kwargs):
            raise AssertionError("pipeline should not run without golden code")

    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "out.json"
    dataset.write_text(json.dumps({"spec": "make a module"}) + "\n", encoding="utf-8")

    summary = run_stg_dataset_evaluation(
        dataset,
        output,
        pipeline=FakePipeline(),
        stg_bin="definitely-missing-stg",
    )

    assert summary["num_records"] == 1
    assert summary["passed"] == 0
    assert summary["records"][0]["error"] == "missing spec or golden code"


def test_stg_dataset_uses_rag_pipeline_before_stg(tmp_path):
    class FakePipeline:
        def __init__(self):
            self.calls = []

        def run(self, task, retrieve_k=8, context_k=4):
            self.calls.append((task, retrieve_k, context_k))
            return PipelineResponse(
                rtl="module top(input a, output y); assign y = a; endmodule",
                verification=VerificationReport(True, True, []),
                retrieved_doc_ids=["doc1"],
                cache_source="miss",
                repair_attempts=0,
            )

    dataset = tmp_path / "dataset.json"
    output = tmp_path / "out.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "spec": "make a buffer",
                    "golden_code": "module top_ref(input a, output y); assign y = a; endmodule",
                }
            ]
        ),
        encoding="utf-8",
    )
    pipeline = FakePipeline()

    summary = run_stg_dataset_evaluation(
        dataset,
        output,
        pipeline=pipeline,
        stg_bin="definitely-missing-stg",
        retrieve_k=3,
        context_k=2,
    )

    assert pipeline.calls[0][0].prompt == "make a buffer"
    assert pipeline.calls[0][1:] == (3, 2)
    assert summary["records"][0]["rag_generation_passed"] is True
    assert summary["records"][0]["stderr_tail"] == "stg binary not found: definitely-missing-stg"


def test_stg_dataset_saves_all_generated_result_code(tmp_path):
    class FakePipeline:
        def __init__(self):
            self.responses = [
                PipelineResponse(
                    rtl="module ok(input a, output y); assign y = a; endmodule",
                    verification=VerificationReport(True, True, []),
                    retrieved_doc_ids=[],
                    cache_source="miss",
                    repair_attempts=0,
                ),
                PipelineResponse(
                    rtl="this is not valid verilog",
                    verification=VerificationReport(False, False, []),
                    retrieved_doc_ids=[],
                    cache_source="miss",
                    repair_attempts=1,
                ),
            ]

        def run(self, *args, **kwargs):
            return self.responses.pop(0)

    dataset = tmp_path / "dataset.json"
    output = tmp_path / "out.json"
    code_dir = tmp_path / "codes"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "good-case",
                    "spec": "make a buffer",
                    "golden_code": "module ok(input a, output y); assign y = a; endmodule",
                },
                {
                    "id": "bad case",
                    "spec": "make broken code",
                    "golden_code": "module bad; endmodule",
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = run_stg_dataset_evaluation(
        dataset,
        output,
        pipeline=FakePipeline(),
        stg_bin="definitely-missing-stg",
        save_result_code_dir=code_dir,
    )

    assert summary["generated"] == 2
    assert summary["result_code_dir"] == str(code_dir)
    assert (code_dir / "result_00000_good-case.v").read_text(encoding="utf-8").startswith("module ok")
    assert (code_dir / "result_00001_bad_case.v").read_text(encoding="utf-8") == "this is not valid verilog"
    assert summary["records"][0]["generated_code_path"] == str(code_dir / "result_00000_good-case.v")
    assert summary["records"][1]["generated_code_path"] == str(code_dir / "result_00001_bad_case.v")
    assert json.loads(output.read_text(encoding="utf-8"))["result_code_dir"] == str(code_dir)
