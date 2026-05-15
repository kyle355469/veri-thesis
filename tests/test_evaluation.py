import json

import pytest

from rag_rtl.evaluation import iter_tasks


def test_iter_tasks_jsonl_prompt_records(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"prompt": "Design an inverter.", "constraints": "use assign"}) + "\n",
        encoding="utf-8",
    )

    loaded = list(iter_tasks(tasks))

    assert len(loaded) == 1
    assert loaded[0].prompt == "Design an inverter."
    assert loaded[0].constraints == ["use assign"]


def test_iter_tasks_json_array_accepts_spec_alias(tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            [
                {
                    "spec": "Design a BCD adder.",
                    "golden": "module top_module; endmodule",
                    "target_hdl": "verilog",
                }
            ]
        ),
        encoding="utf-8",
    )

    loaded = list(iter_tasks(tasks))

    assert len(loaded) == 1
    assert loaded[0].prompt == "Design a BCD adder."
    assert loaded[0].target_hdl == "verilog"


def test_iter_tasks_reports_missing_prompt_or_spec(tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"golden": "module m; endmodule"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing one of"):
        list(iter_tasks(tasks))
