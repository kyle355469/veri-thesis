import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "run_cvdp_eval.py"
    spec = importlib.util.spec_from_file_location("run_cvdp_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_cvdp_eval"] = module
    spec.loader.exec_module(module)
    return module


def test_cvdp_discovery_filters_requested_cids_and_infers_target_file(tmp_path):
    module = load_script()
    dataset = tmp_path / "cvdp.jsonl"
    rows = [
        {
            "id": "cvdp_copilot_keep_0001",
            "categories": ["cid003", "easy"],
            "input": {"prompt": "Build module keep.", "context": {}},
            "output": {"response": "", "context": {"rtl/keep.sv": ""}},
            "harness": {"files": {"src/.env": "TOPLEVEL = keep\nVERILOG_SOURCES = /code/rtl/keep.sv\n"}},
        },
        {
            "id": "cvdp_copilot_skip_0001",
            "categories": ["cid012", "easy"],
            "input": {"prompt": "Build module skip.", "context": {}},
            "output": {"response": "", "context": {"rtl/skip.sv": ""}},
            "harness": {"files": {"src/.env": "TOPLEVEL = skip\nVERILOG_SOURCES = /code/rtl/skip.sv\n"}},
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        cvdp_root=str(tmp_path),
        dataset=[str(dataset)],
        cid=["03"],
        include=[],
        limit=None,
    )

    problems = module.discover_problems(args)

    assert [problem.problem_id for problem in problems] == ["cvdp_copilot_keep_0001"]
    assert problems[0].primary_cid == "cid003"
    assert problems[0].top_module == "keep"
    assert problems[0].expected_files == ("rtl/keep.sv",)


def test_cvdp_pass_at_groups_by_cid_and_problem():
    module = load_script()
    records = [
        {"cid": "cid002", "problem": "same_name", "sample": 1, "passed": True},
        {"cid": "cid002", "problem": "same_name", "sample": 2, "passed": False},
        {"cid": "cid003", "problem": "same_name", "sample": 1, "passed": False},
        {"cid": "cid003", "problem": "same_name", "sample": 2, "passed": False},
    ]

    rates, denominators = module.compute_pass_at(records, (1, 2))

    assert denominators == {1: 2, 2: 2}
    assert rates[1] == 0.25
    assert rates[2] == 0.5
