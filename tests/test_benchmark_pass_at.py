import importlib.util
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_verilog_eval_pass_at_uses_standard_estimator_by_problem():
    module = load_script("run_verilog_eval")
    records = [
        {"problem": "a", "sample": 1, "passed": False},
        {"problem": "a", "sample": 2, "passed": True},
        {"problem": "a", "sample": 3, "passed": False},
        {"problem": "a", "sample": 4, "passed": False},
        {"problem": "a", "sample": 5, "passed": False},
        {"problem": "b", "sample": 1, "passed": True},
        {"problem": "b", "sample": 2, "passed": False},
        {"problem": "b", "sample": 3, "passed": True},
        {"problem": "b", "sample": 4, "passed": False},
        {"problem": "b", "sample": 5, "passed": False},
    ]

    rates, denominators = module.compute_pass_at(records, (1, 3, 5))

    assert denominators == {1: 2, 3: 2, 5: 2}
    assert_rates_close(rates, {1: 0.3, 3: 0.75, 5: 1.0})


def test_rtllm_pass_at_groups_same_problem_name_by_category():
    module = load_script("run_rtllm_eval")
    records = [
        {"category": "comb", "problem": "adder", "sample": 1, "passed": True},
        {"category": "comb", "problem": "adder", "sample": 2, "passed": False},
        {"category": "comb", "problem": "adder", "sample": 3, "passed": False},
        {"category": "comb", "problem": "adder", "sample": 4, "passed": False},
        {"category": "comb", "problem": "adder", "sample": 5, "passed": False},
        {"category": "seq", "problem": "adder", "sample": 1, "passed": False},
        {"category": "seq", "problem": "adder", "sample": 2, "passed": False},
        {"category": "seq", "problem": "adder", "sample": 3, "passed": False},
        {"category": "seq", "problem": "adder", "sample": 4, "passed": False},
        {"category": "seq", "problem": "adder", "sample": 5, "passed": False},
    ]

    rates, denominators = module.compute_pass_at(records, (1, 3, 5))

    assert denominators == {1: 2, 3: 2, 5: 2}
    assert_rates_close(rates, {1: 0.1, 3: 0.3, 5: 0.5})


def assert_rates_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] is not None
        assert math.isclose(actual[key], value)
