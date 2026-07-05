import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "run_realbench_direct_model.py"
    spec = importlib.util.spec_from_file_location("run_realbench_direct_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_realbench_direct_model"] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_fake_realbench(root: Path) -> None:
    write(
        root / "benchmark_info.py",
        "benchmark_info = {'aes': {'aes_sbox': []}}\n"
        "system_info = {}\n",
    )
    write(
        root / "problems" / "aes" / "problems.jsonl",
        json.dumps({"task": "aes_sbox", "problem": "Build an aes_sbox module with input a and output b."}) + "\n",
    )
    write(root / "aes" / "aes_sbox" / "verification" / "aes_sbox_top.sv", "module old; endmodule\n")


class FakeClient:
    def __init__(self):
        self.messages = []

    def reset_request_log(self):
        pass

    def current_requests(self):
        return []

    def chat(self, messages, temperature, max_tokens):
        self.messages.append(messages)
        return {
            "content": "```verilog\nmodule aes_sbox(input [7:0] a, output [7:0] b); assign b = a; endmodule\n```",
            "_finish_reason": "stop",
        }


def parse_args(module, root: Path, output: Path, *extra: str):
    return module.build_parser().parse_args(
        [
            "--realbench-root",
            str(root),
            "--output-dir",
            str(output),
            "--task-level",
            "module",
            "--no-prepare-problems",
            *extra,
        ]
    )


def test_direct_prompt_uses_problem_and_direct_output_rules(tmp_path):
    module = load_script()
    root = tmp_path / "real_bench"
    make_fake_realbench(root)
    args = parse_args(module, root, tmp_path / "out")
    task = module.discover_tasks(args)[0]

    prompt = module.direct_prompt(task, args)

    assert "Build an aes_sbox module" in prompt
    assert "Required public top module name: aes_sbox" in prompt
    assert "Do not include a testbench" in prompt
    assert "Retrieved Context" not in prompt


def test_run_one_direct_writes_generated_code_and_evaluates(tmp_path):
    module = load_script()
    root = tmp_path / "real_bench"
    output = tmp_path / "out"
    make_fake_realbench(root)
    fake_make = tmp_path / "fake_make.py"
    write(
        fake_make,
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "text = Path('aes_sbox_top.sv').read_text(encoding='utf-8')\n"
        "if 'module aes_sbox' not in text:\n"
        "    raise SystemExit(1)\n"
        "print('Hint: Output has no mismatches')\n",
    )
    fake_make.chmod(0o755)
    args = parse_args(
        module,
        root,
        output,
        "--realbench-verifier",
        "native",
        "--make-bin",
        str(fake_make),
    )
    task = module.discover_tasks(args)[0]
    catalog = module.build_task_catalog(task, output)
    item = module.WorkItem(task=task, sample=1)

    record = module.run_one_direct(item, args, output, catalog, FakeClient())

    assert record["pipeline"] == "direct_model"
    assert record["generated"] is True
    assert record["passed"] is True
    assert Path(record["generated_code_path"]).read_text(encoding="utf-8").startswith("module aes_sbox")
    assert Path(record["raw_response_path"]).exists()
    assert Path(record["prompt_path"]).exists()
