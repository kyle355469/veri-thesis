import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from rag_rtl.types import Diagnostic, VerificationReport
from rtl_agent.agent import AgentResult


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "run_agentic_benchmark_matrix.py"
    spec = importlib.util.spec_from_file_location("run_agentic_benchmark_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_agentic_benchmark_matrix"] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FakeProblem:
    problem_id: str = "adder"
    category: str = "comb"
    top_module: str = "adder"
    description_path: Path = Path("design_description.txt")
    testbench_path: Path = Path("testbench.v")
    reference_path: Path = Path("verified_adder.v")
    prompt: str = "Make an adder"


@dataclass(frozen=True)
class FakeItem:
    problem: FakeProblem
    sample: int


class FakeClient:
    def __init__(self):
        self.reset_calls = 0

    def reset_usage(self):
        self.reset_calls += 1

    def reset_request_log(self):
        pass

    def current_requests(self):
        return []

    def current_usage(self):
        return {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "llm_requests": 1,
            "api_usage_requests": 1,
            "estimated_usage_requests": 0,
        }


class FakeAgent:
    def run(self, task):
        return AgentResult(
            rtl="module wrong_name(input a, output y); assign y = a; endmodule",
            final_text="```verilog\nmodule wrong_name(input a, output y); assign y = a; endmodule\n```",
            verification=VerificationReport(
                syntax_passed=True,
                lint_passed=True,
                diagnostics=[Diagnostic(tool="stub", passed=True)],
            ),
            steps=2,
            used_tools=True,
            stopped_reason="final",
        )


class FakeFactory:
    def __init__(self):
        self.workspace_roots = []
        self.top_modules = []

    def build(self, *, workspace_root, top_module):
        self.workspace_roots.append(workspace_root)
        self.top_modules.append(top_module)
        return FakeAgent()


class FakeModule:
    @staticmethod
    def build_task(problem, args):
        from rag_rtl.types import RtlTask

        return RtlTask(prompt=problem.prompt, top_module=problem.top_module)

    @staticmethod
    def generated_code_path(output_dir, item):
        return output_dir / item.problem.category / item.problem.problem_id / "adder_sample01.v"

    @staticmethod
    def generation_log_path(output_dir, item):
        return output_dir / item.problem.category / item.problem.problem_id / "adder_sample01-generate.log"

    @staticmethod
    def simulation_log_path(output_dir, item):
        return output_dir / item.problem.category / item.problem.problem_id / "adder_sample01-iverilog.log"

    @staticmethod
    def normalize_generated_code(code, top_module):
        return code.replace("wrong_name", top_module, 1)

    @staticmethod
    def write_generation_log(path, item, response, error, reused):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"error = {error}\nreused = {reused}\n", encoding="utf-8")

    @staticmethod
    def write_simulation_log(path, result):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"passfail = {result.passfail}\n", encoding="utf-8")

    @staticmethod
    def evaluate_with_iverilog(item, candidate_path, log_path, args):
        text = candidate_path.read_text(encoding="utf-8")
        assert "module adder" in text
        result = argparse.Namespace(
            passed=True,
            passfail=".",
            compile_returncode=0,
            simulation_returncode=0,
            failures=0,
            compile_command=["iverilog"],
            run_command=["./adder_sample01"],
            stdout="Your Design Passed",
            stderr="",
            compile_s=0.1,
            simulation_s=0.2,
            error=None,
        )
        FakeModule.write_simulation_log(log_path, result)
        return result

    @staticmethod
    def response_metadata(response):
        return {
            "rag_generation_passed": bool(response and response.verification.passed),
            "syntax_passed": bool(response and response.verification.syntax_passed),
            "lint_passed": bool(response and response.verification.lint_passed),
            "repair_attempts": None,
            "cache_source": response.cache_source if response else None,
            "retrieved_doc_ids": [],
            "timings": {},
        }

    @staticmethod
    def verification_diagnostics(report):
        if not report:
            return []
        return [{"tool": item.tool, "passed": item.passed} for item in report.diagnostics]


def test_normalize_benchmark_aliases():
    module = load_script()

    assert module.normalize_benchmarks([]) == ["verilog-eval-v2-ntu", "rtllm-v2"]
    assert module.normalize_benchmarks(["verilog-eval", "rtllm", "rtllm-v2"]) == [
        "verilog-eval-v2-ntu",
        "rtllm-v2",
    ]


def test_run_one_agentic_writes_agent_record_and_evaluates_generated_code(tmp_path):
    module = load_script()
    item = FakeItem(problem=FakeProblem(), sample=1)
    args = argparse.Namespace(resume=False, evaluate_only=False)
    factory = FakeFactory()
    client = FakeClient()

    record = module.run_one_agentic(
        FakeModule,
        "rtllm-v2",
        item,
        args,
        tmp_path,
        factory,
        client,
    )

    assert record["passed"] is True
    assert record["passfail"] == "."
    assert record["agent_steps"] == 2
    assert record["agent_used_tools"] is True
    assert record["total_tokens"] == 15
    assert Path(record["generated_code_path"]).read_text(encoding="utf-8").startswith("module adder")
    assert Path(record["agent_report_path"]).exists()
    assert factory.top_modules == ["adder"]
