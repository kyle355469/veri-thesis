import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "run_agentic_ip_reuse_realbench.py"
    spec = importlib.util.spec_from_file_location("run_agentic_ip_reuse_realbench", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_agentic_ip_reuse_realbench"] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_fake_realbench(root: Path) -> None:
    write(
        root / "benchmark_info.py",
        "benchmark_info = {\n"
        "  'aes': {'aes_key_expand_128': ['aes_sbox']},\n"
        "  'e203_hbirdv2': {'e203_clk_ctrl': ['sirv_gnrl_dffr']},\n"
        "}\n"
        "system_info = {\n"
        "  'aes_cipher_top': ['aes_cipher_top', 'aes_key_expand_128', 'aes_sbox'],\n"
        "}\n",
    )
    write(
        root / "problems" / "aes" / "problems.jsonl",
        json.dumps({"task": "aes_key_expand_128", "problem": "Build AES key expansion"}) + "\n",
    )
    write(
        root / "problems" / "e203_hbirdv2" / "problems.jsonl",
        json.dumps({"task": "e203_clk_ctrl", "problem": "Build E203 clock control"}) + "\n",
    )
    write(
        root / "problems" / "system" / "problems.jsonl",
        json.dumps({"task": "aes_cipher_top", "problem": "Build complete AES cipher"}) + "\n",
    )

    write(root / "aes" / "aes_key_expand_128" / "aes_key_expand_128.v", "module aes_key_expand_128; endmodule\n")
    write(root / "aes" / "aes_key_expand_128" / "verification" / "aes_key_expand_128_top.sv", "module aes_key_expand_128; endmodule\n")
    write(root / "aes" / "aes_key_expand_128" / "verification" / "aes_sbox.v", "module aes_sbox; endmodule\n")
    write(root / "aes" / "aes_sbox" / "aes_sbox.v", "module aes_sbox; endmodule\n")

    write(root / "e203_hbirdv2" / "e203_clk_ctrl" / "e203_clk_ctrl.v", "module e203_clk_ctrl; endmodule\n")
    write(root / "e203_hbirdv2" / "e203_clk_ctrl" / "verification" / "e203_clk_ctrl_top.sv", "module e203_clk_ctrl; endmodule\n")
    write(
        root / "e203_hbirdv2" / "e203_clk_ctrl" / "verification" / "sirv_gnrl_dffs.v",
        "module sirv_gnrl_dffr; endmodule\n",
    )
    write(root / "e203_hbirdv2" / "e203_defines" / "e203_defines.v", "`define E203_FAKE 1\n")
    write(root / "e203_hbirdv2" / "config" / "config.v", "`define E203_CONFIG_FAKE 1\n")

    write(root / "system" / "aes_cipher_top" / "aes_cipher_top_top.sv", "module aes_cipher_top; endmodule\n")
    write(root / "system" / "aes_cipher_top" / "aes_key_expand_128.v", "module aes_key_expand_128; endmodule\n")
    write(root / "system" / "aes_cipher_top" / "aes_sbox.v", "module aes_sbox; endmodule\n")


def parse_args(module, root: Path, output: Path, *extra: str):
    return module.build_parser().parse_args(
        [
            "--realbench-root",
            str(root),
            "--output-dir",
            str(output),
            "--no-prepare-problems",
            *extra,
        ]
    )


def test_discovers_module_and_system_tasks_from_realbench_info(tmp_path):
    module = load_script()
    root = tmp_path / "real_bench"
    make_fake_realbench(root)
    args = parse_args(module, root, tmp_path / "out", "--task-level", "both")

    tasks = module.discover_tasks(args)

    assert [task.task for task in tasks] == ["aes_key_expand_128", "e203_clk_ctrl", "aes_cipher_top"]
    assert tasks[0].level == "module"
    assert tasks[0].dependencies == ["aes_sbox"]
    assert tasks[-1].level == "system"
    assert tasks[-1].dependencies == ["aes_key_expand_128", "aes_sbox"]


def test_dependency_index_excludes_target_and_records_access(tmp_path):
    module = load_script()
    root = tmp_path / "real_bench"
    output = tmp_path / "out"
    make_fake_realbench(root)
    args = parse_args(module, root, output, "--task-level", "module", "--include", "aes_key_expand")
    task = module.discover_tasks(args)[0]

    bundle = module.build_task_index(task, args, output)

    assert bundle.missing_dependencies == []
    assert bundle.dependency_paths["aes_sbox"].endswith("verification/aes_sbox.v")
    assert not any(document.metadata["source_path"].endswith("aes_key_expand_128.v") for document in bundle.documents)
    assert bundle.access == {"aes_sbox": True}


def test_dependency_resolution_scans_verification_files_for_helper_module(tmp_path):
    module = load_script()
    root = tmp_path / "real_bench"
    make_fake_realbench(root)
    args = parse_args(module, root, tmp_path / "out", "--task-level", "module", "--include", "e203_clk_ctrl")
    task = module.discover_tasks(args)[0]

    source = module.resolve_dependency_source(task, "sirv_gnrl_dffr")

    assert source is not None
    assert source.path.name == "sirv_gnrl_dffs.v"


def test_dry_run_writes_records_and_realbench_solution_files(tmp_path):
    module = load_script()
    root = tmp_path / "real_bench"
    output = tmp_path / "out"
    make_fake_realbench(root)
    args = parse_args(module, root, output, "--task-level", "both", "--dry-run")

    summary = module.run_realbench(args)

    records_path = output / "records.jsonl"
    assert summary["dry_run"] is True
    assert records_path.exists()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert records[0]["ip_db_doc_count"] >= 1
    assert (output / "samples" / "agentic_ip_reuse" / "aes.jsonl").exists()
    assert (output / "samples" / "agentic_ip_reuse" / "system.jsonl").exists()


def test_rtl_mosaic_bridge_delegates_to_agentic_engine(tmp_path):
    module = load_script()
    root = tmp_path / "real_bench"
    rtl_mosaic = tmp_path / "rtl_mosaic"
    make_fake_realbench(root)
    write(
        rtl_mosaic / "eval" / "run_chipbench_eval.py",
        "import json, sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "engine = sys.argv[sys.argv.index('--engine') + 1]\n"
        "(out / 'summary.json').write_text(json.dumps({'benchmark': 'rtl-mosaic', 'engine': engine}))\n",
    )
    args = parse_args(
        module,
        root,
        tmp_path / "out",
        "--benchmark",
        "rtl-mosaic",
        "--rtl-mosaic-root",
        str(rtl_mosaic),
        "--dry-run",
    )

    payload = module.run_rtl_mosaic(args)

    assert payload["summary"]["engine"] == "agentic"
    assert "--engine" in payload["command"]
    assert "agentic" in payload["command"]
    assert (tmp_path / "out" / "rtl_mosaic_summary.json").exists()
