import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts" / "agentic_ip_reuse_web.py"
    spec = importlib.util.spec_from_file_location("agentic_ip_reuse_web_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AgenticIpReuseWebTests(unittest.TestCase):
    def test_first_open_port_is_strict_when_one_attempt_is_requested(self):
        module = load_script()
        probe = mock.MagicMock()
        probe.__enter__.return_value = probe
        probe.bind.side_effect = OSError("busy")

        with mock.patch.object(module.socket, "socket", return_value=probe):
            with self.assertRaisesRegex(RuntimeError, "Port 8780 is already in use"):
                module._first_open_port("127.0.0.1", 8780, attempts=1)

    def test_first_open_port_can_fall_back_when_requested(self):
        module = load_script()
        probe = mock.MagicMock()
        probe.__enter__.return_value = probe
        probe.bind.side_effect = [OSError("busy"), None]

        with mock.patch.object(module.socket, "socket", return_value=probe):
            port = module._first_open_port("127.0.0.1", 8780, attempts=2)

        self.assertEqual(port, 8781)

    def test_job_artifacts_are_opaque_and_job_scoped(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registered = root / "index.txt"
            unregistered = root / "secret.txt"
            registered.write_text("index\n", encoding="utf-8")
            unregistered.write_text("secret\n", encoding="utf-8")
            job_id = "job-one"
            module.JOBS[job_id] = {"job_id": job_id, "artifacts": [], "updated_at": 0}
            module.JOB_ARTIFACTS[job_id] = {}

            module._register_job_artifacts(job_id, {"index": str(registered)})

            payload = module._job_payload(job_id)
            artifact = payload["artifacts"][0]
            self.assertNotIn("path", artifact)
            self.assertNotIn(str(registered), artifact["url"])
            self.assertEqual(module._job_artifact_path(job_id, artifact["artifact_id"]), registered.resolve())
            self.assertIsNone(module._job_artifact_path("different-job", artifact["artifact_id"]))
            self.assertIsNone(module._job_artifact_path(job_id, str(unregistered)))

    def test_job_payload_summarizes_live_submodule_progress(self):
        module = load_script()
        job_id = "progress-job"
        module.JOBS[job_id] = {
            "job_id": job_id,
            "state": "running",
            "task": "cpu",
            "stages": [
                {
                    "stage": "decomposition",
                    "status": "complete",
                    "detail": {"module_count": 2, "modules": ["alu", "decoder"]},
                },
                {"stage": "module_generation", "status": "running", "detail": {"module": "alu"}},
                {"stage": "module_verification", "status": "complete", "detail": {"module": "alu"}},
                {"stage": "module_generation", "status": "running", "detail": {"module": "decoder"}},
            ],
            "result": None,
        }

        payload = module._job_payload(job_id)

        self.assertEqual(payload["progress"]["total"], 2)
        self.assertEqual(payload["progress"]["completed"], 1)
        self.assertEqual(payload["progress"]["active"], 1)
        self.assertEqual(payload["progress"]["percent"], 50)
        self.assertEqual(
            {item["name"]: item["status"] for item in payload["progress"]["modules"]},
            {"alu": "complete", "decoder": "generating"},
        )

    def test_job_payload_enriches_final_generation_tree(self):
        module = load_script()
        job_id = "tree-job"
        module.JOBS[job_id] = {
            "job_id": job_id,
            "state": "complete",
            "task": "cpu",
            "stages": [],
            "result": {
                "record": {
                    "top_module": "cpu",
                    "module_generation": [
                        {"module": "adder", "kind": "leaf", "repair_attempts": 1, "passed": True},
                        {"module": "alu", "kind": "composite", "repair_attempts": 0, "passed": True},
                    ],
                    "decomposition_tree": {
                        "module": "cpu",
                        "kind": "root",
                        "children": [
                            {
                                "module": "alu",
                                "kind": "composite",
                                "children": [{"module": "adder", "kind": "leaf", "children": []}],
                            }
                        ],
                    },
                }
            },
        }

        payload = module._job_payload(job_id)
        tree = payload["generation_tree"]

        self.assertEqual(tree["module"], "cpu")
        self.assertEqual(tree["status"], "complete")
        self.assertEqual(tree["children"][0]["module"], "alu")
        self.assertEqual(tree["children"][0]["order"], 2)
        self.assertEqual(tree["children"][0]["children"][0]["repair_attempts"], 1)
        self.assertEqual(tree["children"][0]["children"][0]["status"], "complete")

    def test_job_payload_builds_fallback_tree_for_small_run(self):
        module = load_script()
        job_id = "small-tree-job"
        module.JOBS[job_id] = {
            "job_id": job_id,
            "state": "complete",
            "task": "small cpu",
            "stages": [
                {
                    "stage": "agent_start",
                    "status": "running",
                    "detail": {"top_module": "cpu"},
                },
                {
                    "stage": "decomposition",
                    "status": "complete",
                    "detail": {"module_count": 2, "modules": ["alu", "decoder"]},
                },
                {"stage": "rtl_generation", "status": "complete", "detail": {}},
            ],
            "result": {"record": {"top_module": "cpu"}},
        }

        payload = module._job_payload(job_id)

        self.assertEqual(payload["progress"]["completed"], 2)
        self.assertEqual(payload["generation_tree"]["module"], "cpu")
        self.assertEqual(
            [child["module"] for child in payload["generation_tree"]["children"]],
            ["alu", "decoder"],
        )

    def test_job_payload_exposes_live_generated_code_from_workspace(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            combined = workspace / "combined" / "aes_cipher_top.sv"
            combined.parent.mkdir(parents=True)
            combined.write_text("module aes_cipher_top; endmodule\n", encoding="utf-8")
            job_id = "live-code-job"
            module.JOBS[job_id] = {
                "job_id": job_id,
                "state": "running",
                "task": "aes_cipher_top",
                "stages": [
                    {
                        "stage": "large_spec_workspace",
                        "status": "complete",
                        "detail": {"workspace_dir": str(workspace)},
                    }
                ],
                "result": None,
            }

            payload = module._job_payload(job_id)

            self.assertEqual(payload["live_generated_code"], "module aes_cipher_top; endmodule\n")
            self.assertEqual(payload["live_generated_code_path"], str(combined))


if __name__ == "__main__":
    unittest.main()
