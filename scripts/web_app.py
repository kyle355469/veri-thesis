from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_rtl.config import CacheConfig, FixedPipeConfig, RuntimeConfig, ToolCallingConfig
from rag_rtl.embeddings import make_embedder
from rag_rtl.json_utils import json_default
from rag_rtl.pipeline import FixedPipeRtlPipeline, RagRtlPipeline
from rag_rtl.reporting import build_latest_report
from rag_rtl.types import PipelineResponse, RtlTask
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier


DEFAULT_PRIMARY_INDEXES = [
    "indexes/rtl_hash",
    "indexes/rtl_datapath_hash",
    "indexes/smoke",
    "indexes/smoke_wrapper",
]
DEFAULT_STRUCTURE_INDEXES = [
    "indexes/rtl_datapath_hash",
    "indexes/smoke",
    "indexes/smoke_wrapper",
]


@dataclass
class WebSettings:
    host: str
    port: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RTL generation test website.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    settings = WebSettings(host=args.host, port=_first_open_port(args.host, args.port))
    server = ThreadingHTTPServer((settings.host, settings.port), _make_handler(settings))
    print(f"RTL generation website running at http://{settings.host}:{settings.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping website server.")


def _make_handler(settings: WebSettings):
    class RtlWebHandler(BaseHTTPRequestHandler):
        server_version = "RtlGenerationWeb/1.0"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/style.css":
                self._send_text(STYLE_CSS, "text/css; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._send_text(APP_JS, "text/javascript; charset=utf-8")
                return
            if parsed.path == "/api/config":
                self._send_json(_config_payload())
                return
            if parsed.path == "/api/tags":
                query = parse_qs(parsed.query)
                index = query.get("index", [""])[0]
                self._send_json({"tags": _tag_options(_safe_relative(index))})
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path != "/api/generate":
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                payload = self._read_json()
                result = _run_generation(payload)
            except Exception as exc:  # noqa: BLE001 - report errors to the UI.
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, **result})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} - {fmt % args}")

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body or "{}")

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=json_default, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or mimetypes.types_map.get(".txt", "text/plain"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"ok": False, "error": message}, status=status)

    return RtlWebHandler


def _config_payload() -> Dict[str, Any]:
    primary = _first_existing(DEFAULT_PRIMARY_INDEXES)
    structure = _first_existing(DEFAULT_STRUCTURE_INDEXES)
    return {
        "defaults": {
            "mode": "generate",
            "index": primary,
            "spec_index": primary,
            "code_structure_index": structure,
            "embedder": "hash",
            "target_hdl": "verilog",
            "retrieve_k": 8,
            "context_k": 4,
            "structure_retrieve_k": 8,
            "structure_context_k": 4,
            "max_repair_attempts": 1,
            "second_edition_repair_attempts": 1,
            "generation_temperature": 0.4,
            "max_tokens": 2048,
            "cache_mode": "keywords",
            "cache_reuse_threshold": 0.95,
            "cache_evidence_threshold": 0.88,
            "cache": "data/history_cache.json",
            "monitor": "runs/web_monitor.jsonl",
            "failed_log": "runs/web_failed_attempts.jsonl",
            "testbench": "",
            "test_command": "",
        },
        "indexes": _discover_indexes(),
        "tags": _tag_options(primary),
    }


def _run_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    mode = str(payload.get("mode") or "generate")
    selected_tags = [str(tag) for tag in payload.get("tags") or [] if str(tag).strip()]
    tag_match = str(payload.get("tag_match") or "any")
    tag_target = str(payload.get("tag_target") or "auto")
    embedder_name = str(payload.get("embedder") or "hash")
    embedder = make_embedder(embedder_name)
    verifier = RtlVerifier(
        testbench_path=_optional_path(payload.get("testbench")),
        test_command=_optional_string(payload.get("test_command")),
    )
    task = RtlTask(
        prompt=prompt,
        target_hdl=str(payload.get("target_hdl") or "verilog"),
        module_signature=_optional_string(payload.get("module_signature")),
        constraints=_constraints_from_payload(payload),
        max_repair_attempts=_int_value(payload, "max_repair_attempts", 1),
        top_module=_optional_string(payload.get("top_module")),
    )
    cache_config = CacheConfig(
        path=_path_value(payload, "cache", "data/history_cache.json"),
        mode=str(payload.get("cache_mode") or "keywords"),
        reuse_threshold=_float_value(payload, "cache_reuse_threshold", 0.95),
        evidence_threshold=_float_value(payload, "cache_evidence_threshold", 0.88),
    )
    runtime_config = RuntimeConfig(
        monitor_path=_path_value(payload, "monitor", "runs/web_monitor.jsonl"),
        failed_log_path=_path_value(payload, "failed_log", "runs/web_failed_attempts.jsonl"),
        verbose_generation=True,
        generation_temperature=_float_value(payload, "generation_temperature", 0.4),
        max_tokens=_int_value(payload, "max_tokens", 2048),
    )
    tool_config = ToolCallingConfig(
        enabled=bool(payload.get("enable_tool_calling")),
        choice=str(payload.get("tool_choice") or "auto"),
        max_rounds=_int_value(payload, "max_tool_rounds", 4),
    )
    started = time.perf_counter()
    if mode == "fixed_pipe":
        spec_store = _load_store(_path_value(payload, "spec_index", _first_existing(DEFAULT_PRIMARY_INDEXES)))
        structure_store = _load_store(
            _path_value(payload, "code_structure_index", _first_existing(DEFAULT_STRUCTURE_INDEXES))
        )
        if _should_filter_tags("fixed_pipe", tag_target, "primary"):
            spec_store = _filter_store_by_tags(spec_store, selected_tags, tag_match)
        if _should_filter_tags("fixed_pipe", tag_target, "structure"):
            structure_store = _filter_store_by_tags(structure_store, selected_tags, tag_match)
        pipeline = FixedPipeRtlPipeline(
            spec_store=spec_store,
            code_structure_store=structure_store,
            embedder=embedder,
            verifier=verifier,
            cache_config=cache_config,
            runtime_config=runtime_config,
            tool_config=tool_config,
            fixed_pipe_config=FixedPipeConfig(
                yosys_bin=str(payload.get("yosys_bin") or "yosys"),
                yosys_timeout_s=_int_value(payload, "yosys_timeout_s", 30),
                second_edition_repair_attempts=_int_value(payload, "second_edition_repair_attempts", 1),
            ),
        )
        response = pipeline.run(
            task,
            retrieve_k=_int_value(payload, "retrieve_k", 8),
            context_k=_int_value(payload, "context_k", 4),
            structure_retrieve_k=_int_value(payload, "structure_retrieve_k", 8),
            structure_context_k=_int_value(payload, "structure_context_k", 4),
        )
    else:
        store = _load_store(_path_value(payload, "index", _first_existing(DEFAULT_PRIMARY_INDEXES)))
        if _should_filter_tags("generate", tag_target, "primary"):
            store = _filter_store_by_tags(store, selected_tags, tag_match)
        pipeline = RagRtlPipeline(
            store=store,
            embedder=embedder,
            verifier=verifier,
            cache_config=cache_config,
            runtime_config=runtime_config,
            tool_config=tool_config,
        )
        response = pipeline.run(
            task,
            retrieve_k=_int_value(payload, "retrieve_k", 8),
            context_k=_int_value(payload, "context_k", 4),
        )

    report = build_latest_report(response)
    report["web"] = {
        "selected_tags": selected_tags,
        "tag_match": tag_match,
        "tag_target": tag_target,
        "elapsed_s": time.perf_counter() - started,
    }
    return {
        "report": report,
        "generations": _generation_steps(response),
    }


def _generation_steps(response: PipelineResponse) -> List[Dict[str, Any]]:
    steps: Dict[Tuple[str, int], Dict[str, Any]] = {}
    order: List[Tuple[str, int]] = []
    stage = "first"
    for action in response.llm_actions:
        name = action.get("action")
        if name == "second_edition_generation_attempt":
            stage = "second"
        action_stage = "second" if str(name or "").startswith("second_edition") else stage
        attempt = int(action.get("attempt") or 0)
        key = (action_stage, attempt)
        if key not in steps:
            steps[key] = {
                "stage": action_stage,
                "attempt": attempt,
                "model_text": "",
                "rtl": "",
                "result_code": "",
                "verification": None,
                "actions": [],
            }
            order.append(key)
        steps[key]["actions"].append(action)
        if name in {"llm_final_response", "second_edition_raw_model_text"}:
            steps[key]["model_text"] = action.get("content") or action.get("content_preview") or ""
        if name in {"rtl_extracted", "second_edition_rtl_extracted"}:
            steps[key]["rtl"] = action.get("rtl") or action.get("rtl_preview") or ""
            steps[key]["result_code"] = steps[key]["rtl"]
        if name in {"verification_result", "second_edition_verification_result"}:
            steps[key]["verification"] = {
                "passed": action.get("passed"),
                "syntax_passed": action.get("syntax_passed"),
                "lint_passed": action.get("lint_passed"),
                "failed_tools": action.get("failed_tools", []),
            }
    if not order and response.cache_source == "history":
        return [
            {
                "stage": "history",
                "attempt": 0,
                "model_text": "",
                "rtl": response.rtl,
                "result_code": response.rtl,
                "verification": {"passed": response.verification.passed},
                "actions": response.llm_actions,
            }
        ]
    if response.cache_source == "history":
        for step in steps.values():
            if not step["result_code"]:
                step["result_code"] = response.rtl
            if not step["rtl"]:
                step["rtl"] = response.rtl
    elif order:
        steps[order[-1]]["result_code"] = response.rtl
    return [steps[key] for key in order]


def _should_filter_tags(mode: str, target: str, store_role: str) -> bool:
    if target == "none":
        return False
    if target == "both":
        return True
    if target == "primary":
        return store_role == "primary"
    if target == "structure":
        return store_role == "structure"
    return store_role == ("structure" if mode == "fixed_pipe" else "primary")


def _filter_store_by_tags(store: VectorStore, selected_tags: List[str], match: str) -> VectorStore:
    if not selected_tags:
        return store
    selected = set(selected_tags)
    indices: List[int] = []
    for index, document in enumerate(store.documents):
        tags = set(document.tags)
        matched = selected <= tags if match == "all" else bool(selected & tags)
        if matched:
            indices.append(index)
    if not indices:
        raise ValueError(f"No documents matched the selected tags: {', '.join(selected_tags)}")
    return VectorStore([store.documents[index] for index in indices], store.vectors[np.asarray(indices)])


def _load_store(path: str | Path) -> VectorStore:
    path = _safe_relative(str(path))
    if not path:
        raise ValueError("Index path is required.")
    return VectorStore.load(ROOT / path)


def _tag_options(index: str | Path) -> List[Dict[str, Any]]:
    path = _safe_relative(str(index))
    doc_path = ROOT / path / "documents.jsonl"
    counts: Dict[str, int] = {}
    if not doc_path.exists():
        return []
    with doc_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            for tag in payload.get("tags") or []:
                counts[str(tag)] = counts.get(str(tag), 0) + 1
    return [{"tag": tag, "count": counts[tag]} for tag in sorted(counts, key=lambda item: item.lower())]


def _discover_indexes() -> List[str]:
    base = ROOT / "indexes"
    if not base.exists():
        return []
    indexes = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "documents.jsonl").exists() and (child / "vectors.npy").exists():
            indexes.append(str(child.relative_to(ROOT)))
    return indexes


def _first_existing(paths: Iterable[str]) -> str:
    for path in paths:
        if (ROOT / path / "documents.jsonl").exists() and (ROOT / path / "vectors.npy").exists():
            return path
    discovered = _discover_indexes()
    return discovered[0] if discovered else ""


def _constraints_from_payload(payload: Dict[str, Any]) -> List[str]:
    value = payload.get("constraints") or []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _path_value(payload: Dict[str, Any], key: str, default: str) -> str:
    value = str(payload.get(key) or default).strip()
    return _safe_relative(value)


def _safe_relative(path: str) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the repository: {path}") from exc
    return str(candidate.relative_to(ROOT))


def _optional_path(value: Any) -> Optional[str]:
    text = _optional_string(value)
    return _safe_relative(text) if text else None


def _optional_string(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _int_value(payload: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _float_value(payload: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _first_open_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No open port found from {preferred} to {preferred + 49}.")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RTL Generation Lab</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>RTL Generation Lab</h1>
      <p id="envLine">Local pipeline</p>
    </div>
    <button id="runButton" type="button">Run</button>
  </header>

  <main class="workspace">
    <section class="prompt-pane">
      <div class="section-head">
        <h2>User Prompt</h2>
        <span id="promptCount">0 chars</span>
      </div>
      <textarea id="prompt" spellcheck="false" placeholder="Paste or write the Verilog task here."></textarea>
      <div class="inline-fields">
        <label>Module signature <input id="moduleSignature" type="text"></label>
        <label>Top module <input id="topModule" type="text"></label>
        <label>Target HDL <input id="targetHdl" type="text" value="verilog"></label>
      </div>
      <label class="stacked">Constraints
        <textarea id="constraints" class="small-textarea" spellcheck="false"></textarea>
      </label>
    </section>

    <aside class="settings-pane">
      <h2>Settings</h2>
      <div class="control-grid">
        <label>Pipeline
          <select id="mode">
            <option value="generate">Generate</option>
            <option value="fixed_pipe">Fixed pipe</option>
          </select>
        </label>
        <label>Embedder <input id="embedder" type="text" value="hash"></label>
        <label>Index <select id="index"></select></label>
        <label>Spec index <select id="specIndex"></select></label>
        <label>Structure index <select id="codeStructureIndex"></select></label>
        <label>Retrieve K <input id="retrieveK" type="number" min="1" value="8"></label>
        <label>Context K <input id="contextK" type="number" min="1" value="4"></label>
        <label>Repair attempts <input id="maxRepairAttempts" type="number" min="0" value="1"></label>
        <label>Structure retrieve K <input id="structureRetrieveK" type="number" min="1" value="8"></label>
        <label>Structure context K <input id="structureContextK" type="number" min="1" value="4"></label>
        <label>Second repairs <input id="secondEditionRepairAttempts" type="number" min="0" value="1"></label>
        <label>Temperature <input id="generationTemperature" type="number" min="0" max="2" step="0.01" value="0.4"></label>
        <label>Max tokens <input id="maxTokens" type="number" min="1" value="2048"></label>
        <label>Cache mode
          <select id="cacheMode">
            <option value="keywords">keywords</option>
            <option value="direct">direct</option>
          </select>
        </label>
        <label>Reuse threshold <input id="cacheReuseThreshold" type="number" step="0.01" value="0.95"></label>
        <label>Evidence threshold <input id="cacheEvidenceThreshold" type="number" step="0.01" value="0.88"></label>
        <label>Tool choice <input id="toolChoice" type="text" value="auto"></label>
        <label>Tool rounds <input id="maxToolRounds" type="number" min="1" value="4"></label>
        <label>Yosys bin <input id="yosysBin" type="text" value="yosys"></label>
        <label>Yosys timeout <input id="yosysTimeout" type="number" min="1" value="30"></label>
        <label>Testbench <input id="testbench" type="text"></label>
        <label>Test command <input id="testCommand" type="text"></label>
        <label>Cache path <input id="cachePath" type="text" value="data/history_cache.json"></label>
        <label>Monitor log <input id="monitorPath" type="text" value="runs/web_monitor.jsonl"></label>
        <label>Failed log <input id="failedLogPath" type="text" value="runs/web_failed_attempts.jsonl"></label>
      </div>

      <div class="toggle-row">
        <label><input id="enableToolCalling" type="checkbox"> Enable tool calling</label>
      </div>

      <div class="tag-head">
        <h2>Tags</h2>
        <input id="tagSearch" type="search" placeholder="Filter tags">
      </div>
      <div class="tag-options">
        <label><input type="radio" name="tagMatch" value="any" checked> Any</label>
        <label><input type="radio" name="tagMatch" value="all"> All</label>
        <label>Target
          <select id="tagTarget">
            <option value="auto">auto</option>
            <option value="primary">primary</option>
            <option value="structure">structure</option>
            <option value="both">both</option>
            <option value="none">none</option>
          </select>
        </label>
      </div>
      <div id="tags" class="tags"></div>
    </aside>

    <section class="results-pane">
      <div class="section-head">
        <h2>Results</h2>
        <span id="status">Idle</span>
      </div>
      <div id="summary" class="summary-grid"></div>
      <div id="generations" class="generations"></div>
      <div class="report-block">
        <h2>Full Report</h2>
        <pre id="reportJson"></pre>
      </div>
    </section>
  </main>
  <script src="/app.js"></script>
</body>
</html>
"""


STYLE_CSS = """
:root {
  color-scheme: light;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --panel-soft: #eef3f7;
  --ink: #16202a;
  --muted: #687382;
  --line: #d8e0e8;
  --accent: #0f7c80;
  --accent-dark: #075e62;
  --danger: #b42318;
  --ok: #157347;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.topbar {
  min-height: 76px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 22px;
  background: #10242c;
  color: #fff;
}
h1, h2, p { margin: 0; }
h1 { font-size: 22px; font-weight: 740; }
h2 { font-size: 14px; font-weight: 760; }
.topbar p { margin-top: 3px; color: #c8d6df; font-size: 13px; }
button {
  min-width: 104px;
  height: 42px;
  border: 0;
  border-radius: 8px;
  background: var(--accent);
  color: white;
  font-weight: 760;
  cursor: pointer;
}
button:hover { background: var(--accent-dark); }
button:disabled { opacity: .55; cursor: progress; }
.workspace {
  display: grid;
  grid-template-columns: minmax(420px, 1.1fr) minmax(320px, .74fr);
  grid-template-areas:
    "prompt settings"
    "results settings";
  gap: 14px;
  padding: 14px;
}
.prompt-pane, .settings-pane, .results-pane {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.prompt-pane { grid-area: prompt; }
.settings-pane { grid-area: settings; max-height: calc(100vh - 104px); overflow: auto; position: sticky; top: 14px; }
.results-pane { grid-area: results; min-height: 340px; }
.section-head, .tag-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
#promptCount, #status { color: var(--muted); font-size: 12px; }
textarea, input, select {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  font: inherit;
}
input, select { min-height: 34px; padding: 7px 9px; }
textarea { padding: 10px; line-height: 1.48; resize: vertical; }
#prompt {
  min-height: 360px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 14px;
}
.small-textarea { min-height: 86px; margin-top: 6px; }
.inline-fields, .control-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.control-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
label { display: block; color: var(--muted); font-size: 12px; font-weight: 680; }
label input, label select { margin-top: 5px; color: var(--ink); font-weight: 500; }
.stacked { margin-top: 10px; }
.toggle-row, .tag-options {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  margin: 12px 0;
}
.toggle-row input, .tag-options input { width: auto; min-height: auto; margin: 0 5px 0 0; }
.tag-head { margin-top: 16px; }
.tag-head input { max-width: 168px; }
.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  max-height: 238px;
  overflow: auto;
  padding: 8px;
  background: var(--panel-soft);
  border-radius: 8px;
}
.tag-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 999px;
  padding: 5px 9px;
  font-size: 12px;
  cursor: pointer;
}
.tag-chip input { width: auto; min-height: auto; margin: 0; }
.tag-count { color: var(--muted); }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfcfd;
}
.metric span { display: block; color: var(--muted); font-size: 12px; }
.metric strong { display: block; margin-top: 4px; font-size: 18px; overflow-wrap: anywhere; }
.generation {
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
}
.generation-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  background: #e8f1f1;
  border-bottom: 1px solid var(--line);
}
.badge {
  display: inline-block;
  min-width: 58px;
  text-align: center;
  padding: 3px 7px;
  border-radius: 999px;
  font-size: 12px;
  color: #fff;
  background: var(--muted);
}
.badge.pass { background: var(--ok); }
.badge.fail { background: var(--danger); }
.code-columns {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0;
}
.code-panel { min-width: 0; padding: 10px; }
.code-panel + .code-panel { border-left: 1px solid var(--line); }
pre {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: 12px;
  line-height: 1.45;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.code-panel pre, #reportJson {
  max-height: 420px;
  overflow: auto;
  padding: 10px;
  background: #111820;
  color: #e9f1f5;
  border-radius: 6px;
}
.report-block { margin-top: 14px; }
.report-block h2 { margin-bottom: 8px; }
@media (max-width: 980px) {
  .workspace {
    grid-template-columns: 1fr;
    grid-template-areas: "prompt" "settings" "results";
  }
  .settings-pane { position: static; max-height: none; }
  .inline-fields, .control-grid, .summary-grid, .code-columns { grid-template-columns: 1fr; }
  .code-panel + .code-panel { border-left: 0; border-top: 1px solid var(--line); }
}
"""


APP_JS = """
const state = { config: null, tags: [] };

const $ = (id) => document.getElementById(id);

async function init() {
  const res = await fetch('/api/config');
  state.config = await res.json();
  populateIndexes();
  applyDefaults(state.config.defaults || {});
  state.tags = state.config.tags || [];
  renderTags();
  bindEvents();
}

function populateIndexes() {
  const indexes = state.config.indexes || [];
  for (const id of ['index', 'specIndex', 'codeStructureIndex']) {
    const select = $(id);
    select.innerHTML = indexes.map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`).join('');
  }
}

function applyDefaults(defaults) {
  $('mode').value = defaults.mode || 'generate';
  $('index').value = defaults.index || '';
  $('specIndex').value = defaults.spec_index || defaults.index || '';
  $('codeStructureIndex').value = defaults.code_structure_index || defaults.index || '';
  $('embedder').value = defaults.embedder || 'hash';
  $('targetHdl').value = defaults.target_hdl || 'verilog';
  $('retrieveK').value = defaults.retrieve_k || 8;
  $('contextK').value = defaults.context_k || 4;
  $('structureRetrieveK').value = defaults.structure_retrieve_k || 8;
  $('structureContextK').value = defaults.structure_context_k || 4;
  $('maxRepairAttempts').value = defaults.max_repair_attempts || 1;
  $('secondEditionRepairAttempts').value = defaults.second_edition_repair_attempts || 1;
  $('generationTemperature').value = defaults.generation_temperature ?? 0.4;
  $('maxTokens').value = defaults.max_tokens || 2048;
  $('cacheMode').value = defaults.cache_mode || 'keywords';
  $('cacheReuseThreshold').value = defaults.cache_reuse_threshold || 0.95;
  $('cacheEvidenceThreshold').value = defaults.cache_evidence_threshold || 0.88;
  $('cachePath').value = defaults.cache || 'data/history_cache.json';
  $('monitorPath').value = defaults.monitor || 'runs/web_monitor.jsonl';
  $('failedLogPath').value = defaults.failed_log || 'runs/web_failed_attempts.jsonl';
  $('testbench').value = defaults.testbench || '';
  $('testCommand').value = defaults.test_command || '';
  updatePromptCount();
}

function bindEvents() {
  $('runButton').addEventListener('click', runGeneration);
  $('prompt').addEventListener('input', updatePromptCount);
  $('tagSearch').addEventListener('input', renderTags);
  $('index').addEventListener('change', refreshTags);
  $('codeStructureIndex').addEventListener('change', refreshTags);
  $('mode').addEventListener('change', refreshTags);
}

function selectedTagIndex() {
  return $('mode').value === 'fixed_pipe' ? $('codeStructureIndex').value : $('index').value;
}

async function refreshTags() {
  const res = await fetch('/api/tags?index=' + encodeURIComponent(selectedTagIndex()));
  const payload = await res.json();
  state.tags = payload.tags || [];
  renderTags();
}

function renderTags() {
  const query = $('tagSearch')?.value?.trim().toLowerCase() || '';
  const items = state.tags.filter((item) => item.tag.toLowerCase().includes(query));
  $('tags').innerHTML = items.map((item) => `
    <label class="tag-chip">
      <input type="checkbox" value="${escapeHtml(item.tag)}">
      <span>${escapeHtml(item.tag)}</span>
      <span class="tag-count">${item.count}</span>
    </label>
  `).join('') || '<span class="tag-count">No tags found</span>';
}

function updatePromptCount() {
  $('promptCount').textContent = `${$('prompt').value.length} chars`;
}

async function runGeneration() {
  const button = $('runButton');
  button.disabled = true;
  $('status').textContent = 'Running';
  $('summary').innerHTML = '';
  $('generations').innerHTML = '';
  $('reportJson').textContent = '';
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formPayload()),
    });
    const payload = await res.json();
    if (!payload.ok) throw new Error(payload.error || 'Generation failed');
    renderResult(payload.report, payload.generations || []);
    $('status').textContent = payload.report.summary?.passed ? 'Passed' : 'Finished';
  } catch (error) {
    $('status').textContent = 'Error';
    $('generations').innerHTML = `<div class="generation"><div class="generation-head"><strong>Error</strong><span class="badge fail">fail</span></div><div class="code-panel"><pre>${escapeHtml(error.message)}</pre></div></div>`;
  } finally {
    button.disabled = false;
  }
}

function formPayload() {
  return {
    mode: $('mode').value,
    prompt: $('prompt').value,
    module_signature: $('moduleSignature').value,
    top_module: $('topModule').value,
    target_hdl: $('targetHdl').value,
    constraints: $('constraints').value,
    embedder: $('embedder').value,
    index: $('index').value,
    spec_index: $('specIndex').value,
    code_structure_index: $('codeStructureIndex').value,
    retrieve_k: Number($('retrieveK').value),
    context_k: Number($('contextK').value),
    structure_retrieve_k: Number($('structureRetrieveK').value),
    structure_context_k: Number($('structureContextK').value),
    max_repair_attempts: Number($('maxRepairAttempts').value),
    second_edition_repair_attempts: Number($('secondEditionRepairAttempts').value),
    generation_temperature: Number($('generationTemperature').value),
    max_tokens: Number($('maxTokens').value),
    cache_mode: $('cacheMode').value,
    cache_reuse_threshold: Number($('cacheReuseThreshold').value),
    cache_evidence_threshold: Number($('cacheEvidenceThreshold').value),
    cache: $('cachePath').value,
    monitor: $('monitorPath').value,
    failed_log: $('failedLogPath').value,
    testbench: $('testbench').value,
    test_command: $('testCommand').value,
    enable_tool_calling: $('enableToolCalling').checked,
    tool_choice: $('toolChoice').value,
    max_tool_rounds: Number($('maxToolRounds').value),
    yosys_bin: $('yosysBin').value,
    yosys_timeout_s: Number($('yosysTimeout').value),
    tag_match: document.querySelector('input[name="tagMatch"]:checked')?.value || 'any',
    tag_target: $('tagTarget').value,
    tags: [...document.querySelectorAll('#tags input:checked')].map((item) => item.value),
  };
}

function renderResult(report, generations) {
  const summary = report.summary || {};
  $('summary').innerHTML = [
    metric('Passed', summary.passed ? 'yes' : 'no'),
    metric('Syntax', summary.syntax_passed ? 'yes' : 'no'),
    metric('Lint', summary.lint_passed ? 'yes' : 'no'),
    metric('Cache', summary.cache_source || 'miss'),
    metric('Repairs', summary.repair_attempts ?? 0),
    metric('Retrieved', summary.retrieved_count ?? 0),
    metric('Total s', formatNumber(summary.total_s)),
    metric('Tags', (report.web?.selected_tags || []).join(', ') || 'none'),
  ].join('');
  $('generations').innerHTML = generations.map(renderGeneration).join('') || '<div class="generation"><div class="code-panel"><pre>No generation actions were recorded.</pre></div></div>';
  $('reportJson').textContent = JSON.stringify(report, null, 2);
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function renderGeneration(step) {
  const passed = step.verification?.passed;
  const badge = passed === true ? '<span class="badge pass">pass</span>' : passed === false ? '<span class="badge fail">fail</span>' : '<span class="badge">n/a</span>';
  const title = `${step.stage === 'second' ? 'Second edition' : step.stage === 'history' ? 'History cache' : 'First edition'} attempt ${step.attempt}`;
  return `
    <article class="generation">
      <div class="generation-head"><strong>${escapeHtml(title)}</strong>${badge}</div>
      <div class="code-columns">
        <div class="code-panel">
          <h2>Model Text</h2>
          <pre>${escapeHtml(step.model_text || '')}</pre>
        </div>
        <div class="code-panel">
          <h2>Extracted RTL</h2>
          <pre>${escapeHtml(step.rtl || '')}</pre>
        </div>
        <div class="code-panel">
          <h2>Result Code</h2>
          <pre>${escapeHtml(step.result_code || step.rtl || '')}</pre>
        </div>
      </div>
    </article>
  `;
}

function formatNumber(value) {
  return typeof value === 'number' ? value.toFixed(3) : '';
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

init();
"""


if __name__ == "__main__":
    main()
