from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .agent import AgentConfig, AgenticIpReuseAgent
from .cli import default_catalog_path
from .hierarchical import HierarchicalAgent, HierarchicalConfig
from .json_utils import json_default, preview_text
from .llm import MockLlmClient, VllmClient
from .repository import JsonIpRepository
from .tools import AgentToolExecutor
from .types import AgentResult, DesignTask


DEFAULT_OUTPUT_ROOT = "agentic_ip_reuse_web_runs"


@dataclass
class RunState:
    run_id: str
    config: Dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: str = ""
    traceback_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "events": self.events,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback_text,
        }


class RunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, RunState] = {}

    def create(self, config: Dict[str, Any]) -> RunState:
        run_id = uuid.uuid4().hex[:10]
        state = RunState(run_id=run_id, config=config)
        with self._lock:
            self._runs[run_id] = state
        return state

    def get(self, run_id: str) -> Optional[RunState]:
        with self._lock:
            return self._runs.get(run_id)

    def append_event(self, run_id: str, event: Dict[str, Any]) -> None:
        event = {"time": time.time(), **event}
        with self._lock:
            state = self._runs[run_id]
            state.events.append(event)

    def update(self, run_id: str, **updates: Any) -> None:
        with self._lock:
            state = self._runs[run_id]
            for key, value in updates.items():
                setattr(state, key, value)


class ObservableLlmClient:
    def __init__(self, client: Any, store: RunStore, run_id: str):
        self.client = client
        self.store = store
        self.run_id = run_id
        self.call_count = 0

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self.call_count += 1
        self.store.append_event(
            self.run_id,
            {
                "kind": "llm_call",
                "title": f"LLM call {self.call_count}",
                "message": _summarize_messages(messages),
                "data": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": parallel_tool_calls,
                    "available_tools": [_tool_name(tool) for tool in tools or []],
                    "message_count": len(messages),
                },
            },
        )
        message = self.client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        tool_calls = message.get("tool_calls") or []
        self.store.append_event(
            self.run_id,
            {
                "kind": "llm_response",
                "title": f"LLM response {self.call_count}",
                "message": _summarize_llm_message(message),
                "data": {
                    "content_preview": preview_text(str(message.get("content") or ""), 1200),
                    "tool_calls": [_summarize_tool_call(call) for call in tool_calls],
                },
            },
        )
        return message


class ObservableToolExecutor:
    def __init__(self, executor: AgentToolExecutor, store: RunStore, run_id: str):
        self.executor = executor
        self.store = store
        self.run_id = run_id
        self.repository = executor.repository
        self.output_dir = executor.output_dir

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        self.store.append_event(
            self.run_id,
            {
                "kind": "tool_call",
                "title": name,
                "message": f"Tool call: {name}",
                "data": {"arguments": arguments},
            },
        )
        result_text = self.executor.execute(name, arguments)
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            result = {"ok": False, "error": result_text}
        self.store.append_event(
            self.run_id,
            {
                "kind": "tool_result",
                "title": name,
                "message": _summarize_tool_result(name, result),
                "data": {"result": result},
                "ok": bool(result.get("ok", False)),
            },
        )
        return result_text


def run_agent_thread(store: RunStore, state: RunState) -> None:
    run_id = state.run_id
    config = state.config
    try:
        store.update(run_id, status="running", started_at=time.time())
        store.append_event(run_id, {"kind": "stage", "title": "Agent start", "message": "Preparing task, catalog, LLM, and tools."})

        task = DesignTask(
            prompt=str(config["prompt"]),
            target_hdl=str(config.get("target_hdl") or "systemverilog"),
            constraints=_lines(config.get("constraints")),
            known_interfaces=_lines(config.get("known_interfaces")),
            ppa_targets=_lines(config.get("ppa_targets")),
        )
        catalog = str(config.get("catalog") or default_catalog_path())
        output_dir = str(config.get("output_dir") or _default_output_dir(run_id))

        repository = JsonIpRepository(catalog)
        base_executor = AgentToolExecutor(repository, output_dir)
        executor = ObservableToolExecutor(base_executor, store, run_id)
        llm = ObservableLlmClient(_build_llm(config), store, run_id)
        agent_config = AgentConfig(
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 8192)),
            tool_choice=config.get("tool_choice") or "auto",
            max_steps=int(config.get("max_steps", 16)),
        )

        if bool(config.get("hierarchical", False)):
            store.append_event(run_id, {"kind": "stage", "title": "Hierarchical run", "message": "Recursive module decomposition is enabled."})
            h_agent = HierarchicalAgent(
                llm_client=llm,
                base_executor=executor,  # type: ignore[arg-type]
                agent_config=agent_config,
                h_config=HierarchicalConfig(max_depth=int(config.get("max_depth", 2))),
            )
            h_plan = h_agent.run(task)
            h_plan.write_hierarchical_summary(Path(output_dir))
            result = h_plan.result
        else:
            agent = AgenticIpReuseAgent(
                llm_client=llm,
                tool_executor=executor,  # type: ignore[arg-type]
                config=agent_config,
            )
            result = agent.run(task)

        report_path = Path(output_dir) / "agent_result.json"
        report_path.write_text(json.dumps(result.to_dict(), default=json_default, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result_dict = result.to_dict()
        result_dict["report_path"] = str(report_path)
        store.append_event(
            run_id,
            {
                "kind": "final",
                "title": "Final result",
                "message": f"Completed with stopped_reason={result.stopped_reason}.",
                "data": {"artifact_paths": result.artifact_paths, "report_path": str(report_path)},
            },
        )
        store.update(run_id, status="succeeded", ended_at=time.time(), result=result_dict)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        store.append_event(
            run_id,
            {
                "kind": "error",
                "title": "Run failed",
                "message": str(exc),
                "data": {"traceback": tb},
                "ok": False,
            },
        )
        store.update(run_id, status="failed", ended_at=time.time(), error=str(exc), traceback_text=tb)


def _build_llm(config: Dict[str, Any]) -> Any:
    if bool(config.get("mock_llm", True)):
        return MockLlmClient()
    return VllmClient(
        base_url=str(config.get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")),
        model=str(config.get("model") or os.getenv("VLLM_MODEL", "siliconmind-server")),
        api_key=str(config.get("api_key") or os.getenv("VLLM_API_KEY", "EMPTY")),
        timeout_s=int(config.get("llm_timeout_s", 1200)),
    )


def _default_output_dir(run_id: str) -> str:
    return str((Path.cwd() / DEFAULT_OUTPUT_ROOT / run_id).resolve())


def _lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _tool_name(tool: Dict[str, Any]) -> str:
    return str((tool.get("function") or {}).get("name") or "unknown")


def _summarize_messages(messages: Sequence[Dict[str, Any]]) -> str:
    last = messages[-1] if messages else {}
    role = last.get("role", "unknown")
    content = last.get("content")
    if isinstance(content, str) and content.strip():
        return f"{len(messages)} message(s); latest {role}: {preview_text(content, 240)}"
    if last.get("tool_calls"):
        return f"{len(messages)} message(s); latest {role} requested tool calls"
    return f"{len(messages)} message(s); latest role={role}"


def _summarize_llm_message(message: Dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        names = ", ".join(_summarize_tool_call(call)["name"] for call in tool_calls)
        return f"Model requested tool(s): {names}"
    return f"Model returned final/content: {preview_text(str(message.get('content') or ''), 240)}"


def _summarize_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function") or {}
    args = function.get("arguments") or "{}"
    try:
        parsed_args = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        parsed_args = {"_raw_arguments": str(args)}
    return {
        "id": call.get("id"),
        "name": str(function.get("name") or "unknown"),
        "arguments": parsed_args,
    }


def _summarize_tool_result(name: str, result: Dict[str, Any]) -> str:
    if not result.get("ok", False):
        return f"{name} failed: {preview_text(str(result.get('error', 'unknown error')), 240)}"
    if name == "search_reuse_ip":
        return f"Found {len(result.get('candidates') or [])} candidate(s)."
    if name == "inspect_reuse_ip":
        candidate = ((result.get("description") or {}).get("candidate") or {})
        return f"Inspected {candidate.get('ip_id', 'IP candidate')}."
    if name == "evaluate_ip_candidate":
        assessment = result.get("assessment") or {}
        return f"Scored {assessment.get('ip_id', 'IP')} as {assessment.get('recommendation', 'review')}."
    if name == "generate_rtl_module":
        return f"Wrote {result.get('module_name', 'module')} to {result.get('path', '')}."
    if name == "validate_verilog":
        errors = len(result.get("errors") or [])
        return f"Validation via {result.get('linter', 'unknown')}: {errors} error(s)."
    if name == "check_port_compatibility":
        return f"{len(result.get('matched') or [])} matched, {len(result.get('issues') or [])} issue(s)."
    return f"{name} completed."


class DashboardHandler(BaseHTTPRequestHandler):
    server: "DashboardServer"

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._send_text(HTTPStatus.OK, INDEX_HTML, "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/runs/"):
            run_id = self.path.rsplit("/", 1)[-1]
            state = self.server.store.get(run_id)
            if state is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown run: {run_id}"})
                return
            self._send_json(HTTPStatus.OK, state.to_dict())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            payload = self._read_json()
            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "prompt is required"})
                return
            if not payload.get("catalog"):
                payload["catalog"] = str(default_catalog_path())
            state = self.server.store.create(payload)
            thread = threading.Thread(target=run_agent_thread, args=(self.server.store, state), daemon=True)
            thread.start()
            self._send_json(HTTPStatus.ACCEPTED, {"run_id": state.run_id})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, default=json_default, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: HTTPStatus, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: RunStore):
        super().__init__(server_address, DashboardHandler)
        self.store = store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agentic IP Reuse web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    server = DashboardServer((args.host, args.port), RunStore())
    print(f"Agentic IP Reuse dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agentic IP Reuse Dashboard</title>
  <style>
    :root {
      --bg: #f5f6f1;
      --panel: #ffffff;
      --ink: #202322;
      --muted: #66706c;
      --line: #d9ded8;
      --green: #168a62;
      --amber: #b26b00;
      --red: #b42318;
      --cyan: #0b7285;
      --violet: #6950a1;
      --code: #111716;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, textarea, select { font: inherit; }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 400px) minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfbf8;
      padding: 18px;
      overflow: auto;
    }
    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 20px; }
    h2 { font-size: 15px; margin: 18px 0 10px; }
    h3 { font-size: 14px; }
    label { display: block; font-size: 12px; color: var(--muted); margin: 12px 0 6px; }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      min-width: 0;
    }
    textarea { resize: vertical; min-height: 96px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      color: var(--ink);
      font-size: 13px;
    }
    .check input { width: auto; }
    .actions { display: flex; gap: 8px; margin-top: 16px; }
    .btn {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 12px;
      background: #fff;
      cursor: pointer;
      min-height: 38px;
    }
    .btn.primary {
      background: var(--green);
      color: #fff;
      border-color: var(--green);
      flex: 1;
    }
    .btn:disabled { opacity: .55; cursor: wait; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #fff;
      font-size: 13px;
      color: var(--muted);
    }
    .dot { width: 8px; height: 8px; border-radius: 99px; background: var(--muted); }
    .running .dot { background: var(--amber); }
    .succeeded .dot { background: var(--green); }
    .failed .dot { background: var(--red); }
    .grid {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(320px, 42%) minmax(0, 1fr);
      gap: 1px;
      background: var(--line);
    }
    .pane {
      min-width: 0;
      min-height: 0;
      background: var(--panel);
      overflow: auto;
      padding: 16px;
    }
    .timeline {
      display: grid;
      gap: 10px;
    }
    .event {
      border: 1px solid var(--line);
      border-left: 4px solid var(--muted);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }
    .event.llm_call { border-left-color: var(--cyan); }
    .event.llm_response { border-left-color: var(--violet); }
    .event.tool_call { border-left-color: var(--amber); }
    .event.tool_result { border-left-color: var(--green); }
    .event.error { border-left-color: var(--red); }
    .event.final { border-left-color: var(--green); }
    .event-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 5px;
    }
    .kind {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .msg { color: var(--muted); font-size: 13px; line-height: 1.4; word-break: break-word; }
    details { margin-top: 8px; }
    summary { cursor: pointer; color: var(--cyan); font-size: 12px; }
    pre {
      margin: 8px 0 0;
      padding: 10px;
      border-radius: 6px;
      background: var(--code);
      color: #eef5f2;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .tabs {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .tab {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
    }
    .tab.active { background: #202322; color: #fff; border-color: #202322; }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .stage {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfbf8;
      min-height: 74px;
    }
    .stage strong { display: block; font-size: 13px; margin-bottom: 5px; }
    .stage span { color: var(--muted); font-size: 12px; }
    .error-box {
      border: 1px solid #f2b8b5;
      background: #fff7f6;
      color: var(--red);
      border-radius: 6px;
      padding: 10px;
      min-height: 46px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .muted { color: var(--muted); }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
      .stage-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>Agentic IP Reuse</h1>
      <label for="prompt">Prompt</label>
      <textarea id="prompt">Build a simple streaming FIR accelerator with reusable FIFO and AXI-lite control</textarea>
      <label for="catalog">Catalog path</label>
      <input id="catalog" value="">
      <label for="output_dir">Output directory</label>
      <input id="output_dir" value="">
      <div class="row">
        <div>
          <label for="target_hdl">Target HDL</label>
          <input id="target_hdl" value="systemverilog">
        </div>
        <div>
          <label for="max_steps">Max steps</label>
          <input id="max_steps" type="number" min="1" max="64" value="16">
        </div>
      </div>
      <label for="constraints">Constraints</label>
      <textarea id="constraints" placeholder="one per line"></textarea>
      <label for="known_interfaces">Known interfaces</label>
      <textarea id="known_interfaces" placeholder="AXI-lite&#10;valid-ready"></textarea>
      <label for="ppa_targets">PPA targets</label>
      <textarea id="ppa_targets" placeholder="low area&#10;single clock"></textarea>
      <div class="row">
        <div>
          <label for="temperature">Temperature</label>
          <input id="temperature" type="number" step="0.1" value="0.2">
        </div>
        <div>
          <label for="max_tokens">Max tokens</label>
          <input id="max_tokens" type="number" value="8192">
        </div>
      </div>
      <label for="base_url">vLLM base URL</label>
      <input id="base_url" value="http://localhost:8000/v1">
      <label for="model">Model</label>
      <input id="model" value="siliconmind-server">
      <label class="check"><input id="mock_llm" type="checkbox" checked> Use mock LLM</label>
      <label class="check"><input id="hierarchical" type="checkbox"> Hierarchical decomposition</label>
      <div class="actions">
        <button class="btn primary" id="runBtn">Run</button>
        <button class="btn" id="clearBtn">Clear</button>
      </div>
    </aside>
    <main>
      <header>
        <div>
          <h1>Process Viewer</h1>
          <div class="muted" id="runMeta">No run yet</div>
        </div>
        <div class="status" id="status"><span class="dot"></span><span>idle</span></div>
      </header>
      <div class="grid">
        <section class="pane">
          <h2>LLM and Tool Timeline</h2>
          <div class="timeline" id="timeline"></div>
        </section>
        <section class="pane">
          <div class="tabs">
            <button class="tab active" data-tab="stages">Stages</button>
            <button class="tab" data-tab="result">Final Result</button>
            <button class="tab" data-tab="errors">Errors</button>
            <button class="tab" data-tab="raw">Raw Run</button>
          </div>
          <div id="panel-stages"></div>
          <div id="panel-result" hidden><pre id="finalJson">{}</pre></div>
          <div id="panel-errors" hidden><div class="error-box" id="errorText">No errors.</div></div>
          <div id="panel-raw" hidden><pre id="rawJson">{}</pre></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    let currentRun = null;
    let timer = null;
    let lastPayload = null;

    function formPayload() {
      return {
        prompt: $("prompt").value,
        catalog: $("catalog").value,
        output_dir: $("output_dir").value,
        target_hdl: $("target_hdl").value || "systemverilog",
        constraints: $("constraints").value,
        known_interfaces: $("known_interfaces").value,
        ppa_targets: $("ppa_targets").value,
        temperature: Number($("temperature").value || 0.2),
        max_tokens: Number($("max_tokens").value || 8192),
        max_steps: Number($("max_steps").value || 16),
        base_url: $("base_url").value,
        model: $("model").value,
        mock_llm: $("mock_llm").checked,
        hierarchical: $("hierarchical").checked,
        max_depth: 2,
        tool_choice: "auto"
      };
    }

    async function startRun() {
      $("runBtn").disabled = true;
      renderStatus("queued");
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(formPayload())
      });
      const body = await response.json();
      if (!response.ok) {
        $("runBtn").disabled = false;
        renderError(body.error || "Failed to start run");
        renderStatus("failed");
        return;
      }
      currentRun = body.run_id;
      $("runMeta").textContent = `Run ${currentRun}`;
      poll();
      timer = setInterval(poll, 850);
    }

    async function poll() {
      if (!currentRun) return;
      const response = await fetch(`/api/runs/${currentRun}`);
      const payload = await response.json();
      lastPayload = payload;
      render(payload);
      if (payload.status === "succeeded" || payload.status === "failed") {
        clearInterval(timer);
        timer = null;
        $("runBtn").disabled = false;
      }
    }

    function render(payload) {
      renderStatus(payload.status);
      $("runMeta").textContent = `Run ${payload.run_id} · ${payload.events.length} event(s)`;
      renderTimeline(payload.events || []);
      renderStages(payload.result && payload.result.structured_plan);
      $("finalJson").textContent = JSON.stringify((payload.result && payload.result.structured_plan) || {}, null, 2);
      $("rawJson").textContent = JSON.stringify(payload, null, 2);
      renderError(payload.error || collectToolErrors(payload.events || []));
    }

    function renderStatus(status) {
      const node = $("status");
      node.className = `status ${status}`;
      node.querySelector("span:last-child").textContent = status;
    }

    function renderTimeline(events) {
      const timeline = $("timeline");
      timeline.innerHTML = "";
      if (!events.length) {
        timeline.innerHTML = `<div class="muted">Waiting for run events.</div>`;
        return;
      }
      for (const event of events) {
        const item = document.createElement("article");
        item.className = `event ${event.kind || ""}`;
        item.innerHTML = `
          <div class="event-head">
            <h3>${escapeHtml(event.title || event.kind || "event")}</h3>
            <span class="kind">${escapeHtml(event.kind || "")}</span>
          </div>
          <div class="msg">${escapeHtml(event.message || "")}</div>
          <details>
            <summary>details</summary>
            <pre>${escapeHtml(JSON.stringify(event.data || {}, null, 2))}</pre>
          </details>
        `;
        timeline.appendChild(item);
      }
    }

    function renderStages(plan) {
      const stages = [
        ["Requirements", plan && plan.requirements],
        ["Module Decomposition", plan && plan.modules],
        ["IP Reuse Decisions", plan && plan.reuse_decisions],
        ["Integration Plan", plan && plan.integration_plan],
        ["Verification Plan", plan && plan.verification_plan],
        ["Debug Plan", plan && plan.debug_plan]
      ];
      $("panel-stages").innerHTML = `
        <div class="stage-grid">
          ${stages.map(([name, value]) => `
            <div class="stage">
              <strong>${escapeHtml(name)}</strong>
              <span>${stageSummary(value)}</span>
            </div>`).join("")}
        </div>
        <pre>${escapeHtml(JSON.stringify(plan || {}, null, 2))}</pre>
      `;
    }

    function stageSummary(value) {
      if (!value) return "Waiting for final plan.";
      if (Array.isArray(value)) return `${value.length} item(s)`;
      if (typeof value === "object") return `${Object.keys(value).length} field(s)`;
      return String(value);
    }

    function collectToolErrors(events) {
      const failed = events.filter((event) => event.kind === "tool_result" && event.ok === false);
      if (!failed.length) return "";
      return failed.map((event) => `${event.title}: ${event.message}`).join("\n");
    }

    function renderError(text) {
      $("errorText").textContent = text || "No errors.";
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        for (const id of ["stages", "result", "errors", "raw"]) {
          $(`panel-${id}`).hidden = button.dataset.tab !== id;
        }
      });
    });

    $("runBtn").addEventListener("click", startRun);
    $("clearBtn").addEventListener("click", () => {
      currentRun = null;
      lastPayload = null;
      if (timer) clearInterval(timer);
      timer = null;
      $("timeline").innerHTML = "";
      $("finalJson").textContent = "{}";
      $("rawJson").textContent = "{}";
      $("runMeta").textContent = "No run yet";
      renderError("");
      renderStatus("idle");
      $("runBtn").disabled = false;
    });

    renderStages(null);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
