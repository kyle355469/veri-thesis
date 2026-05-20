from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag_rtl.json_utils import json_default, preview_text


DEFAULT_ALLOWED_COMMANDS = {
    "cat",
    "find",
    "grep",
    "head",
    "ls",
    "rg",
    "sed",
    "tail",
    "wc",
}


WORKSPACE_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file inside the configured workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path to read."},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file inside the configured workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative output path."},
                    "content": {"type": "string", "description": "Complete text content to write."},
                    "append": {"type": "boolean", "default": False},
                    "create_dirs": {"type": "boolean", "default": True},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory inside the configured workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a non-shell command inside the workspace. The first argv item must be in "
                "the configured command allowlist, such as rg, grep, ls, cat, sed, head, tail, or wc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command and arguments, for example [\"rg\", \"pattern\", \".\"].",
                    },
                    "cwd": {"type": "string", "default": "."},
                    "timeout_s": {"type": "integer", "minimum": 1, "maximum": 120},
                    "max_output_chars": {"type": "integer", "minimum": 100, "maximum": 20000},
                },
                "required": ["argv"],
            },
        },
    },
]


class WorkspaceToolExecutor:
    def __init__(
        self,
        root: str | Path = ".",
        allowed_commands: Optional[Iterable[str]] = None,
        timeout_s: int = 20,
        max_output_chars: int = 6000,
    ) -> None:
        self.root = Path(root).resolve()
        self.allowed_commands = set(allowed_commands or DEFAULT_ALLOWED_COMMANDS)
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name == "read_file":
                payload = self.read_file(
                    path=str(arguments.get("path", "")),
                    start_line=int(arguments.get("start_line", 1)),
                    max_lines=int(arguments.get("max_lines", 200)),
                )
            elif name == "write_file":
                payload = self.write_file(
                    path=str(arguments.get("path", "")),
                    content=str(arguments.get("content", "")),
                    append=bool(arguments.get("append", False)),
                    create_dirs=bool(arguments.get("create_dirs", True)),
                )
            elif name == "list_dir":
                payload = self.list_dir(
                    path=str(arguments.get("path", ".")),
                    max_entries=int(arguments.get("max_entries", 200)),
                )
            elif name == "run_command":
                payload = self.run_command(
                    argv=_string_list(arguments.get("argv", [])),
                    cwd=str(arguments.get("cwd", ".")),
                    timeout_s=int(arguments.get("timeout_s", self.timeout_s)),
                    max_output_chars=int(arguments.get("max_output_chars", self.max_output_chars)),
                )
            else:
                payload = {"ok": False, "error": f"unknown workspace tool: {name}"}
        except Exception as exc:
            payload = {"ok": False, "tool": name, "error": str(exc)}
        return json.dumps(payload, default=json_default, ensure_ascii=False)

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> Dict[str, Any]:
        target = self._resolve(path)
        start_line = max(start_line, 1)
        max_lines = min(max(max_lines, 1), 2000)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        return {
            "ok": True,
            "tool": "read_file",
            "path": self._relative(target),
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "total_lines": len(lines),
            "content": "\n".join(selected),
            "truncated": start_line - 1 + max_lines < len(lines),
        }

    def write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
        create_dirs: bool = True,
    ) -> Dict[str, Any]:
        target = self._resolve(path)
        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return {
            "ok": True,
            "tool": "write_file",
            "path": self._relative(target),
            "bytes": len(content.encode("utf-8")),
            "append": append,
        }

    def list_dir(self, path: str = ".", max_entries: int = 200) -> Dict[str, Any]:
        target = self._resolve(path)
        max_entries = min(max(max_entries, 1), 1000)
        entries = []
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name))[:max_entries]:
            entries.append(
                {
                    "name": child.name,
                    "path": self._relative(child),
                    "type": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return {
            "ok": True,
            "tool": "list_dir",
            "path": self._relative(target),
            "entries": entries,
            "truncated": len(entries) == max_entries,
        }

    def run_command(
        self,
        argv: Sequence[str],
        cwd: str = ".",
        timeout_s: Optional[int] = None,
        max_output_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not argv:
            return {"ok": False, "tool": "run_command", "error": "argv must not be empty"}
        command = Path(argv[0]).name
        if argv[0] != command:
            return {"ok": False, "tool": "run_command", "error": "command must be an executable name, not a path"}
        if command not in self.allowed_commands:
            return {
                "ok": False,
                "tool": "run_command",
                "error": f"command not allowed: {command}",
                "allowed_commands": sorted(self.allowed_commands),
            }
        blocked_arg = _blocked_path_argument(argv[1:])
        if blocked_arg:
            return {
                "ok": False,
                "tool": "run_command",
                "error": f"command argument escapes workspace: {blocked_arg}",
            }
        workdir = self._resolve(cwd)
        timeout = max(1, min(int(timeout_s or self.timeout_s), 120))
        output_limit = max(100, min(int(max_output_chars or self.max_output_chars), 20000))
        try:
            completed = subprocess.run(
                list(argv),
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "tool": "run_command",
                "argv": list(argv),
                "cwd": self._relative(workdir),
                "error": f"command timed out after {timeout}s",
                "stdout": preview_text(exc.stdout or "", output_limit),
                "stderr": preview_text(exc.stderr or "", output_limit),
            }
        return {
            "ok": True,
            "tool": "run_command",
            "argv": list(argv),
            "cwd": self._relative(workdir),
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout": preview_text(completed.stdout, output_limit),
            "stderr": preview_text(completed.stderr, output_limit),
        }

    def _resolve(self, path: str) -> Path:
        if not path:
            raise ValueError("path must not be empty")
        raw = Path(path)
        target = raw if raw.is_absolute() else self.root / raw
        resolved = target.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"path escapes workspace root: {path}")
        return resolved

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))


class CompositeToolExecutor:
    def __init__(self, *executors: Any) -> None:
        self.executors = executors

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        for executor in self.executors:
            if _executor_supports(executor, name):
                return executor.execute(name, arguments)
        return json.dumps({"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False)


def _executor_supports(executor: Any, name: str) -> bool:
    if hasattr(executor, name):
        return True
    if isinstance(executor, WorkspaceToolExecutor):
        return name in {"read_file", "write_file", "list_dir", "run_command"}
    return False


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _blocked_path_argument(args: Sequence[str]) -> str:
    for arg in args:
        if not arg or arg.startswith("-"):
            continue
        path = Path(arg)
        if path.is_absolute() or ".." in path.parts:
            return arg
    return ""
