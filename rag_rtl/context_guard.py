"""Pre-flight context-window guard for OpenAI-compatible (vLLM) servers.

vLLM rejects any request where prompt_tokens + max_tokens exceeds the served
max_model_len (it does not clamp), and a prompt that alone exceeds the window
can only ever fail. This module checks both *before* the request is sent:

* the served context window is read once per server from ``GET /v1/models``
  (every vLLM model entry reports ``max_model_len``);
* the prompt is measured with the server's own tokenizer via ``POST /tokenize``
  (falling back to a chars/4 heuristic when the endpoint is unavailable);
* ``clamp_max_tokens`` then lowers the request's ``max_tokens`` to what still
  fits, or raises :class:`ContextLengthError` when the prompt leaves less than
  ``min_completion_tokens`` of room -- callers treat that as "skip this task",
  not as a server failure.

Set ``VLLM_CONTEXT_GUARD=0`` to disable the guard entirely (requests are then
sent exactly as composed, as before).
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Sequence

CHARS_PER_TOKEN = 4  # heuristic used only when /tokenize is unavailable
PER_MESSAGE_OVERHEAD_TOKENS = 8  # chat-template wrapper per message (heuristic path)
DEFAULT_MARGIN_TOKENS = 128
DEFAULT_MIN_COMPLETION_TOKENS = 512

_V1_SUFFIX_RE = re.compile(r"/v1/?$")

# Per-server caches so concurrent samples don't re-probe on every request.
_CACHE_LOCK = threading.Lock()
_SERVER_LIMITS: Dict[str, Optional[int]] = {}
_TOKENIZE_SUPPORTED: Dict[str, bool] = {}


class ContextLengthError(RuntimeError):
    """The prompt does not fit the served context window; nothing was sent."""

    def __init__(self, prompt_tokens: int, max_model_len: int, min_completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.max_model_len = max_model_len
        self.min_completion_tokens = min_completion_tokens
        super().__init__(
            f"prompt needs ~{prompt_tokens} tokens but the served context window is "
            f"{max_model_len}; fewer than {min_completion_tokens} tokens would remain "
            "for the completion, so the request was skipped before being sent"
        )


def guard_enabled() -> bool:
    return os.getenv("VLLM_CONTEXT_GUARD", "1") != "0"


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout_s: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def server_max_model_len(base_url: str, api_key: str = "EMPTY", timeout_s: int = 30) -> Optional[int]:
    """The served max_model_len reported by GET /v1/models (cached per base_url)."""
    with _CACHE_LOCK:
        if base_url in _SERVER_LIMITS:
            return _SERVER_LIMITS[base_url]
    limit: Optional[int] = None
    try:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        lengths = [
            int(item["max_model_len"])
            for item in body.get("data", [])
            if isinstance(item, dict) and item.get("max_model_len")
        ]
        if lengths:
            limit = min(lengths)
    except (urllib.error.URLError, OSError, ValueError, KeyError, json.JSONDecodeError):
        limit = None
    with _CACHE_LOCK:
        _SERVER_LIMITS[base_url] = limit
    return limit


def _messages_text(messages: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if content:
            parts.append(str(content))
    return "\n".join(parts)


def _heuristic_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    text = _messages_text(messages)
    return len(text) // CHARS_PER_TOKEN + PER_MESSAGE_OVERHEAD_TOKENS * max(len(messages), 1)


def count_prompt_tokens(
    base_url: str,
    model: str,
    messages: Sequence[Dict[str, Any]],
    api_key: str = "EMPTY",
    timeout_s: int = 60,
) -> int:
    """Prompt tokens after chat templating, via the server's /tokenize endpoint.

    vLLM serves /tokenize at the server root (not under /v1). Tries the chat
    (messages) form first so the template/think-tag overhead is included, then
    the flat-prompt form, then the chars/4 heuristic; a server without the
    endpoint is remembered so it is only probed once.
    """
    with _CACHE_LOCK:
        supported = _TOKENIZE_SUPPORTED.get(base_url, True)
    if supported:
        root = _V1_SUFFIX_RE.sub("", base_url.rstrip("/"))
        url = f"{root}/tokenize"
        for payload in (
            {"model": model, "messages": list(messages), "add_generation_prompt": True},
            {"model": model, "prompt": _messages_text(messages)},
        ):
            try:
                body = _post_json(url, payload, api_key, timeout_s)
            except urllib.error.HTTPError:
                continue  # this request shape was rejected; try the next one
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                break  # endpoint unreachable; fall through to the heuristic
            count = body.get("count")
            if isinstance(count, int):
                return count
        with _CACHE_LOCK:
            _TOKENIZE_SUPPORTED[base_url] = False
    return _heuristic_tokens(messages)


def clamp_max_tokens(
    payload: Dict[str, Any],
    base_url: str,
    api_key: str = "EMPTY",
    timeout_s: int = 60,
    max_model_len: Optional[int] = None,
    min_completion_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
    margin_tokens: int = DEFAULT_MARGIN_TOKENS,
) -> None:
    """Fit a chat-completion payload into the served context window, in place.

    Lowers ``payload["max_tokens"]`` so prompt + completion fits max_model_len,
    and raises :class:`ContextLengthError` when the prompt leaves less than
    ``min_completion_tokens`` of room. No-op when the guard is disabled or the
    context window is unknown.
    """
    if not guard_enabled():
        return
    limit = max_model_len or server_max_model_len(base_url, api_key=api_key)
    if not limit:
        return
    prompt_tokens = count_prompt_tokens(
        base_url, payload.get("model", ""), payload.get("messages", []), api_key=api_key, timeout_s=timeout_s
    )
    available = limit - prompt_tokens - margin_tokens
    if available < min_completion_tokens:
        raise ContextLengthError(prompt_tokens, limit, min_completion_tokens)
    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > available:
        payload["max_tokens"] = available
