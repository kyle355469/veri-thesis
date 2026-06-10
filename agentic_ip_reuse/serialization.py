from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rag_rtl.json_utils import dumps_json, json_default

from .constants import JSON_BLOCK_RE
from .types import AgenticIpReuseResult


def dumps_result(result: AgenticIpReuseResult, *, indent: Optional[int] = 2) -> str:
    return dumps_json(result.to_dict(), indent=indent)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in JSON_BLOCK_RE.finditer(text))
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def string_or_unknown(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def dict_or_empty(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def metadata_json(value: Any) -> str:
    return json.dumps(value, default=json_default, ensure_ascii=False)
