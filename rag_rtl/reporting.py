from __future__ import annotations

from typing import Any, Dict, List, Optional

from .json_utils import json_default
from .types import Diagnostic, PipelineResponse


def build_latest_report(response: PipelineResponse) -> Dict[str, Any]:
    cache_decision = response.metadata.get("cache_decision") or {}
    best_history_match = response.metadata.get("best_history_match")
    return {
        "summary": {
            "passed": response.verification.passed,
            "syntax_passed": response.verification.syntax_passed,
            "lint_passed": response.verification.lint_passed,
            "cache_source": response.cache_source,
            "repair_attempts": response.repair_attempts,
            "retrieved_count": len(response.retrieved_doc_ids),
            "total_s": response.timings.get("total_s"),
        },
        "task": {
            "prompt": response.prompt,
        },
        "llm_actions": response.llm_actions,
        "cache": {
            "decision": cache_decision.get("decision"),
            "mode": cache_decision.get("mode"),
            "score": cache_decision.get("score"),
            "matched_query": cache_decision.get("matched_query"),
            "candidate_count": cache_decision.get("candidate_count"),
            "query_keywords": cache_decision.get("query_keywords", []),
            "matched_keywords": cache_decision.get("matched_keywords", []),
            "best_history_match": _summarize_history_match(best_history_match),
        },
        "retrieval": {
            "doc_ids": response.retrieved_doc_ids,
        },
        "verification": {
            "passed": response.verification.passed,
            "syntax_passed": response.verification.syntax_passed,
            "lint_passed": response.verification.lint_passed,
            "diagnostics": [_format_diagnostic(diagnostic) for diagnostic in response.verification.diagnostics],
        },
        "timings": response.timings,
        "rtl": {
            "line_count": len(response.rtl.splitlines()) if response.rtl else 0,
            "code": response.rtl,
        },
        "raw_metadata": response.metadata,
    }


def _format_diagnostic(diagnostic: Diagnostic) -> Dict[str, Any]:
    return {
        "tool": diagnostic.tool,
        "passed": diagnostic.passed,
        "missing": diagnostic.missing,
        "returncode": diagnostic.returncode,
        "stdout_tail": _tail(diagnostic.stdout),
        "stderr_tail": _tail(diagnostic.stderr),
    }


def _summarize_history_match(match: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not match:
        return None
    return {
        "query": match.get("query"),
        "score": match.get("score"),
        "matched_keywords": match.get("matched_keywords", []),
    }


def _tail(text: str, limit: int = 2500) -> str:
    text = text or ""
    return text[-limit:]
