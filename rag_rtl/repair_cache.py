"""Semantic cache of verified Verilator-diagnostic -> fix pairs.

Keys are identifier-stripped diagnostic signatures so the same failure pattern
recurs across tasks (a PINNOTFOUND for `wb_clk_i` and one for `core_clk` map to
the same signature). Values are unified-diff excerpts of the model's own
verified repairs — never benchmark reference RTL — and are only ever injected
into repair prompts as advisory guidance, never applied directly.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import Embedder
from .history_cache import HistorySemanticCache

DIAG_LINE_RE = re.compile(r"%(Error|Warning)(?:-([A-Z0-9]+))?\s*:?\s*(.*)")
FILE_LOCATION_RE = re.compile(r"\S+\.\w+:\d+(?::\d+)?:?\s*")
QUOTED_IDENTIFIER_RE = re.compile(r"'[^']*'")
NUMBER_RE = re.compile(r"\b\d+\b")
MAX_SIGNATURE_LINES = 40


@dataclass
class DiagnosticSignature:
    text: str
    error_codes: List[str]
    raw_excerpt: str


@dataclass
class RepairHint:
    text: str
    score: float
    decision: str
    error_codes: List[str] = field(default_factory=list)


def normalize_diagnostic_line(line: str) -> Optional[tuple[str, str]]:
    """Return (error_code, normalized_message) for a %Error/%Warning line, else None."""
    match = DIAG_LINE_RE.search(line)
    if match is None:
        return None
    severity, code, message = match.groups()
    code = code or severity.upper()
    message = FILE_LOCATION_RE.sub("", message)
    message = QUOTED_IDENTIFIER_RE.sub("'<id>'", message)
    message = NUMBER_RE.sub("<n>", message)
    message = re.sub(r"\s+", " ", message).strip()
    return code, f"%{severity}-{code}: {message}" if match.group(2) else f"%{severity}: {message}"


def normalize_diagnostics(diagnostics: List[Dict[str, Any]]) -> Optional[DiagnosticSignature]:
    """Distill verifier diagnostics (asdict'd rag_rtl.types.Diagnostic items)
    into a path- and identifier-free signature suitable as a semantic cache key."""
    lines: List[str] = []
    codes: List[str] = []
    raw_lines: List[str] = []
    for diagnostic in diagnostics or []:
        for stream in ("stderr", "stdout"):
            for line in str(diagnostic.get(stream, "") or "").splitlines():
                normalized = normalize_diagnostic_line(line)
                if normalized is None:
                    continue
                code, text = normalized
                if text not in lines:
                    lines.append(text)
                    codes.append(code)
                    raw_lines.append(line.strip())
    if not lines:
        return None
    order = sorted(range(len(lines)), key=lambda index: lines[index])[:MAX_SIGNATURE_LINES]
    return DiagnosticSignature(
        text="\n".join(lines[index] for index in order),
        error_codes=sorted({codes[index] for index in order}),
        raw_excerpt="\n".join(raw_lines[index] for index in order),
    )


def diagnostic_keywords(text: str) -> List[str]:
    """Keyword extractor for HistorySemanticCache gating: the Verilator error codes.

    Lowercased because the cache's keyword normalization lowercases entry keywords."""
    codes: List[str] = []
    for line in text.splitlines():
        normalized = normalize_diagnostic_line(line)
        if normalized is not None and normalized[0].lower() not in codes:
            codes.append(normalized[0].lower())
    return codes


def fix_diff(failing_rtl: str, repaired_rtl: str, max_chars: int) -> str:
    diff_lines = difflib.unified_diff(
        failing_rtl.splitlines(),
        repaired_rtl.splitlines(),
        fromfile="failing.sv",
        tofile="repaired.sv",
        lineterm="",
        n=2,
    )
    diff = "\n".join(diff_lines)
    if len(diff) > max_chars:
        diff = diff[:max_chars] + "\n... [diff truncated] ..."
    return diff


class RepairFixCache:
    """Wraps HistorySemanticCache with diagnostic-signature keys and diff values."""

    def __init__(
        self,
        embedder: Embedder,
        path: str | Path,
        evidence_threshold: float = 0.85,
        reuse_threshold: float = 0.95,
        max_hint_chars: int = 1800,
        max_size: int = 1000,
    ):
        self.cache = HistorySemanticCache(
            embedder=embedder,
            path=path,
            reuse_threshold=reuse_threshold,
            evidence_threshold=evidence_threshold,
            mode="keywords",
            max_size=max_size,
            keyword_extractor=diagnostic_keywords,
        )
        self.max_hint_chars = max_hint_chars
        self._lock = threading.Lock()
        self._stats = {"lookups": 0, "hits": 0, "puts": 0, "hit_scores": []}

    def lookup_hint(self, signature: Optional[DiagnosticSignature]) -> Optional[RepairHint]:
        if signature is None:
            return None
        lookup = self.cache.lookup(signature.text)
        with self._lock:
            self._stats["lookups"] += 1
            if lookup.decision in {"reuse", "evidence"}:
                self._stats["hits"] += 1
                self._stats["hit_scores"].append(round(float(lookup.score or 0.0), 4))
        if lookup.decision not in {"reuse", "evidence"} or lookup.entry is None:
            return None
        try:
            payload = json.loads(lookup.entry.result)
        except (TypeError, json.JSONDecodeError):
            return None
        fix_note = str(payload.get("fix_note", "")).strip()
        if not fix_note:
            return None
        codes = [str(code) for code in payload.get("error_codes", [])]
        text = (
            f"Diagnostic pattern: {', '.join(codes) or 'unclassified'}\n"
            f"Previously verified fix (unified diff excerpt):\n{fix_note}"
        )
        if len(text) > self.max_hint_chars:
            text = text[: self.max_hint_chars] + "\n... [hint truncated] ..."
        return RepairHint(
            text=text,
            score=float(lookup.score or 0.0),
            decision=lookup.decision,
            error_codes=codes,
        )

    def record_fix(
        self,
        signature: Optional[DiagnosticSignature],
        failing_rtl: str,
        repaired_rtl: str,
        *,
        task_id: str = "",
        attempt: int = 0,
    ) -> None:
        """Store a fix that PASSED verification. Callers must only invoke this
        after the repaired RTL verified clean (cache-poisoning guard)."""
        if signature is None:
            return
        diff = fix_diff(failing_rtl, repaired_rtl, self.max_hint_chars)
        if not diff.strip():
            return
        payload = {
            "error_codes": signature.error_codes,
            "fix_note": diff,
            "task_id": task_id,
            "verified": True,
        }
        self.cache.put(
            signature.text,
            json.dumps(payload, ensure_ascii=False),
            metadata={
                "keywords": [code.lower() for code in signature.error_codes],
                "task_id": task_id,
                "attempt": attempt,
                "verified": True,
            },
        )
        with self._lock:
            self._stats["puts"] += 1

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {**self._stats, "hit_scores": list(self._stats["hit_scores"]), "entries": len(self.cache.entries)}
