"""Slice a monolithic spec down to the error-relevant portion for repair prompts.

When a repair fires, the failing diagnostic (or testbench mismatch) names the
offending signal/port/output; this module pulls the interface contract plus the
spec section(s) that actually describe those signals, so the repair prompt is
focused instead of carrying (or blindly tail-truncating) the whole spec.

Reuses existing primitives rather than re-implementing parsing:
- heading-bounded sectioning mirrors `manifest.markdown_chunks`,
- the interface contract comes from `manifest.verbatim_interface_excerpts`,
- diagnostic parsing inverts `rag_rtl.repair_cache.normalize_diagnostic_line`
  (which *strips* the quoted identifiers we want to *extract*).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rag_rtl.repair_cache import DIAG_LINE_RE, QUOTED_IDENTIFIER_RE

from .manifest import verbatim_interface_excerpts

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_FUNC_OUTPUT_RE = re.compile(r"Output\s+'?([A-Za-z_][A-Za-z0-9_$]*)'?", re.IGNORECASE)
# Identifiers shorter than this match too many sections to be useful as anchors.
_MIN_IDENT_LEN = 3


def spec_sections(spec: str) -> List[str]:
    """Split a markdown spec into heading-bounded sections (same rule as
    `markdown_chunks`): a line whose first non-space char is '#' starts a new
    section. Any preamble before the first heading is its own section."""
    sections: List[str] = []
    current: List[str] = []
    for line in spec.splitlines(keepends=True):
        if line.lstrip().startswith("#") and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))
    return sections


def extract_diagnostic_signals(diagnostics: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
    """Return (error_codes, identifiers) named by Verilator diagnostics.

    Verilator single-quotes the offending pin/signal/module (e.g.
    `Pin not found: 'wb_clk_i'`), which `normalize_diagnostic_line` discards; here
    we keep those quoted identifier tokens as the anchors to slice the spec on."""
    codes: Set[str] = set()
    identifiers: Set[str] = set()
    for diagnostic in diagnostics or []:
        for stream in ("stderr", "stdout"):
            text = str(diagnostic.get(stream, "") or "")
            for line in text.splitlines():
                match = DIAG_LINE_RE.search(line)
                if match is None:
                    continue
                severity, code, message = match.groups()
                codes.add((code or severity).upper())
                for quoted in QUOTED_IDENTIFIER_RE.findall(message):
                    for token in _IDENT_RE.findall(quoted.strip("'")):
                        if len(token) >= _MIN_IDENT_LEN:
                            identifiers.add(token)
    return codes, identifiers


def extract_function_signals(function_info: str) -> Set[str]:
    """Output signal names from a RealBench mismatch report (handles both
    `Output 'clk_out' has ...` and the unquoted `Output text_out has ...`)."""
    return set(_FUNC_OUTPUT_RE.findall(function_info or ""))


def _section_mentions(section_lower: str, identifiers_lower: Set[str]) -> bool:
    for ident in identifiers_lower:
        if re.search(rf"(?<![A-Za-z0-9_$]){re.escape(ident)}(?![A-Za-z0-9_$])", section_lower):
            return True
    return False


def slice_spec(spec: str, *, identifiers: Set[str], max_chars: int) -> str:
    """Build the focused slice: the interface contract (port tables + module
    headers) followed by the spec sections mentioning any `identifier`, in
    document order, within `max_chars`. Falls back to the overview section when
    no section matched. Returns "" only when nothing usable could be assembled."""
    if not spec:
        return ""
    parts: List[str] = []
    used = 0

    interface_budget = max(2000, max_chars // 2)
    excerpt = verbatim_interface_excerpts(spec, max_chars=interface_budget)
    if excerpt.strip():
        parts.append(excerpt)
        used += len(excerpt) + 2

    identifiers_lower = {ident.lower() for ident in identifiers if ident}
    sections = spec_sections(spec)
    matched = 0
    for section in sections:
        if used >= max_chars:
            break
        if not _section_mentions(section.lower(), identifiers_lower):
            continue
        if used + len(section) + 2 > max_chars:
            remaining = max_chars - used
            if remaining <= 500:
                break
            section = section[:remaining]
        parts.append(section)
        used += len(section) + 2
        matched += 1

    # Nothing matched a section: keep the first/overview section as an anchor.
    if matched == 0 and sections:
        first = sections[0]
        if used + len(first) + 2 <= max_chars:
            parts.append(first)

    return "\n\n".join(parts).strip()


def slice_spec_for_diagnostics(
    spec: str,
    *,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    function_info: Optional[str] = None,
    max_chars: int,
) -> str:
    """Orchestrator used by the repair loops. Derives the focus identifiers from
    syntax `diagnostics` or a functional `function_info` and returns the focused
    slice, or "" when there is no identifier to focus on (a pure PARSE failure) —
    the caller then keeps the full spec so repair is never starved of context."""
    if not spec:
        return ""
    identifiers: Set[str] = set()
    if diagnostics:
        identifiers |= extract_diagnostic_signals(diagnostics)[1]
    if function_info:
        identifiers |= extract_function_signals(function_info)
    if not identifiers:
        return ""
    return slice_spec(spec, identifiers=identifiers, max_chars=max_chars)
