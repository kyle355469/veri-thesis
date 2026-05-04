from __future__ import annotations

import re
from typing import List

from .types import RetrievalHit


MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:#\s*\(|\()", re.MULTILINE)
PORT_RE = re.compile(r"\b(input|output|inout)\b[^;,\n]*(?:[,;])", re.MULTILINE)


class ContextSummarizer:
    """Compression hook used when retrieved examples exceed the prompt budget."""

    def __init__(self, max_chars: int = 18000):
        self.max_chars = max_chars

    def maybe_summarize(self, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        total_chars = sum(len(hit.document.problem) + len(hit.document.solution) for hit in hits)
        if total_chars <= self.max_chars:
            return hits

        summarized: List[RetrievalHit] = []
        for hit in hits:
            doc = hit.document
            modules = ", ".join(MODULE_RE.findall(doc.solution)) or "unknown"
            ports = " ".join(match.group(0).strip() for match in PORT_RE.finditer(doc.solution))
            doc.problem = doc.problem[:1800].strip()
            doc.solution = (
                f"// Summarized retrieved RTL example\n"
                f"// Modules: {modules}\n"
                f"// Key ports: {ports[:1200]}\n"
                f"{doc.solution[:2200].strip()}"
            )
            doc.metadata["summarized"] = True
            summarized.append(hit)
        return summarized
