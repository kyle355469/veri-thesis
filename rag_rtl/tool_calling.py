from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .json_utils import json_default
from .retrieval import LexicalReranker, Retriever
from .summarizer import ContextSummarizer
from .verifier import RtlVerifier


RTL_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_rtl_context",
            "description": "Retrieve relevant RTL examples from the local vector index for the current hardware design task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Verilog/SystemVerilog design or verification request to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of vector hits to retrieve before reranking.",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 8,
                    },
                    "context_k": {
                        "type": "integer",
                        "description": "Number of reranked examples to return.",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_yosys",
            "description": "Run Yosys syntax/elaboration checks on candidate Verilog RTL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rtl": {"type": "string", "description": "Candidate Verilog/SystemVerilog source code."},
                    "top_module": {
                        "type": "string",
                        "description": "Optional top module name for hierarchy checking.",
                    },
                },
                "required": ["rtl"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_verilator",
            "description": "Run Verilator lint-only checks on candidate Verilog/SystemVerilog RTL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rtl": {"type": "string", "description": "Candidate Verilog/SystemVerilog source code."},
                },
                "required": ["rtl"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_rtl",
            "description": "Run the complete configured RTL verifier, including Yosys, Verilator, and any external testbench.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rtl": {"type": "string", "description": "Candidate Verilog/SystemVerilog source code."},
                    "top_module": {
                        "type": "string",
                        "description": "Optional top module name for hierarchy checking and external test command placeholders.",
                    },
                },
                "required": ["rtl"],
            },
        },
    },
]


class RtlToolExecutor:
    def __init__(
        self,
        retriever: Retriever,
        reranker: LexicalReranker,
        summarizer: ContextSummarizer,
        verifier: RtlVerifier,
        default_top_module: Optional[str] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.summarizer = summarizer
        self.verifier = verifier
        self.default_top_module = default_top_module

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name == "retrieve_rtl_context":
                payload = self.retrieve_rtl_context(
                    query=str(arguments.get("query", "")),
                    top_k=int(arguments.get("top_k", 8)),
                    context_k=int(arguments.get("context_k", 4)),
                )
            elif name == "run_yosys":
                payload = self.run_yosys(
                    rtl=str(arguments.get("rtl", "")),
                    top_module=self._top_module(arguments),
                )
            elif name == "run_verilator":
                payload = self.run_verilator(rtl=str(arguments.get("rtl", "")))
            elif name == "verify_rtl":
                payload = self.verify_rtl(
                    rtl=str(arguments.get("rtl", "")),
                    top_module=self._top_module(arguments),
                )
            else:
                payload = {"ok": False, "error": f"unknown tool: {name}"}
        except Exception as exc:
            payload = {"ok": False, "tool": name, "error": str(exc)}
        return json.dumps(payload, default=json_default, ensure_ascii=False)

    def retrieve_rtl_context(self, query: str, top_k: int = 8, context_k: int = 4) -> Dict[str, Any]:
        top_k = min(max(top_k, 1), 20)
        context_k = min(max(context_k, 1), 10)
        hits = self.retriever.retrieve(query, top_k=top_k)
        hits = self.reranker.rerank(query, hits, top_k=context_k)
        hits = self.summarizer.maybe_summarize(hits)
        return {
            "ok": True,
            "tool": "retrieve_rtl_context",
            "query": query,
            "hits": [
                {
                    "doc_id": hit.document.doc_id,
                    "score": hit.score,
                    "rerank_score": hit.rerank_score,
                    "tags": hit.document.tags,
                    "problem": hit.document.problem,
                    "solution": hit.document.solution,
                }
                for hit in hits
            ],
        }

    def run_yosys(self, rtl: str, top_module: Optional[str] = None) -> Dict[str, Any]:
        return {"ok": True, "tool": "run_yosys", "diagnostic": self.verifier.run_yosys(rtl, top_module)}

    def run_verilator(self, rtl: str) -> Dict[str, Any]:
        return {"ok": True, "tool": "run_verilator", "diagnostic": self.verifier.run_verilator(rtl)}

    def verify_rtl(self, rtl: str, top_module: Optional[str] = None) -> Dict[str, Any]:
        report = self.verifier.verify(rtl, top_module=top_module)
        return {"ok": True, "tool": "verify_rtl", "passed": report.passed, "verification": report}

    def _top_module(self, arguments: Dict[str, Any]) -> Optional[str]:
        top_module = arguments.get("top_module") or self.default_top_module
        return str(top_module) if top_module else None
