from __future__ import annotations

import re


ACTION_VALUES = {"reuse", "configure", "adapt", "new"}
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
HDL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
MODULE_DECL_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")

METADATA_ALIASES = {
    "function_match": ("function_match", "function", "behavior"),
    "interface_compatibility": ("interface_compatibility", "interface", "bus", "protocol"),
    "configurability": ("configurability", "parameters", "parameterized", "configurable"),
    "verification_status": ("verification_status", "verification", "verified", "testbench", "formal"),
    "license": ("license", "licence"),
    "synthesis_support": ("synthesis_support", "synthesis", "synthesizable", "timing"),
    "documentation_quality": ("documentation_quality", "documentation", "docs", "readme"),
}

MODULE_CATEGORIES = [
    "Input Interface",
    "Buffer / FIFO",
    "Processing Core",
    "Memory Controller",
    "Output Interface",
]

CRITERIA = [
    "function_match",
    "interface_compatibility",
    "configurability",
    "verification_status",
    "license",
    "synthesis_support",
    "documentation_quality",
]
