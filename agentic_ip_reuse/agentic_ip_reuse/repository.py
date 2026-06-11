from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .types import CRITERIA, IpAssessment, IpCandidate, IpDescription

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


class JsonIpRepository:
    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path)
        self._descriptions = self._load(self.catalog_path)

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[IpCandidate]:
        filters = filters or {}
        query_tokens = _tokens(query)
        candidates: List[IpCandidate] = []
        for description in self._descriptions.values():
            candidate = description.candidate
            if not _matches_filters(candidate, filters):
                continue
            haystack = _tokens(" ".join(_candidate_text(candidate)))
            overlap = len(query_tokens & haystack)
            score = overlap / max(len(query_tokens), 1)
            if overlap == 0 and query_tokens:
                continue
            copied = IpCandidate(**asdict(candidate))
            copied.score = round(score, 4)
            candidates.append(copied)
        candidates.sort(key=lambda item: (item.score, item.name), reverse=True)
        return candidates[: max(1, min(int(top_k), 50))]

    def inspect(self, ip_id: str) -> IpDescription:
        try:
            return self._descriptions[ip_id]
        except KeyError as exc:
            raise KeyError(f"unknown IP id: {ip_id}") from exc

    def score(self, candidate: IpCandidate, module_requirements: Dict[str, Any]) -> IpAssessment:
        module_name = str(module_requirements.get("module_name") or module_requirements.get("name") or "module")
        required_text = " ".join(
            str(item)
            for item in [
                module_name,
                module_requirements.get("role", ""),
                " ".join(_string_list(module_requirements.get("interfaces", []))),
                " ".join(_string_list(module_requirements.get("requirements", []))),
            ]
        )
        required_tokens = _tokens(required_text)
        candidate_tokens = _tokens(" ".join(_candidate_text(candidate)))
        interface_tokens = _tokens(" ".join(candidate.interfaces))
        requested_interfaces = _tokens(" ".join(_string_list(module_requirements.get("interfaces", []))))

        criteria_scores = {
            "function_match": _ratio(required_tokens, candidate_tokens),
            "interface_compatibility": 1.0 if requested_interfaces and requested_interfaces <= interface_tokens else _ratio(requested_interfaces, interface_tokens),
            "configurability": min(1.0, len(candidate.parameters) / 3.0),
            "verification_status": _quality_score(candidate.verification),
            "license": 0.0 if candidate.license.lower() in {"unknown", "proprietary"} else 1.0,
            "synthesis_support": _text_quality(candidate.synthesis, ["synth", "timing", "fpga", "asic", "yosys"]),
            "documentation_quality": _text_quality(candidate.documentation, ["complete", "integration", "datasheet", "examples", "register"]),
        }
        criteria_notes = {
            "function_match": "Token overlap between module intent and IP description.",
            "interface_compatibility": "Checks requested protocol/interface tokens against IP interfaces.",
            "configurability": "Rewards explicit width/depth/mode/clock parameters.",
            "verification_status": "Rewards testbench, formal, lint, or regression evidence.",
            "license": f"Catalog license is {candidate.license}.",
            "synthesis_support": candidate.synthesis,
            "documentation_quality": candidate.documentation,
        }
        total = sum(criteria_scores.get(item, 0.0) for item in CRITERIA) / len(CRITERIA)
        recommendation = "reuse" if total >= 0.68 else "adapt" if total >= 0.45 else "new RTL required"
        return IpAssessment(
            ip_id=candidate.ip_id,
            module_name=module_name,
            total_score=round(total, 4),
            criteria_scores={key: round(value, 4) for key, value in criteria_scores.items()},
            criteria_notes=criteria_notes,
            recommendation=recommendation,
        )

    @staticmethod
    def _load(path: Path) -> Dict[str, IpDescription]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        descriptions: Dict[str, IpDescription] = {}
        for item in payload.get("ips", []):
            candidate = IpCandidate(
                ip_id=str(item["ip_id"]),
                name=str(item["name"]),
                summary=str(item.get("summary", "")),
                category=str(item.get("category", "")),
                interfaces=_string_list(item.get("interfaces", [])),
                parameters=dict(item.get("parameters", {})),
                license=str(item.get("license", "unknown")),
                verification=_string_list(item.get("verification", [])),
                synthesis=str(item.get("synthesis", "unknown")),
                documentation=str(item.get("documentation", "unknown")),
                tags=_string_list(item.get("tags", [])),
                criteria=dict(item.get("criteria", {})),
            )
            descriptions[candidate.ip_id] = IpDescription(
                candidate=candidate,
                behavior=str(item.get("behavior", "")),
                integration_notes=_string_list(item.get("integration_notes", [])),
                known_limits=_string_list(item.get("known_limits", [])),
            )
        return descriptions


def _candidate_text(candidate: IpCandidate) -> Iterable[str]:
    yield candidate.name
    yield candidate.summary
    yield candidate.category
    yield " ".join(candidate.interfaces)
    yield " ".join(candidate.tags)
    yield " ".join(candidate.verification)
    yield candidate.synthesis
    yield candidate.documentation
    yield " ".join(candidate.criteria.values())


def _matches_filters(candidate: IpCandidate, filters: Dict[str, Any]) -> bool:
    category = filters.get("category")
    if category and str(category).lower() != candidate.category.lower():
        return False
    interface = filters.get("interface")
    if interface and str(interface).lower() not in {item.lower() for item in candidate.interfaces}:
        return False
    license_value = filters.get("license")
    if license_value and str(license_value).lower() != candidate.license.lower():
        return False
    return True


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}


def _ratio(needles: set[str], haystack: set[str]) -> float:
    if not needles:
        return 0.5
    return len(needles & haystack) / len(needles)


def _quality_score(items: List[str]) -> float:
    text = " ".join(items).lower()
    score = 0.0
    for keyword in ["testbench", "formal", "lint", "regression", "coverage"]:
        if keyword in text:
            score += 0.25
    return min(score, 1.0)


def _text_quality(text: str, keywords: List[str]) -> float:
    lowered = text.lower()
    if not lowered or lowered == "unknown":
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return max(0.35, min(1.0, hits / max(len(keywords) - 1, 1)))


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]
