from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Keys the planner (and downstream legacy adapter) have used for the selected IP.
_IP_KEYS = ("selected_ip", "ip", "ip_id", "reuse_ip", "selected_doc_id", "reuse")
_MODULE_KEYS = ("module_name", "module", "name")

# difflib ratio below which a remap is too speculative to trust.
_FUZZY_CUTOFF = 0.6


def ground_reuse_decisions(
    plan: Dict[str, Any], catalog_ids: Sequence[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Closed-vocabulary validation (#2).

    Rewrite every reuse_decisions entry so its ``selected_ip`` is a real catalog
    ip_id: keep exact matches, remap near-misses (e.g. ``e203_exu_core`` ->
    ``e203_exu``), and drop names with no plausible catalog match (marking the
    module ``new_rtl_required``). Also normalizes the IP onto ``selected_ip`` and
    the module name onto ``module_name`` so the artifact matrix and legacy
    adapter both read it. Returns ``(plan, report)``; ``plan`` is mutated in place.

    With an empty catalog the plan is returned untouched.
    """
    report = {"exact": 0, "remapped": 0, "dropped": 0, "unmatched_no_catalog": 0, "changes": []}
    decisions = plan.get("reuse_decisions")
    if not isinstance(decisions, list):
        return plan, report

    lookup = _build_lookup(catalog_ids)
    for entry in decisions:
        if not isinstance(entry, dict):
            continue
        module = _first(entry, _MODULE_KEYS)
        if module is not None:
            entry["module_name"] = module
        raw_ip = _first(entry, _IP_KEYS)
        if not raw_ip:
            continue
        if not lookup:
            # No catalog to validate against; leave the name as-is.
            entry["selected_ip"] = raw_ip
            report["unmatched_no_catalog"] += 1
            continue
        canonical, kind = _match_ip(raw_ip, lookup, list(catalog_ids))
        if canonical is None:
            entry["selected_ip"] = None
            entry["new_rtl_required"] = True
            entry.setdefault("risk_notes", [])
            if isinstance(entry["risk_notes"], list):
                entry["risk_notes"].append(
                    f"Planner named '{raw_ip}', absent from the catalog; dropped, generate new RTL."
                )
            report["dropped"] += 1
            report["changes"].append({"module": module, "from": raw_ip, "to": None, "kind": "dropped"})
        else:
            entry["selected_ip"] = canonical
            if kind == "exact":
                report["exact"] += 1
            else:
                report["remapped"] += 1
                report["changes"].append(
                    {"module": module, "from": raw_ip, "to": canonical, "kind": kind}
                )
    return plan, report


def completeness_gaps(plan: Dict[str, Any], has_catalog: bool) -> List[str]:
    """Plan-completeness check (#3).

    Names what a non-trivial plan is missing. Only flags reuse/integration gaps
    when the task actually has reuse candidates (``has_catalog``) — leaf modules
    with no catalog legitimately have neither.
    """
    gaps: List[str] = []
    if has_catalog and not _nonempty_list(plan.get("reuse_decisions")):
        gaps.append("reuse_decisions")
    if has_catalog and not _nonempty_list(plan.get("integration_plan")):
        gaps.append("integration_plan")
    if not _nonempty(plan.get("requirements")):
        gaps.append("requirements")
    return gaps


def completeness_score(plan: Dict[str, Any]) -> int:
    """Higher = more complete. Used to keep the better of two plan attempts."""
    reuse = [
        entry
        for entry in _as_list(plan.get("reuse_decisions"))
        if isinstance(entry, dict) and _first(entry, _IP_KEYS)
    ]
    return len(reuse) + len(_as_list(plan.get("integration_plan"))) + len(_as_list(plan.get("modules")))


def _build_lookup(catalog_ids: Sequence[str]) -> Dict[str, str]:
    """Lowercased identifier -> canonical ip_id."""
    return {str(ip).lower(): str(ip) for ip in catalog_ids if ip}


def _match_ip(
    name: str, lookup: Dict[str, str], catalog_ids: List[str]
) -> Tuple[Optional[str], str]:
    key = name.strip().lower()
    if key in lookup:
        return lookup[key], "exact"
    # Prefix/substring kinship handles invented suffixes/prefixes
    # (e203_exu_core -> e203_exu, sirv_ram -> sirv_gnrl_ram is left to fuzzy).
    kin = [
        cid
        for low, cid in lookup.items()
        if len(low) >= 4 and (key.startswith(low) or low.startswith(key) or key in low or low in key)
    ]
    if kin:
        return max(kin, key=len), "prefix"
    close = difflib.get_close_matches(key, list(lookup.keys()), n=1, cutoff=_FUZZY_CUTOFF)
    if close:
        return lookup[close[0]], "fuzzy"
    return None, "dropped"


def _first(entry: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() not in {"none", "null"}:
            return value.strip()
    return None


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _nonempty(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_nonempty(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None
