"""Routing logic for the RealBench cascade router.

Decides whether a RealBench task/sample should go to the cheap ``direct`` model or
the full agentic ``pipeline``, keyed on the causal axis we validated: *is the
substantive logic implemented in this module, or delegated to instantiated
sub-modules / trivially thin?*

Three tiers, each usable independently (see scripts/run_realbench_routed.py):

* Tier-0 spec pre-route  -> :func:`route_pre` (LLM or keyword features -> :func:`decide_pre`)
* Tier-1 plan-probe       -> :func:`decide_plan` (judges the structured plan the pipeline emits)
* Tier-2 oracle label     -> :func:`oracle_label` (golden ``own_cells`` + structural guard)

The deciders are deterministic so the routing decision is auditable and diffable;
the only model call is the optional LLM feature extractor (Tier-0, ``--decider llm``).

Empirically the two flows are COMPLEMENTARY, not nested. The pipeline's retrieval / reuse /
integration / repair is exactly what wrapper & integration modules (class "A") need -- it
supplies the external sub-module interfaces they must wire, which direct single-shot only
hallucinates. But that same machinery invents spurious structure that derails self-contained
algorithmic modules (class "B"), which direct writes cleanly in one shot. So **A passes under
pipeline, B passes under direct**: routing sends A->pipeline and B->direct, and *either*
misroute loses a would-be pass (there is no "safe" over-provision direction).

RULE VERSIONS. The paragraph above describes the **v1** polarity, which the Jul 2026
ref-wrap cleaning falsified: with wrap samples invalidated, pipeline >= direct on 58/60
RealBench tasks, so misrouting to pipeline only costs compute while misrouting to direct
costs passes. **v2** (:func:`decide_pre_v2`) encodes the corrected, asymmetric rule --
pipeline is the default; direct is reserved for narrow self-contained computational leaves
(LUT/CRC/key-expand/small queues). Rules are selected by tag via :data:`ROUTE_RULES`
(``rule=`` on :func:`route_pre`, ``--route-rule`` on the routed runner, or the
``RTL_ROUTE_RULE`` env var); ``v1`` remains available to reproduce pre-cleaning runs.
v2 thresholds were fitted in-sample on the 60 RealBench module tasks against wrap-cleaned
hybrid + Full-2T outcomes (26/60 @120B profile, 22/60 @20B profile, zero lost-solvable).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_rtl.json_utils import dumps_json

# Empirical routing polarity: structural/wrapper modules (class "A") pass under the
# pipeline (it supplies the external sub-module interfaces they must wire); self-contained
# algorithmic modules (class "B") pass under direct single-shot (the pipeline's reuse /
# decomposition machinery invents structure and derails them). The flows are complementary.
CLASS_TO_FLOW = {"A": "pipeline", "B": "direct"}

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


@dataclass
class RouteFeatures:
    """Structured features used by :func:`decide_pre`.

    Emitted either by an LLM (``source="llm"``) or by regex (``source="keyword"``).
    ``raw`` keeps the model's verbatim JSON for auditing.
    """

    delegates_to_submodules: bool = False
    is_memory: bool = False
    thin_control: bool = False
    has_fsm: bool = False
    has_cdc: bool = False
    has_algorithm: bool = False
    est_state_bits: int = 0
    confidence: float = 1.0
    source: str = "keyword"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# RealBench appends a shared ``*_defines`` doc to every module spec; strip it so the
# size/keyword signals reflect the module itself, not the boilerplate.
_BOILERPLATE_MARKER = "some global variables involved in the document come from"


def _spec_body(spec: str) -> str:
    cut = spec.lower().find(_BOILERPLATE_MARKER)
    return spec[:cut] if cut > 0 else spec


# Keyword baseline (decider sub-ablation). Matched over the WHOLE body because the
# wrapper disclosure often lives in deeper sections (e.g. e203_ifu_minidec line 97
# "acts as a wrapper and mainly encapsulates the e203_exu_decode module").
_FSM_RE = re.compile(r"state machine|\bfsm\b|next[ _]state|current[ _]state|state transition|state diagram", re.I)
_CDC_RE = re.compile(r"clock domain|asynchronous|cross[- ]?clock|gray[ -]?code|metastab|\bfifo\b", re.I)
_ALGO_RE = re.compile(
    r"encrypt|decrypt|cipher|mixcolumns|subbytes|shiftrows|\bcrc\b|round transformation|"
    r"decode .*instruction|instruction.*decod|opcode|address generation|\bagu\b",
    re.I,
)
_WRAP_RE = re.compile(
    r"\bwrapper\b|top[- ]?level module|integrat|encapsulat|instantiat|coordinat|"
    r"consist[s]? of the following|sub[- ]?module|based on a generic|memory management",
    re.I,
)
_THIN_RE = re.compile(
    r"does not perform calculation|only generate.*control|simply (pass|forward)|does not respond to reset",
    re.I,
)
_MEM_RE = re.compile(r"\bram\b|tightly coupled memory|data storage|instruction storage|memory module", re.I)


def extract_features_keyword(spec: str) -> RouteFeatures:
    body = _spec_body(spec)
    return RouteFeatures(
        delegates_to_submodules=bool(_WRAP_RE.search(body)),
        is_memory=bool(_MEM_RE.search(body)),
        thin_control=bool(_THIN_RE.search(body)),
        has_fsm=bool(_FSM_RE.search(body)),
        has_cdc=bool(_CDC_RE.search(body)),
        has_algorithm=bool(_ALGO_RE.search(body)),
        est_state_bits=0,
        confidence=1.0,
        source="keyword",
        raw={},
    )


_FEATURE_PROMPT = """You are a routing feature extractor for an RTL code-generation pipeline.
Read the ENTIRE Verilog module specification below and judge, for THIS module only,
whether the substantive logic must be IMPLEMENTED in this module or is DELEGATED to
instantiated sub-modules / is trivially thin.

A module can SOUND like it computes (e.g. "preliminary decoding") yet actually be a
wrapper that instantiates another module and passes signals through. Read the whole
spec (including Registers / Submodule sections) before deciding.

Return ONLY a single JSON object, no prose, with exactly these fields:
{
  "delegates_to_submodules": boolean,  // mainly instantiates/encapsulates/integrates sub-modules and wires them
  "is_memory": boolean,                // RAM / memory encapsulation
  "thin_control": boolean,             // pure combinational control or field routing; "does not perform calculations"
  "has_fsm": boolean,                  // multi-state finite state machine / multi-cycle control
  "has_cdc": boolean,                  // clock-domain crossing / async FIFO / synchronizers
  "has_algorithm": boolean,            // datapath algorithm: encryption rounds, CRC, full instruction decode, address arithmetic
  "est_state_bits": integer,           // rough number of state/register bits THIS module itself must hold (0 for pure wrappers)
  "confidence": number                 // 0..1 confidence in the above
}

SPECIFICATION:
"""


def _parse_json_obj(text: str) -> Dict[str, Any]:
    """Extract the first balanced ``{...}`` object from model text."""
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return {}
    return {}


def _feat_cache_path(cache_dir: str | Path, spec: str) -> Path:
    digest = hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"feat_{digest}.json"


def extract_features_llm(
    spec: str,
    client: Any,
    cache_dir: Optional[str | Path] = None,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> RouteFeatures:
    """Prompt the served model for structured features (temperature 0, cached by spec hash)."""
    cache_path = _feat_cache_path(cache_dir, spec) if cache_dir else None
    if cache_path and cache_path.exists():
        return RouteFeatures(**json.loads(cache_path.read_text(encoding="utf-8")))

    message = client.chat(
        [{"role": "user", "content": _FEATURE_PROMPT + _spec_body(spec)}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = _parse_json_obj(str(message.get("content") or ""))

    def _b(key: str) -> bool:
        return bool(parsed.get(key))

    try:
        state_bits = int(parsed.get("est_state_bits") or 0)
    except (TypeError, ValueError):
        state_bits = 0
    try:
        conf = float(parsed.get("confidence")) if parsed.get("confidence") is not None else 1.0
    except (TypeError, ValueError):
        conf = 1.0

    feats = RouteFeatures(
        delegates_to_submodules=_b("delegates_to_submodules"),
        is_memory=_b("is_memory"),
        thin_control=_b("thin_control"),
        has_fsm=_b("has_fsm"),
        has_cdc=_b("has_cdc"),
        has_algorithm=_b("has_algorithm"),
        est_state_bits=state_bits,
        confidence=conf,
        source="llm",
        raw=parsed,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(dumps_json(feats.to_dict(), indent=2), encoding="utf-8")
    return feats


# --------------------------------------------------------------------------- #
# Tier-0 decision
# --------------------------------------------------------------------------- #


def decide_pre(feats: RouteFeatures, confidence_tau: float = 0.5) -> str:
    """Spec pre-route decision: ``"direct"`` | ``"pipeline"`` | ``"uncertain"``."""
    if feats.source == "llm" and feats.confidence < confidence_tau:
        return "uncertain"
    # Self-contained algorithm / state -> direct (the pipeline's scaffolding derails it).
    if feats.has_fsm or feats.has_cdc or feats.has_algorithm:
        return "direct"
    # Wrapper / thin / memory -> pipeline (needs the retrieved sub-module interfaces).
    if feats.delegates_to_submodules or feats.is_memory or feats.thin_control:
        return "pipeline"
    return "uncertain"


def _spec_ports(body: str) -> int:
    """Port count from a (boilerplate-stripped) spec body: markdown port-table rows,
    falling back to bare input/output/inout keyword hits."""
    return len(re.findall(r"\|\s*(input|output|inout)\b", body, re.I)) or len(
        re.findall(r"\b(input|output|inout)\b", body, re.I)
    )


def size_fallback(spec: str) -> str:
    """Resolve an ``uncertain`` Tier-0 decision by spec size (v1 ``pre`` arm only)."""
    body = _spec_body(spec)
    # Larger/wider specs skew toward self-contained algorithmic modules (Set B -> direct);
    # short specs skew toward thin wrappers/integration (Set A -> pipeline).
    return "direct" if (len(body) > 9000 or _spec_ports(body) > 25) else "pipeline"


# --------------------------------------------------------------------------- #
# Tier-0 v2: wrap-cleaned asymmetric rule (pipeline-default)
# --------------------------------------------------------------------------- #

# Numeric caps per model-scale profile. The rule shape is shared; only the caps widen
# with model capability (the direct-safe set grows with scale). ``bits_*`` gate on the
# LLM-estimated state bits this module itself holds -- it is what separates the
# direct-tie set (crc_16=16, clkgate=1) from pipeline-much-better tasks (rcon=36,
# aes_key_expand_128=132) at 20B.
V2_PROFILES: Dict[str, Dict[str, int]] = {
    "20b": {"ports": 30, "spec_plain": 6000, "bits_plain": 20, "spec_algo": 6000, "bits_algo": 64},
    "120b": {"ports": 30, "spec_plain": 6000, "bits_plain": 140, "spec_algo": 12000, "bits_algo": 140},
}


def decide_pre_v2(feats: RouteFeatures, spec: str, profile: str = "20b") -> str:
    """v2 spec pre-route: ``"direct"`` | ``"pipeline"`` (total -- never ``"uncertain"``).

    Pipeline is the safe default (post-cleaning, a pipeline misroute only costs compute);
    direct is carved out for narrow self-contained computational leaves. Clause order:

    1. delegation -> pipeline (wrappers/memories need the pipeline's interface wiring)
    2. thin field-routing wider than 10 ports -> pipeline (direct scores 0 on these at 20B)
    3. narrow + short spec + small own state -> direct
    4. self-contained algorithm within the profile caps -> direct

    ``has_fsm`` deliberately does NOT veto direct (key_expand/oitf/tx_filler carry fsm=Y).
    Calibrated for LLM features (``--decider llm``); the keyword extractor leaves
    ``est_state_bits`` at 0, which makes clauses 3-4 over-admit narrow tasks.
    """
    prof = V2_PROFILES[profile]
    body = _spec_body(spec)
    ports = _spec_ports(body)
    slen = len(body)
    bits = max(0, int(feats.est_state_bits or 0))
    if feats.delegates_to_submodules:
        return "pipeline"
    if feats.thin_control and ports > 10:
        return "pipeline"
    if ports <= prof["ports"] and slen <= prof["spec_plain"] and bits <= prof["bits_plain"]:
        return "direct"
    if feats.has_algorithm and ports <= prof["ports"] and slen <= prof["spec_algo"] and bits <= prof["bits_algo"]:
        return "direct"
    return "pipeline"


# Version-tag registry: tag -> (feats, spec) -> "direct" | "pipeline" | "uncertain".
# New rule revisions get a new tag here; old tags stay frozen so any past run can be
# reproduced by its recorded tag.
ROUTE_RULES: Dict[str, Any] = {
    "v1": lambda feats, spec: decide_pre(feats),
    "v2-20b": lambda feats, spec: decide_pre_v2(feats, spec, "20b"),
    "v2-120b": lambda feats, spec: decide_pre_v2(feats, spec, "120b"),
}
DEFAULT_ROUTE_RULE = "v1"


def resolve_route_rule(tag: Optional[str]) -> Tuple[str, Any]:
    """Resolve a rule tag (``None`` -> ``RTL_ROUTE_RULE`` env -> :data:`DEFAULT_ROUTE_RULE`)."""
    tag = tag or os.environ.get("RTL_ROUTE_RULE") or DEFAULT_ROUTE_RULE
    try:
        return tag, ROUTE_RULES[tag]
    except KeyError:
        raise ValueError(f"unknown route rule {tag!r}; known tags: {sorted(ROUTE_RULES)}") from None


def route_pre(
    spec: str,
    decider: str = "keyword",
    client: Any = None,
    cache_dir: Optional[str | Path] = None,
    *,
    force: bool = False,
    rule: Optional[str] = None,
) -> Tuple[str, RouteFeatures]:
    """Tier-0 entry point. ``rule`` selects the decision version from :data:`ROUTE_RULES`
    (default: ``RTL_ROUTE_RULE`` env var, else ``v1``). For v1, ``force`` resolves
    ``uncertain`` via :func:`size_fallback`; v2 rules are total and never uncertain."""
    _, decide = resolve_route_rule(rule)
    if decider == "llm":
        if client is None:
            raise ValueError("decider='llm' requires a chat client")
        feats = extract_features_llm(spec, client, cache_dir)
    else:
        feats = extract_features_keyword(spec)
    decision = decide(feats, spec)
    if force and decision == "uncertain":
        decision = size_fallback(spec)
    return decision, feats


# --------------------------------------------------------------------------- #
# Tier-1 decision (plan-probe)
# --------------------------------------------------------------------------- #

# Decisive delegation phrasing: a wrapper that instantiates a decoder is still a
# wrapper (the algorithm lives in the submodule), so these win over algorithm words.
_PLAN_WRAP_RE = re.compile(r"\bwrap(s|per|ping)?\b|encapsulat|reuse existing|memory management|tightly coupled", re.I)
_PLAN_INST_RE = re.compile(r"instantiat|integrat", re.I)
# Strong "this module implements the logic itself" verbs that block a delegation read.
_PLAN_OWNIMPL_RE = re.compile(
    r"\bimplement\b|encrypt|decrypt|cipher|\bcrc\b|\bround\b|state machine|\bfsm\b|"
    r"mixcolumns|subbytes|shiftrows|comput|calculat",
    re.I,
)
# Algorithmic intent that, absent a delegation read, means pipeline.
_PLAN_PIPE_RE = re.compile(
    r"\bimplement\b|encrypt|decrypt|cipher|\bcrc\b|\bround\b|state machine|\bfsm\b|"
    r"mixcolumns|subbytes|shiftrows|comput|calculat|decode .*instruction|instruction.*decod|"
    r"address generation",
    re.I,
)


def _unwrap_plan(planner_result_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(planner_result_dict, dict):
        return {}
    structured = planner_result_dict.get("structured_plan")
    if isinstance(structured, dict):
        return structured
    return planner_result_dict


def decide_plan(planner_result_dict: Dict[str, Any]) -> str:
    """Judge the structured plan -> ``"pipeline"`` (keep) | ``"direct"`` (bail).

    ``"pipeline"`` = a wrapper/integration module that needs the pipeline's interface wiring;
    ``"direct"`` = a self-contained algorithmic module the pipeline would derail, so discard
    the plan and generate directly. Reads the plan's natural-language ``requirements.functionality``
    + ``ppa_constraints.area`` (NOT the module/reuse counts, which the planner over/under-
    decomposes). Defaults to ``pipeline`` when unclear -- the plan is already a sunk cost, so
    only bail to direct when the module is clearly self-contained.
    """
    plan = _unwrap_plan(planner_result_dict)
    req = plan.get("requirements")
    req = req if isinstance(req, dict) else {}
    func = str(req.get("functionality") or "")
    ppa = req.get("ppa_constraints")
    if isinstance(ppa, dict):
        area = str(ppa.get("area") or "")
    elif isinstance(ppa, str):
        area = ppa
    else:
        area = ""
    text = f"{func} {area}"
    own_impl = bool(_PLAN_OWNIMPL_RE.search(func))
    # Decisive delegation (explicit wrapper/reuse, or "instantiates/integrates" without an
    # own-implementation verb): this module needs the pipeline's interface wiring -> keep it.
    if _PLAN_WRAP_RE.search(text) or (_PLAN_INST_RE.search(func) and not own_impl):
        return "pipeline"
    # Self-contained algorithm -> bail to direct (discard the plan).
    if _PLAN_PIPE_RE.search(text):
        return "direct"
    # Unclear: keep the pipeline (sunk-cost plan); only bail when clearly self-contained.
    return "pipeline"


# --------------------------------------------------------------------------- #
# Tier-2 oracle label (golden own_cells + structural guard) -- eval only
# --------------------------------------------------------------------------- #

OWN_CELLS_THRESHOLD = 250


def _family(task: str) -> str:
    if task.startswith("aes"):
        return "aes"
    if task.startswith("sd"):
        return "sdc"
    return "e203_hbirdv2"


def _include_dirs(root: Path, family: str) -> List[Path]:
    if family == "e203_hbirdv2":
        return [root / "e203_hbirdv2" / "e203_defines", root / "e203_hbirdv2" / "config"]
    if family == "sdc":
        return [root / "sdc" / "sd_defines"]
    return []


def _static_counts(vpath: Path) -> Dict[str, int]:
    src = vpath.read_text(errors="ignore")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return {
        "assign": len(re.findall(r"\bassign\b", src)),
        "always": len(re.findall(r"\balways\b", src)),
        "case": len(re.findall(r"\bcase[zx]?\b", src)),
        "insts": len(re.findall(r"^\s*[A-Za-z_]\w*\s+(?:#\s*\([^;]*?\)\s*)?[A-Za-z_]\w*\s*\(", src, re.M)),
    }


def _synth_own_cells(vpath: Path, incdirs: List[Path], extra_files: List[Path], top: str) -> Optional[int]:
    """Synthesize ``top`` and return its local cell count (excluding submodules) via yosys.

    Returns ``None`` if yosys is unavailable or synthesis fails. The local count is parsed
    from the per-module ``stat`` section; a wrapper with no own logic yields 0.
    """
    inc = " ".join(f"-I {shlex.quote(str(d))}" for d in incdirs)
    files = " ".join(shlex.quote(str(p)) for p in [vpath, *extra_files])
    with tempfile.TemporaryDirectory() as tmp:
        stat_path = Path(tmp) / "stat.txt"
        script = f"read_verilog -sv {inc} {files}; synth -top {top}; tee -o {shlex.quote(str(stat_path))} stat"
        try:
            subprocess.run(
                ["yosys", "-p", script],
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if not stat_path.exists():
            return None
        text = stat_path.read_text(errors="ignore")
    section = re.search(r"===\s+" + re.escape(top) + r"\s+===(.*?)(?:\n===|\Z)", text, re.S)
    if not section:
        return None
    cell_match = re.search(r"(\d+)\s+cells", section.group(1))
    return int(cell_match.group(1)) if cell_match else 0


def oracle_label(
    task: str,
    realbench_root: str | Path,
    top: Optional[str] = None,
    dep_files: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Ground-truth A/B class + target flow for ``task`` from the golden RTL (eval only).

    Class ``"A"`` (structural/wrapper/thin -> **pipeline**) if the module is a pure delegation /
    regular-array wrapper or its own synthesized logic is below :data:`OWN_CELLS_THRESHOLD`;
    else class ``"B"`` (self-contained core -> **direct**). The ``flow`` field is the target
    flow via :data:`CLASS_TO_FLOW`.

    ``dep_files`` lets the caller supply the module's dependency sources so hierarchical
    modules synthesize (and report an accurate ``own_cells``); without them such modules
    fall back to the structural heuristic, which still yields the correct A/B label.
    """
    root = Path(realbench_root)
    family = _family(task)
    top = top or task
    vpath = root / family / task / f"{task}.v"
    counts = _static_counts(vpath)

    # G1: pure delegation / regular array / structural wrapper. Decided from the static
    # parse alone, BEFORE synthesis -- this both classifies wrappers correctly and avoids
    # the very slow synth of RAM-backed wrappers (e203_*_ram, e203_srams) whose own logic
    # is ~0 anyway.
    if counts["insts"] >= 2 and counts["always"] == 0 and counts["case"] == 0 and counts["assign"] <= 15:
        return {"label": "A", "flow": CLASS_TO_FLOW["A"], "reason": "wrapper/regular", "own_cells": None, **counts}

    incdirs = _include_dirs(root, family)
    extra = list(dep_files or [])
    if family == "e203_hbirdv2":
        extra += sorted((root / "e203_hbirdv2" / "general").glob("*.v"))
    own = _synth_own_cells(vpath, incdirs, extra, top)

    if own is None:
        # No synth signal available; fall back to the structural heuristic only.
        label, reason = ("B", "core (no-synth, has logic)") if (counts["always"] or counts["case"] or counts["assign"] > 15) else ("A", "thin (no-synth)")
    elif own < OWN_CELLS_THRESHOLD:
        label, reason = "A", f"thin (own_cells={own})"
    else:
        label, reason = "B", f"core (own_cells={own})"

    return {
        "label": label,
        "flow": CLASS_TO_FLOW[label],
        "reason": reason,
        "own_cells": own,
        "insts": counts["insts"],
        "always": counts["always"],
        "case": counts["case"],
        "assign": counts["assign"],
    }
