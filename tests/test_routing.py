"""Tests for rag_rtl.routing and the golden A/B label table.

* Pure unit tests (deciders, keyword features, LLM-feature parsing) always run.
* oracle_label / committed-label tests are skipped when yosys or the RealBench dataset
  is unavailable, so CI without the dataset still passes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rag_rtl import routing

REPO_ROOT = Path(__file__).resolve().parents[1]
REALBENCH_ROOT = Path("/home/kai/eval_dt/real_bench")
LABELS_PATH = REPO_ROOT / "routing" / "route_labels.json"

# The two sets the routing design was validated against this session.
GOLD = {
    "e203_dtcm_ram": "A", "e203_exu_alu_csrctrl": "A", "e203_exu_alu_rglr": "A", "e203_exu_regfile": "A",
    "e203_ifu": "A", "e203_ifu_minidec": "A", "e203_itcm_ram": "A", "e203_srams": "A",
    "aes_cipher_top": "B", "aes_inv_cipher_top": "B", "e203_exu_branchslv": "B", "e203_exu_alu_lsuagu": "B",
    "e203_exu_decode": "B", "sd_cmd_master": "B", "sd_cmd_serial_host": "B", "sd_rx_fifo": "B", "sd_tx_fifo": "B",
}
WANT = {"A": "direct", "B": "pipeline"}


# --------------------------------------------------------------------------- #
# Tier-0 deciders (pure)
# --------------------------------------------------------------------------- #


def test_decide_pre_algorithm_to_direct():
    # Self-contained algorithm -> direct; a B-flag overrides the delegation markers.
    feats = routing.RouteFeatures(has_fsm=True, delegates_to_submodules=True)
    assert routing.decide_pre(feats) == "direct"


def test_decide_pre_wrapper_to_pipeline():
    feats = routing.RouteFeatures(delegates_to_submodules=True)
    assert routing.decide_pre(feats) == "pipeline"


def test_decide_pre_thin_and_memory_to_pipeline():
    assert routing.decide_pre(routing.RouteFeatures(thin_control=True)) == "pipeline"
    assert routing.decide_pre(routing.RouteFeatures(is_memory=True)) == "pipeline"


def test_decide_pre_uncertain_when_no_signal():
    assert routing.decide_pre(routing.RouteFeatures()) == "uncertain"


def test_decide_pre_low_confidence_llm_is_uncertain():
    feats = routing.RouteFeatures(has_algorithm=True, confidence=0.2, source="llm")
    assert routing.decide_pre(feats, confidence_tau=0.5) == "uncertain"


def test_class_to_flow_polarity():
    # Empirical: wrapper class A -> pipeline, self-contained class B -> direct.
    assert routing.CLASS_TO_FLOW == {"A": "pipeline", "B": "direct"}


# --------------------------------------------------------------------------- #
# Tier-1 decide_plan (pure; synthetic plans mirroring real planner output)
# --------------------------------------------------------------------------- #


def _plan(functionality: str, area: str = "") -> dict:
    return {"structured_plan": {"requirements": {"functionality": functionality, "ppa_constraints": {"area": area}}}}


def test_decide_plan_wrapper_stays_pipeline():
    # The minidec case: explicitly a wrapper (mentions "decoded ... instruction") -> keep pipeline.
    plan = _plan("Combinational wrapper that instantiates e203_exu_decode and passes the decoded instruction info")
    assert routing.decide_plan(plan) == "pipeline"


def test_decide_plan_reuse_existing_stays_pipeline():
    assert routing.decide_plan(_plan("DTCM RAM module", area="minimal, reuse existing sirv_gnrl_ram")) == "pipeline"


def test_decide_plan_implement_bails_to_direct():
    assert routing.decide_plan(_plan("Implement a serial host for SD/MMC with CRC and transmission")) == "direct"
    assert routing.decide_plan(_plan("AES-128 core implementing 10 rounds with SubBytes and MixColumns")) == "direct"


def test_decide_plan_decode_instruction_bails_to_direct():
    assert routing.decide_plan(_plan("Decode 32-bit and 16-bit RISC-V instructions into control signals")) == "direct"


def test_decide_plan_defaults_to_pipeline_when_unclear():
    # Unclear plan -> keep the pipeline (sunk-cost plan); register-file is a Set-A/pipeline task.
    assert routing.decide_plan(_plan("A register file with two read ports and one write port")) == "pipeline"


def test_decide_plan_accepts_raw_or_wrapped():
    raw = {"requirements": {"functionality": "Encapsulates two sub-modules"}}
    assert routing.decide_plan(raw) == "pipeline"
    assert routing.decide_plan({"structured_plan": raw}) == "pipeline"


# --------------------------------------------------------------------------- #
# LLM feature extractor (fake client; no server)
# --------------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, payload: dict):
        self._payload = payload

    def chat(self, messages, temperature=0.0, max_tokens=1024):
        return {"content": "here is the analysis\n" + json.dumps(self._payload) + "\ntrailing text"}


def test_extract_features_llm_parses_json(tmp_path):
    payload = {"delegates_to_submodules": True, "has_fsm": False, "est_state_bits": 0, "confidence": 0.9}
    feats = routing.extract_features_llm("spec text", _FakeClient(payload), cache_dir=tmp_path)
    assert feats.source == "llm" and feats.delegates_to_submodules is True and feats.confidence == 0.9
    assert routing.decide_pre(feats) == "pipeline"  # wrapper -> pipeline
    # cached file round-trips
    cached = routing.extract_features_llm("spec text", _FakeClient({"confidence": 0.0}), cache_dir=tmp_path)
    assert cached.delegates_to_submodules is True  # served from cache, not the second payload


# --------------------------------------------------------------------------- #
# Oracle labels (require yosys + dataset)
# --------------------------------------------------------------------------- #

_HAVE_YOSYS = shutil.which("yosys") is not None
_HAVE_DATASET = REALBENCH_ROOT.exists()
_oracle_skip = pytest.mark.skipif(not (_HAVE_YOSYS and _HAVE_DATASET), reason="needs yosys + RealBench dataset")


@_oracle_skip
@pytest.mark.parametrize("task,label", [
    ("e203_exu_alu_csrctrl", "A"),  # thin combinational control, own_cells=122
    ("e203_exu_alu_rglr", "A"),     # thin routing, own_cells=67
    ("e203_exu_branchslv", "B"),    # dense logic, own_cells=699
    ("e203_exu_decode", "B"),       # instruction decoder, own_cells=824
    ("sd_tx_fifo", "B"),            # async FIFO, own_cells=779
    ("e203_srams", "A"),            # pure wrapper (caught by structural guard, no synth)
    ("e203_exu_regfile", "A"),      # regular DFF array (structural guard)
])
def test_oracle_label_matches_gold(task, label):
    result = routing.oracle_label(task, REALBENCH_ROOT)
    assert result["label"] == label, result
    assert result["flow"] == routing.CLASS_TO_FLOW[label]  # A->pipeline, B->direct


@pytest.mark.skipif(not LABELS_PATH.exists(), reason="route_labels.json not generated")
def test_committed_labels_match_gold():
    labels = json.loads(LABELS_PATH.read_text())["labels"]
    mismatches = {t: (g, labels.get(t, {}).get("label")) for t, g in GOLD.items() if labels.get(t, {}).get("label") != g}
    assert not mismatches, f"oracle label mismatches: {mismatches}"
