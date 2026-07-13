# Cleaned accuracy report — ref-wrap cheat removed (120B)

**Date:** 2026-07-13
**Issue:** same direct-prompt `ref_<task>` leak as the 20B run (see
`runs/Full-2T_router_oss20B_sync6_func4/cleaned_accuracy_report.md`).
**Detection:** same regex `^\s*(ref_\w+)\s*(?:#\s*\(|\w+\s*\(|\w+\s*$)` over generated `.sv`.
**Cleaning rule:** wrap samples invalidated (counted as syntax-fail and function-fail).

51 wraps, all in the direct flow; **all 51 compile and all 51 pass** — at 120B the cheat
is a guaranteed pass when attempted.

## Headline (60 tasks × 10 samples)

| Metric | Raw | **Cleaned** |
|---|---|---|
| Tasks solved (pass@10) | 34/60 = 0.567 | **25/60 = 0.417** |
| Per-sample pass (= pass@1) | 247/600 = 0.412 | **196/600 = 0.327** |
| Per-sample syntax | 526/600 = 0.877 | **475/600 = 0.792** |
| pass@5 | 0.526 | **0.394** |

## Per flow

| Flow | n | Raw syn / pass | Cleaned syn / pass |
|---|---|---|---|
| direct (routed) | 281 | 0.915 / 0.367 | **0.733 / 0.185** |
| pipeline (routed) | 319 | 0.843 / 0.451 | 0.843 / 0.451 (unchanged) |

## The 9 tasks that lose "solved" status (solved ONLY via wrap)

e203_exu_alu_lsuagu, e203_exu_alu_muldiv, e203_exu_decode, e203_exu_excp,
e203_ifu_ifetch, e203_ifu_ift2icb, e203_lsu_ctrl, sd_cmd_serial_host, sd_tx_fifo

## Scale comparison after cleaning

| Run | Solved cleaned | pass@1 cleaned | syntax cleaned |
|---|---|---|---|
| Full-2T 20B (6/4) | 21/60 (0.350) | 0.217 | 0.635 |
| Full-2T 120B (6/4) | **25/60 (0.417)** | **0.327** | **0.792** |

The 120B > 20B scale gap survives cleaning (and widens on genuine direct-flow pass:
0.185 vs 0.143), but the raw "same 34 tasks at both scales" coincidence does not —
raw solved-set equality was partly manufactured by wrap passes.

Generated with `clean_wraps.py` (session scratchpad, Jul 13 2026), which reproduces the
20B cleaned report exactly (56 wraps / 55 compile / 54 pass; 21/60; 0.217; 0.635).
