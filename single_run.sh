#!/usr/bin/env bash
set -e

STG_ARGS=()
if [ -n "${STG_GOLDEN_FILE:-}" ]; then
  STG_ARGS+=(--stg-golden-file "$STG_GOLDEN_FILE")
elif [ -n "${STG_GOLDEN:-}" ]; then
  STG_ARGS+=(--stg-golden "$STG_GOLDEN")
fi
if [ -n "${STG_TYPE:-}" ]; then
  STG_ARGS+=(--stg-type "$STG_TYPE")
fi
if [ -n "${STG_MODULE:-}" ]; then
  STG_ARGS+=(--stg-module "$STG_MODULE")
fi
if [ -n "${STG_GOLDEN_MODULE:-}" ]; then
  STG_ARGS+=(--stg-golden-module "$STG_GOLDEN_MODULE")
fi

python3 -m rtl_agent.cli run \
  --index indexes/rtl_hash \
  --prompt 'The provided Verilog code defines a hardware module that takes multiple control signals and input values (both unsigned and signed in various bit widths) and produces a 128-bit output `y`. The output is constructed from eight 16-bit values (`y0` to `y7`) that are generated using complex combinations of arithmetic operations, bitwise operations, and conditional expressions based on the control signals. Each of the `y` values incorporates manipulations of the signed and unsigned input signals, applying shifts, logical comparisons, and negations, effectively implementing a custom logic operation based on the input conditions specified by `ctrl`.' \
  --workspace-root runs/agent_workspace \
  --max-steps 8 \
  --json-report runs/agent_latest_report.json \
  "${STG_ARGS[@]}"
