#!/usr/bin/env python3
"""Extract per-sample LLM request logs from a run's records.jsonl.

Reads the top-level records.jsonl of a run directory (which aggregates the
direct and pipeline flows) and writes performance.jsonl next to it: one line
per sample, keyed by "{task}_sample{sample}", carrying the whole
llm_request_log list of that sample.

Usage::

    python3 scripts/extract_performance_log.py \
        runs/Full-2T_router_oss20B_syn6_func4_s10_t0_performance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs/Full-2T_router_oss20B_syn6_func4_s10_t0_performance",
        help="Run directory containing records.jsonl",
    )
    parser.add_argument(
        "--output",
        help="Output path (default: <run_dir>/performance.jsonl)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found under {run_dir}")
    output_path = Path(args.output) if args.output else run_dir / "performance.jsonl"

    seen: set[str] = set()
    written = 0
    with records_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            each_req = record.get("llm_request_log", [])
            for i, req in enumerate(each_req):
                sample_id = f"{record['task']}_sample{record['sample']}_{i}"
                if sample_id in seen:
                    raise ValueError(f"duplicate sample id: {sample_id}")
                seen.add(sample_id)
                dst.write(
                    json.dumps(
                        {"id": sample_id, "start_time": req.get("start_time"), "start_epoch": req.get("start_epoch"),"latency_s": req.get("latency_s"),"prompt_tokens": req.get("prompt_tokens"),"completion_tokens": req.get("completion_tokens")},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

    print(f"wrote {written} samples to {output_path}")


if __name__ == "__main__":
    main()
