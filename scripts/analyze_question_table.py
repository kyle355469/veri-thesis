#!/usr/bin/env python3
"""Analyze pass/fail reason ratios from a benchmark question_table.md file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_TABLE = Path("runs/agentic_benchmark_matrix/question_table.md")
PASS_REASON = "."

REASON_LABELS = {
    ".": "passed",
    "G": "generation failed",
    "S": "syntax error",
    "C": "compile/build error",
    "R": "wrong result or mismatch",
    "T": "timeout",
    "X": "simulation exited nonzero",
    "I": "infrastructure error",
    "?": "unknown failure",
    "0": "zero-sized numeric constant",
    "c": "clock binding error",
    "e": "explicit cast required",
    "m": "unknown module or binding",
    "n": "empty sensitivity process",
    "r": "reset edge issue",
    "w": "wire/reg declaration issue",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a benchmark question_table.md and accumulate passfail_counts "
            "into per-reason counts and ratios."
        )
    )
    parser.add_argument("question_table", nargs="?", default=str(DEFAULT_TABLE))
    parser.add_argument(
        "--include-success",
        action="store_true",
        help="Include the '.' pass reason in the output table.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--group-by",
        choices=["benchmark", "mode", "category"],
        help="Also break ratios down by one question_table column.",
    )
    parser.add_argument(
        "--by-mode",
        action="store_true",
        help="Shortcut for --group-by mode.",
    )
    return parser


def read_markdown_table(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    header: Optional[List[str]] = None

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        cells = split_markdown_row(line)
        if not cells:
            continue
        if is_separator_row(cells):
            continue
        if header is None:
            header = cells
            continue
        if len(cells) != len(header):
            raise ValueError(
                f"{path}:{line_number}: expected {len(header)} columns from header, got {len(cells)}"
            )
        rows.append(dict(zip(header, cells)))

    if header is None:
        raise ValueError(f"{path}: no markdown table header found")
    if "passfail_counts" not in header:
        raise ValueError(f"{path}: missing required passfail_counts column")
    return rows


def split_markdown_row(line: str) -> List[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]

    cells: List[str] = []
    current: List[str] = []
    escaped = False
    for char in text:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def is_separator_row(cells: Sequence[str]) -> bool:
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def parse_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token in text.split():
        if ":" not in token:
            continue
        reason, count_text = token.rsplit(":", 1)
        if not reason:
            continue
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"invalid passfail count token: {token!r}") from exc
        counts[reason] += count
    return counts


def accumulate_counts(rows: Iterable[Dict[str, str]], group_by: Optional[str] = None) -> Dict[str, Counter[str]]:
    grouped: Dict[str, Counter[str]] = {}
    for row in rows:
        key = row.get(group_by, "") if group_by else "all"
        grouped.setdefault(key, Counter()).update(parse_counts(row.get("passfail_counts", "")))
    return grouped


def build_analysis_rows(
    grouped_counts: Dict[str, Counter[str]],
    *,
    include_success: bool = False,
    group_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, counts in sorted(grouped_counts.items()):
        total = sum(counts.values())
        error_total = sum(count for reason, count in counts.items() if reason != PASS_REASON)
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if reason == PASS_REASON and not include_success:
                continue
            row: Dict[str, Any] = {
                "reason": reason,
                "label": REASON_LABELS.get(reason, ""),
                "count": count,
                "ratio_of_errors": None if reason == PASS_REASON else safe_ratio(count, error_total),
                "ratio_of_all": safe_ratio(count, total),
                "total_errors": error_total,
                "total_samples": total,
            }
            if group_by:
                row[group_by] = group
            rows.append(row)
    return rows


def safe_ratio(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def markdown_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}"
    elif value is None:
        text = ""
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def csv_text(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_value(row.get(field)) for field in fields})
    return output.getvalue()


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


def output_text(rows: Sequence[Dict[str, Any]], fields: Sequence[str], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(list(rows), indent=2) + "\n"
    if output_format == "csv":
        return csv_text(rows, fields)
    return markdown_table(rows, fields)


def fields_for(group_by: Optional[str]) -> List[str]:
    fields = ["reason", "label", "count", "ratio_of_errors", "ratio_of_all", "total_errors", "total_samples"]
    if group_by:
        fields.insert(0, group_by)
    return fields


def resolve_group_by(cli: argparse.Namespace) -> Optional[str]:
    if cli.by_mode:
        if cli.group_by and cli.group_by != "mode":
            raise SystemExit("--by-mode cannot be combined with --group-by other than mode")
        return "mode"
    return cli.group_by


def main() -> None:
    cli = build_parser().parse_args()
    group_by = resolve_group_by(cli)
    table_path = Path(cli.question_table)
    rows = read_markdown_table(table_path)
    grouped_counts = accumulate_counts(rows, group_by)
    analysis_rows = build_analysis_rows(
        grouped_counts,
        include_success=cli.include_success,
        group_by=group_by,
    )
    text = output_text(analysis_rows, fields_for(group_by), cli.format)
    if cli.output:
        Path(cli.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
