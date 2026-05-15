#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, TextIO


DEFAULT_PREFIX = (
    "Please solve the following Verilog coding problem. Think first INTERNALLY about "
    "how to arrive at the correct solution. Then, output ONLY the Verilog code you "
    "designed in this format: <answer>\n"
    "```verilog\n"
    "...\n"
    "```\n"
    "</answer>. No explanations, comments, or additional text are allowed outside of "
    "the specified formatting.\n\n"
    "### Verilog Coding Problem\n\n"
)


def clean_problem_text(
    text: str,
    prefix: str = DEFAULT_PREFIX,
    strip_wrapper_quotes: bool = False,
) -> str:
    if text.startswith(prefix):
        text = text[len(prefix):]
    if strip_wrapper_quotes:
        text = _strip_triple_quote_wrapper(text)
    return text


def clean_problem_value(
    value: Any,
    prefix: str = DEFAULT_PREFIX,
    strip_wrapper_quotes: bool = False,
) -> Tuple[Any, bool]:
    if isinstance(value, str):
        cleaned = clean_problem_text(value, prefix=prefix, strip_wrapper_quotes=strip_wrapper_quotes)
        return cleaned, cleaned != value
    if isinstance(value, list):
        changed = False
        cleaned_items = []
        for item in value:
            if isinstance(item, dict):
                cleaned_item = dict(item)
                content = cleaned_item.get("content")
                if isinstance(content, str):
                    cleaned_content = clean_problem_text(
                        content,
                        prefix=prefix,
                        strip_wrapper_quotes=strip_wrapper_quotes,
                    )
                    if cleaned_content != content:
                        cleaned_item["content"] = cleaned_content
                        changed = True
                cleaned_items.append(cleaned_item)
            else:
                cleaned_items.append(item)
        return cleaned_items, changed
    return value, False


def clean_record(
    record: Dict[str, Any],
    field: str,
    prefix: str,
    strip_wrapper_quotes: bool,
) -> bool:
    if field == "auto":
        changed = False
        for candidate in ("problem", "prompt"):
            if candidate in record:
                changed = clean_record(record, candidate, prefix, strip_wrapper_quotes) or changed
        return changed

    value = record.get(field)
    cleaned, changed = clean_problem_value(value, prefix=prefix, strip_wrapper_quotes=strip_wrapper_quotes)
    if not changed:
        return False
    record[field] = cleaned
    return True


def clean_jsonl(
    input_handle: TextIO,
    output_handle: TextIO,
    field: str,
    prefix: str,
    strip_wrapper_quotes: bool,
) -> Dict[str, int]:
    stats = {"records": 0, "changed": 0}
    for line_no, line in enumerate(input_handle, start=1):
        if not line.strip():
            output_handle.write(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_no} is not a JSON object")
        stats["records"] += 1
        if clean_record(record, field, prefix, strip_wrapper_quotes):
            stats["changed"] += 1
        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stats


def _strip_triple_quote_wrapper(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6:
        return stripped[3:-3].strip()
    return text


def _load_prefix(args: argparse.Namespace) -> str:
    if args.prefix_file:
        return Path(args.prefix_file).read_text(encoding="utf-8")
    if args.prefix is not None:
        return args.prefix
    return DEFAULT_PREFIX


def _write_in_place(
    input_path: Path,
    field: str,
    prefix: str,
    strip_wrapper_quotes: bool,
) -> Dict[str, int]:
    with input_path.open("r", encoding="utf-8") as input_handle:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(input_path.parent),
            delete=False,
            newline="",
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            try:
                stats = clean_jsonl(input_handle, temp_handle, field, prefix, strip_wrapper_quotes)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
    os.replace(temp_path, input_path)
    return stats


def _write_to_output(
    input_path: Path,
    output_path: Path,
    field: str,
    prefix: str,
    strip_wrapper_quotes: bool,
) -> Dict[str, int]:
    with input_path.open("r", encoding="utf-8") as input_handle:
        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            return clean_jsonl(input_handle, output_handle, field, prefix, strip_wrapper_quotes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove the standard SiliconMind problem wrapper prefix from JSONL records."
    )
    parser.add_argument("input", help="Input JSONL file, for example merged.jsonl")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output JSONL file. Omit when using --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input JSONL atomically instead of writing a separate output file.",
    )
    parser.add_argument(
        "--field",
        default="auto",
        help="Field to clean. Use auto, problem, or prompt. Default: auto.",
    )
    parser.add_argument(
        "--prefix",
        help="Custom prefix to remove. Defaults to the known Verilog coding problem wrapper.",
    )
    parser.add_argument(
        "--prefix-file",
        help="Read the prefix to remove from this UTF-8 text file.",
    )
    parser.add_argument(
        "--strip-wrapper-quotes",
        action="store_true",
        help='Also remove surrounding triple quotes after the prefix, if the remaining text is wrapped in """.',
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.in_place and args.output:
        parser.error("output path cannot be used with --in-place")
    if not args.in_place and not args.output:
        parser.error("provide an output path, or use --in-place")
    if args.prefix and args.prefix_file:
        parser.error("--prefix and --prefix-file are mutually exclusive")

    input_path = Path(args.input)
    prefix = _load_prefix(args)
    if args.in_place:
        stats = _write_in_place(input_path, args.field, prefix, args.strip_wrapper_quotes)
        output_label = str(input_path)
    else:
        output_path = Path(args.output)
        stats = _write_to_output(input_path, output_path, args.field, prefix, args.strip_wrapper_quotes)
        output_label = str(output_path)

    print(f"Wrote {output_label}: {stats['changed']} / {stats['records']} records changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
