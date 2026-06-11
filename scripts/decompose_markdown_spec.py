#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


HEADING_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<marks>#{1,6})(?:[ \t]+(?P<title>.*?))?[ \t]*$")
FENCE_RE = re.compile(r"^[ \t]{0,3}(?P<marker>`{3,}|~{3,})")
SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class MarkdownSection:
    index: int
    level: int
    title: str
    slug: str
    content: str


def decompose_markdown(
    text: str,
    min_level: int = 1,
    max_level: int = 6,
    include_preamble: bool = True,
    keep_heading: bool = True,
) -> List[MarkdownSection]:
    """Split Markdown text into sections at ATX headings outside fenced code blocks."""
    if min_level < 1 or max_level > 6 or min_level > max_level:
        raise ValueError("heading levels must satisfy 1 <= min_level <= max_level <= 6")

    lines = text.splitlines(keepends=True)
    sections: List[MarkdownSection] = []
    current_lines: List[str] = []
    current_title = "Preamble"
    current_level = 0
    current_slug = "preamble"
    saw_heading = False
    fence_marker: Optional[str] = None

    def flush() -> None:
        nonlocal current_lines
        content = "".join(current_lines)
        if not content.strip():
            current_lines = []
            return
        if current_level == 0 and not include_preamble:
            current_lines = []
            return
        sections.append(
            MarkdownSection(
                index=len(sections),
                level=current_level,
                title=current_title,
                slug=current_slug,
                content=content,
            )
        )
        current_lines = []

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group("marker")
            if fence_marker is None:
                fence_marker = marker
            elif marker.startswith(fence_marker[0]) and len(marker) >= len(fence_marker):
                fence_marker = None

        heading_match = HEADING_RE.match(line) if fence_marker is None else None
        if heading_match:
            marks = heading_match.group("marks")
            level = len(marks)
            if min_level <= level <= max_level:
                flush()
                saw_heading = True
                raw_title = (heading_match.group("title") or "").strip()
                title = _clean_heading_title(raw_title) or f"Untitled {len(sections) + 1}"
                current_title = title
                current_level = level
                current_slug = slugify(title)
                current_lines = [line] if keep_heading else []
                continue

        if not saw_heading and current_level == 0:
            current_title = "Preamble"
            current_slug = "preamble"
        current_lines.append(line)

    flush()
    return _deduplicate_slugs(sections)


def write_section_files(sections: Iterable[MarkdownSection], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for section in sections:
        path = output_dir / f"{section.index:03d}-{section.slug}.md"
        path.write_text(section.content, encoding="utf-8")
        written.append(path)
    return written


def slugify(value: str) -> str:
    slug = SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "untitled"


def _clean_heading_title(title: str) -> str:
    return re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()


def _deduplicate_slugs(sections: List[MarkdownSection]) -> List[MarkdownSection]:
    counts: dict[str, int] = {}
    deduped: List[MarkdownSection] = []
    for section in sections:
        count = counts.get(section.slug, 0) + 1
        counts[section.slug] = count
        slug = section.slug if count == 1 else f"{section.slug}-{count}"
        deduped.append(
            MarkdownSection(
                index=section.index,
                level=section.level,
                title=section.title,
                slug=slug,
                content=section.content,
            )
        )
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose a Markdown spec into heading-based sections."
    )
    parser.add_argument("input", help="Input Markdown file.")
    parser.add_argument(
        "--output-dir",
        default="decomposed_spec",
        help="Directory for section files when using --format files. Default: decomposed_spec.",
    )
    parser.add_argument(
        "--format",
        choices=["files", "json", "jsonl"],
        default="files",
        help="Output format. Default: files.",
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=1,
        help="Lowest heading level to split on. Default: 1.",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=6,
        help="Highest heading level to split on. Default: 6.",
    )
    parser.add_argument(
        "--drop-heading",
        action="store_true",
        help="Do not include the heading line in each emitted section.",
    )
    parser.add_argument(
        "--skip-preamble",
        action="store_true",
        help="Drop text before the first matching heading.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    sections = decompose_markdown(
        input_path.read_text(encoding="utf-8"),
        min_level=args.min_level,
        max_level=args.max_level,
        include_preamble=not args.skip_preamble,
        keep_heading=not args.drop_heading,
    )

    if args.format == "files":
        written = write_section_files(sections, Path(args.output_dir))
        print(f"Wrote {len(written)} sections to {args.output_dir}")
        return 0

    if args.format == "json":
        print(json.dumps([asdict(section) for section in sections], ensure_ascii=False, indent=2))
        return 0

    for section in sections:
        print(json.dumps(asdict(section), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
