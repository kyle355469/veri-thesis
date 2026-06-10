from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.embeddings import make_embedder

GENERATE_LOG_RE = re.compile(r"^(?P<problem>.+)_sample(?P<sample>\d+)-sv-generate\.log$")


@dataclass(frozen=True)
class RtlDoc:
    doc_id: str
    problem: str
    solution: str

    def text_for_similarity(self, field: str) -> str:
        if field == "problem":
            return self.problem
        if field == "both":
            return f"{self.problem}\n\n{self.solution}".strip()
        return self.solution


@dataclass
class Attempt:
    mode: str
    problem: str
    sample: int
    generation_log_path: Path
    compile_log_path: Optional[Path]
    reference_path: Optional[Path]
    generated_code_path: Optional[Path]
    retrieved_doc_ids: List[str]
    passed: Optional[bool]
    passfail: Optional[str]
    mismatches: Optional[int]
    syntax_passed: Optional[bool]
    lint_passed: Optional[bool]
    rag_generation_passed: Optional[bool]
    cache_source: Optional[str]
    repair_attempts: Optional[int]


@dataclass(frozen=True)
class ScoredAttempt:
    attempt: Attempt
    best_doc_id: Optional[str]
    best_score: Optional[float]
    scored_doc_count: int
    missing_doc_count: int


@dataclass(frozen=True)
class ModeProblemSummary:
    mode: str
    problem: str
    attempts: List[ScoredAttempt]
    representative_doc_ids: List[str]
    retrieval_set_count: int
    best_doc_id: Optional[str]
    best_score: Optional[float]
    scored_doc_count: int
    missing_doc_count: int

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.attempts if item.attempt.passed is True)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class ProblemSummary:
    problem: str
    modes: Dict[str, ModeProblemSummary]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an HTML report showing benchmark attempt success and the highest "
            "similarity between retrieved RTL documents and the Verilog-Eval golden answer."
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        help=(
            "Benchmark mode directory to include. May be repeated. Defaults to existing "
            "runs/benchmark_matrix_sm/verilog-eval/rag and model directories."
        ),
    )
    parser.add_argument("--index", default="indexes/rtl_hash", help="Index directory containing documents.jsonl.")
    parser.add_argument(
        "--output",
        default="runs/benchmark_matrix_sm/verilog-eval/retrieval_similarity_report.html",
        help="Output HTML report path.",
    )
    parser.add_argument("--embedder", default="hash", help="Embedder name accepted by rag_rtl.embeddings.")
    parser.add_argument(
        "--doc-field",
        choices=["solution", "problem", "both"],
        default="solution",
        help="Which retrieved document text to compare against each golden answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of attempts to include after sorting. Useful for quick checks.",
    )
    return parser


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def default_run_dirs() -> List[Path]:
    base = REPO_ROOT / "runs" / "benchmark_matrix_sm" / "verilog-eval"
    return [path for path in (base / "rag", base / "model") if path.exists()]


def load_documents(index_dir: Path) -> Dict[str, RtlDoc]:
    docs_path = index_dir / "documents.jsonl"
    documents: Dict[str, RtlDoc] = {}
    with docs_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{docs_path}:{line_number}: invalid JSON: {exc}") from exc
            doc_id = str(payload.get("doc_id", "")).strip()
            if not doc_id:
                continue
            documents[doc_id] = RtlDoc(
                doc_id=doc_id,
                problem=str(payload.get("problem") or ""),
                solution=str(payload.get("solution") or ""),
            )
    return documents


def parse_key_value_log(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        values[key.strip()] = value.strip()
    return values


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_retrieved_doc_ids(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def mode_from_run_dir(run_dir: Path) -> str:
    if run_dir.name:
        return run_dir.name
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        mode = payload.get("pipeline") or payload.get("mode")
        if mode:
            return str(mode)
    return run_dir.name


def path_from_record(record: Dict[str, Any], key: str) -> Optional[Path]:
    value = record.get(key)
    if not value:
        return None
    return resolve_repo_path(str(value))


def load_attempts_from_summary(run_dir: Path, mode: str) -> List[Attempt]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    attempts: List[Attempt] = []
    for record in records:
        generation_log = path_from_record(record, "generation_log_path")
        if generation_log is None:
            continue
        generation_values = parse_key_value_log(generation_log)
        retrieved_doc_ids = parse_retrieved_doc_ids(generation_values.get("retrieved_doc_ids"))
        if not retrieved_doc_ids:
            retrieved_doc_ids = parse_retrieved_doc_ids(record.get("retrieved_doc_ids"))
        attempts.append(
            Attempt(
                mode=mode,
                problem=str(record.get("problem") or generation_log.parent.name),
                sample=int(record.get("sample") or 0),
                generation_log_path=generation_log,
                compile_log_path=path_from_record(record, "compile_log_path"),
                reference_path=path_from_record(record, "reference_path"),
                generated_code_path=path_from_record(record, "generated_code_path"),
                retrieved_doc_ids=retrieved_doc_ids,
                passed=parse_bool(record.get("passed")),
                passfail=str(record.get("passfail")) if record.get("passfail") is not None else None,
                mismatches=parse_int(record.get("mismatches")),
                syntax_passed=parse_bool(record.get("syntax_passed")),
                lint_passed=parse_bool(record.get("lint_passed")),
                rag_generation_passed=parse_bool(record.get("rag_generation_passed")),
                cache_source=str(record.get("cache_source")) if record.get("cache_source") is not None else None,
                repair_attempts=parse_int(record.get("repair_attempts")),
            )
        )
    return attempts


def infer_reference_path(problem: str, generation_values: Dict[str, str], run_dir: Path) -> Optional[Path]:
    prompt = generation_values.get("prompt")
    if prompt:
        prompt_path = resolve_repo_path(prompt)
        candidate = prompt_path.with_name(f"{problem}_ref.sv")
        if candidate.exists():
            return candidate

    local_candidate = run_dir / "_VerilogEval-v2-NTU" / "dataset_spec-to-rtl" / f"{problem}_ref.sv"
    if local_candidate.exists():
        return local_candidate
    return None


def load_attempts_from_logs(run_dir: Path, mode: str) -> List[Attempt]:
    attempts: List[Attempt] = []
    for generation_log in sorted(run_dir.glob("*/*-sv-generate.log")):
        match = GENERATE_LOG_RE.match(generation_log.name)
        if not match:
            continue
        problem = match.group("problem")
        sample = int(match.group("sample"))
        values = parse_key_value_log(generation_log)
        compile_log = generation_log.with_name(generation_log.name.replace("-sv-generate.log", "-sv-iv-test.log"))
        compile_values = parse_key_value_log(compile_log)
        passfail = compile_values.get("passfail")
        attempts.append(
            Attempt(
                mode=mode,
                problem=values.get("problem") or problem,
                sample=int(values.get("sample") or sample),
                generation_log_path=generation_log,
                compile_log_path=compile_log if compile_log.exists() else None,
                reference_path=infer_reference_path(values.get("problem") or problem, values, run_dir),
                generated_code_path=generation_log.with_name(generation_log.name.replace("-sv-generate.log", ".sv")),
                retrieved_doc_ids=parse_retrieved_doc_ids(values.get("retrieved_doc_ids")),
                passed=(passfail == ".") if passfail is not None else parse_bool(values.get("verification_passed")),
                passfail=passfail,
                mismatches=None,
                syntax_passed=parse_bool(values.get("syntax_passed")),
                lint_passed=parse_bool(values.get("lint_passed")),
                rag_generation_passed=parse_bool(values.get("verification_passed")),
                cache_source=values.get("cache_source"),
                repair_attempts=parse_int(values.get("repair_attempts")),
            )
        )
    return attempts


def load_attempts(run_dir: Path) -> List[Attempt]:
    mode = mode_from_run_dir(run_dir)
    attempts = load_attempts_from_summary(run_dir, mode)
    if attempts:
        return attempts
    return load_attempts_from_logs(run_dir, mode)


def sort_attempts(attempts: Iterable[Attempt]) -> List[Attempt]:
    return sorted(attempts, key=lambda item: (item.mode, item.problem, item.sample))


def unique_ordered(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def read_reference_texts(attempts: Sequence[Attempt]) -> Dict[Path, str]:
    texts: Dict[Path, str] = {}
    for attempt in attempts:
        if attempt.reference_path is None or attempt.reference_path in texts:
            continue
        try:
            texts[attempt.reference_path] = attempt.reference_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            texts[attempt.reference_path] = ""
    return texts


def encode_map(embedder_name: str, texts_by_key: Dict[Any, str]) -> Dict[Any, np.ndarray]:
    if not texts_by_key:
        return {}
    embedder = make_embedder(embedder_name)
    keys = list(texts_by_key)
    vectors = embedder.encode(texts_by_key[key] for key in keys)
    return {key: np.asarray(vector, dtype=np.float32) for key, vector in zip(keys, vectors)}


def score_attempts(
    attempts: Sequence[Attempt],
    documents: Dict[str, RtlDoc],
    embedder_name: str,
    doc_field: str,
) -> List[ScoredAttempt]:
    reference_texts = read_reference_texts(attempts)
    reference_vectors = encode_map(embedder_name, reference_texts)
    retrieved_ids = unique_ordered(doc_id for attempt in attempts for doc_id in attempt.retrieved_doc_ids)
    doc_texts = {
        doc_id: documents[doc_id].text_for_similarity(doc_field)
        for doc_id in retrieved_ids
        if doc_id in documents
    }
    doc_vectors = encode_map(embedder_name, doc_texts)

    scored_attempts: List[ScoredAttempt] = []
    for attempt in attempts:
        ref_vector = reference_vectors.get(attempt.reference_path)
        best_doc_id: Optional[str] = None
        best_score: Optional[float] = None
        scored_count = 0
        missing_count = 0
        for doc_id in attempt.retrieved_doc_ids:
            doc_vector = doc_vectors.get(doc_id)
            if doc_vector is None or ref_vector is None:
                missing_count += 1
                continue
            score = float(doc_vector @ ref_vector)
            scored_count += 1
            if best_score is None or score > best_score:
                best_doc_id = doc_id
                best_score = score
        scored_attempts.append(
            ScoredAttempt(
                attempt=attempt,
                best_doc_id=best_doc_id,
                best_score=best_score,
                scored_doc_count=scored_count,
                missing_doc_count=missing_count,
            )
        )
    return scored_attempts


def display_bool(value: Optional[bool]) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def score_text(score: Optional[float]) -> str:
    if score is None:
        return "-"
    return f"{score:.3f}"


def css_score_width(score: Optional[float]) -> str:
    if score is None:
        return "0%"
    bounded = max(0.0, min(1.0, score))
    return f"{bounded * 100:.1f}%"


def href_for(path: Optional[Path], output_path: Path) -> str:
    if path is None:
        return ""
    try:
        return html.escape(path.relative_to(output_path.parent).as_posix())
    except ValueError:
        return html.escape(path.as_posix())


def link_for(path: Optional[Path], label: str, output_path: Path) -> str:
    if path is None:
        return ""
    href = href_for(path, output_path)
    return f'<a href="{href}">{html.escape(label)}</a>'


def mode_summaries(scored_attempts: Sequence[ScoredAttempt]) -> List[Tuple[str, int, int, float, Optional[float]]]:
    by_mode: Dict[str, List[ScoredAttempt]] = {}
    for item in scored_attempts:
        by_mode.setdefault(item.attempt.mode, []).append(item)

    summaries: List[Tuple[str, int, int, float, Optional[float]]] = []
    for mode, rows in sorted(by_mode.items()):
        total = len(rows)
        passed = sum(1 for row in rows if row.attempt.passed is True)
        scores = [row.best_score for row in rows if row.best_score is not None]
        average = statistics.fmean(scores) if scores else None
        summaries.append((mode, total, passed, passed / total if total else 0.0, average))
    return summaries


def render_attempt_html(scored_attempts: Sequence[ScoredAttempt], output_path: Path, index_dir: Path, doc_field: str) -> str:
    total = max(len(scored_attempts), 1)  # avoid division by zero in summary cards when there are no attempts
    passed = sum(1 for item in scored_attempts if item.attempt.passed is True)
    with_score = sum(1 for item in scored_attempts if item.best_score is not None)
    missing_docs = sum(item.missing_doc_count for item in scored_attempts)
    mode_cards = []
    for mode, mode_total, mode_passed, rate, avg_score in mode_summaries(scored_attempts):
        mode_cards.append(
            f"""
            <section class="metric">
              <div class="metric-label">{html.escape(mode)}</div>
              <div class="metric-value">{mode_passed}/{mode_total}</div>
              <div class="metric-note">pass {rate:.1%} · avg best {score_text(avg_score)}</div>
            </section>
            """
        )

    rows = []
    for item in scored_attempts:
        attempt = item.attempt
        success_class = "pass" if attempt.passed is True else "fail" if attempt.passed is False else "unknown"
        doc_list = ", ".join(attempt.retrieved_doc_ids[:8])
        if len(attempt.retrieved_doc_ids) > 8:
            doc_list += f", +{len(attempt.retrieved_doc_ids) - 8} more"
        rows.append(
            f"""
            <tr class="{success_class}">
              <td>{html.escape(attempt.mode)}</td>
              <td class="problem">{html.escape(attempt.problem)}</td>
              <td>{attempt.sample:02d}</td>
              <td><span class="pill {success_class}">{display_bool(attempt.passed)}</span></td>
              <td>{html.escape(attempt.passfail or "-")}</td>
              <td>{display_bool(attempt.rag_generation_passed)}</td>
              <td>{display_bool(attempt.syntax_passed)}</td>
              <td>{display_bool(attempt.lint_passed)}</td>
              <td>{html.escape(str(attempt.repair_attempts) if attempt.repair_attempts is not None else "-")}</td>
              <td>{html.escape(attempt.cache_source or "-")}</td>
              <td>{len(attempt.retrieved_doc_ids)}</td>
              <td>{item.scored_doc_count}</td>
              <td>{item.missing_doc_count}</td>
              <td class="doc-id" title="{html.escape(item.best_doc_id or '')}">{html.escape(item.best_doc_id or "-")}</td>
              <td class="score-cell">
                <span>{score_text(item.best_score)}</span>
                <span class="score-bar"><span style="width: {css_score_width(item.best_score)}"></span></span>
              </td>
              <td class="links">
                {link_for(attempt.generation_log_path, "gen", output_path)}
                {link_for(attempt.compile_log_path, "iv", output_path)}
                {link_for(attempt.reference_path, "gold", output_path)}
              </td>
              <td class="doc-list" title="{html.escape(doc_list)}">{html.escape(doc_list or "-")}</td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Verilog-Eval Retrieval Similarity Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #697384;
      --line: #d9dee7;
      --pass: #166534;
      --pass-bg: #dcfce7;
      --fail: #9f1239;
      --fail-bg: #ffe4e6;
      --unknown: #57534e;
      --unknown-bg: #e7e5e4;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtle {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      padding: 16px 32px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric-value {{
      margin-top: 4px;
      font-size: 24px;
      font-weight: 700;
    }}
    .metric-note {{ color: var(--muted); }}
    main {{ padding: 0 32px 32px; }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    table {{
      width: 100%;
      min-width: 1440px;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef2f6;
      color: #334155;
      text-align: left;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .03em;
    }}
    tr.pass {{ background: #fbfffc; }}
    tr.fail {{ background: #fffafb; }}
    tr:hover {{ background: #f2f7fb; }}
    .problem {{ font-weight: 650; }}
    .pill {{
      display: inline-flex;
      min-width: 44px;
      justify-content: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 12px;
    }}
    .pill.pass {{ color: var(--pass); background: var(--pass-bg); }}
    .pill.fail {{ color: var(--fail); background: var(--fail-bg); }}
    .pill.unknown {{ color: var(--unknown); background: var(--unknown-bg); }}
    .doc-id {{
      max-width: 280px;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .doc-list {{
      max-width: 420px;
      overflow: hidden;
      text-overflow: ellipsis;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .score-cell {{
      min-width: 150px;
    }}
    .score-cell > span:first-child {{
      display: inline-block;
      width: 44px;
      font-variant-numeric: tabular-nums;
    }}
    .score-bar {{
      display: inline-block;
      width: 82px;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #e5e7eb;
      vertical-align: middle;
    }}
    .score-bar span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      margin-right: 8px;
    }}
    .links a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>Verilog-Eval Retrieval Similarity Report</h1>
    <div class="subtle">
      Compared retrieved document <strong>{html.escape(doc_field)}</strong> text from
      <code>{html.escape(index_dir.relative_to(REPO_ROOT).as_posix() if index_dir.is_relative_to(REPO_ROOT) else index_dir.as_posix())}</code>
      with each attempt's golden <code>*_ref.sv</code>.
    </div>
  </header>
  <section class="metrics">
    <section class="metric">
      <div class="metric-label">Attempts</div>
      <div class="metric-value">{total}</div>
      <div class="metric-note">{passed} passed · {passed / total:.1%} pass rate</div>
    </section>
    <section class="metric">
      <div class="metric-label">Scored Attempts</div>
      <div class="metric-value">{with_score}</div>
      <div class="metric-note">{missing_docs} retrieved IDs missing from selected index</div>
    </section>
    {''.join(mode_cards)}
  </section>
  <main>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            <th>Problem</th>
            <th>Sample</th>
            <th>Success</th>
            <th>Passfail</th>
            <th>RAG Pass</th>
            <th>Syntax</th>
            <th>Lint</th>
            <th>Repairs</th>
            <th>Cache</th>
            <th>Retrieved</th>
            <th>Scored</th>
            <th>Missing</th>
            <th>Best Retrieved Doc</th>
            <th>Best Similarity</th>
            <th>Logs</th>
            <th>Retrieved Doc IDs</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""


def build_problem_summaries(scored_attempts: Sequence[ScoredAttempt]) -> List[ProblemSummary]:
    grouped: Dict[Tuple[str, str], List[ScoredAttempt]] = {}
    for item in scored_attempts:
        grouped.setdefault((item.attempt.problem, item.attempt.mode), []).append(item)

    problems: Dict[str, Dict[str, ModeProblemSummary]] = {}
    for (problem, mode), rows in grouped.items():
        rows = sorted(rows, key=lambda item: item.attempt.sample)
        retrieval_counter = Counter(tuple(item.attempt.retrieved_doc_ids) for item in rows)
        representative_tuple = retrieval_counter.most_common(1)[0][0] if retrieval_counter else ()
        representative_doc_ids = list(representative_tuple)
        best = max(
            (item for item in rows if item.best_score is not None),
            key=lambda item: item.best_score if item.best_score is not None else float("-inf"),
            default=None,
        )
        representative_row = next(
            (item for item in rows if tuple(item.attempt.retrieved_doc_ids) == representative_tuple),
            rows[0] if rows else None,
        )
        problems.setdefault(problem, {})[mode] = ModeProblemSummary(
            mode=mode,
            problem=problem,
            attempts=rows,
            representative_doc_ids=representative_doc_ids,
            retrieval_set_count=len(retrieval_counter),
            best_doc_id=best.best_doc_id if best else None,
            best_score=best.best_score if best else None,
            scored_doc_count=representative_row.scored_doc_count if representative_row else 0,
            missing_doc_count=representative_row.missing_doc_count if representative_row else 0,
        )

    return [ProblemSummary(problem=problem, modes=problems[problem]) for problem in sorted(problems)]


def mode_order(problem_rows: Sequence[ProblemSummary]) -> List[str]:
    return sorted({mode for row in problem_rows for mode in row.modes})


def passfail_counts_text(attempts: Sequence[ScoredAttempt]) -> str:
    counts = Counter(item.attempt.passfail or ("." if item.attempt.passed is True else "?") for item in attempts)
    order = [".", "G", "R", "C", "T", "r", "?"]
    parts = [f"{key}:{counts[key]}" for key in order if counts.get(key)]
    parts.extend(f"{key}:{count}" for key, count in sorted(counts.items()) if key not in order)
    return " ".join(parts) or "-"


def compact_doc_list(doc_ids: Sequence[str], max_items: int = 4) -> str:
    shown = list(doc_ids[:max_items])
    text = ", ".join(shown)
    if len(doc_ids) > max_items:
        text += f", +{len(doc_ids) - max_items}"
    return text or "-"


def sample_mark(item: ScoredAttempt) -> str:
    if item.attempt.passfail:
        return item.attempt.passfail
    if item.attempt.passed is True:
        return "."
    if item.attempt.passed is False:
        return "F"
    return "?"


def sample_class(item: ScoredAttempt) -> str:
    if item.attempt.passed is True or item.attempt.passfail == ".":
        return "pass"
    if item.attempt.passed is False or (item.attempt.passfail and item.attempt.passfail != "."):
        return "fail"
    return "unknown"


def true_count(items: Sequence[ScoredAttempt], attr: str) -> int:
    return sum(1 for item in items if getattr(item.attempt, attr) is True)


def render_sample_strip(summary: ModeProblemSummary) -> str:
    chips = []
    for item in summary.attempts:
        label = sample_mark(item)
        title = f"sample {item.attempt.sample:02d}: {label}"
        chips.append(
            f'<span class="sample {sample_class(item)}" title="{html.escape(title)}">'
            f"{html.escape(label)}</span>"
        )
    return "".join(chips)


def render_result_cell(summary: Optional[ModeProblemSummary]) -> str:
    if summary is None:
        return '<td class="empty">-</td>'
    attempts = summary.attempts
    cache_values = sorted({item.attempt.cache_source for item in attempts if item.attempt.cache_source})
    repair_values = [item.attempt.repair_attempts for item in attempts if item.attempt.repair_attempts is not None]
    repair_text = f"repairs max {max(repair_values)}" if repair_values else "repairs -"
    return f"""
      <td class="mode-result">
        <div class="primary-line">
          <span class="rate">{summary.passed}/{summary.total}</span>
          <span class="muted">{summary.pass_rate:.0%}</span>
        </div>
        <div class="samples">{render_sample_strip(summary)}</div>
        <div class="small">{html.escape(passfail_counts_text(attempts))}</div>
        <div class="small muted">
          gen {true_count(attempts, "rag_generation_passed")}/{summary.total} ·
          syn {true_count(attempts, "syntax_passed")}/{summary.total} ·
          lint {true_count(attempts, "lint_passed")}/{summary.total}
        </div>
        <div class="small muted">{html.escape(repair_text)} · cache {html.escape(",".join(cache_values) or "-")}</div>
      </td>
    """


def render_retrieval_cell(summary: Optional[ModeProblemSummary], output_path: Path) -> str:
    if summary is None:
        return '<td class="empty">-</td>'
    first_attempt = summary.attempts[0].attempt if summary.attempts else None
    retrieved_title = ", ".join(summary.representative_doc_ids)
    variant_text = (
        f"{summary.retrieval_set_count} retrieval sets"
        if summary.retrieval_set_count != 1
        else "1 retrieval set"
    )
    links = ""
    if first_attempt is not None:
        links = " ".join(
            item
            for item in [
                link_for(first_attempt.generation_log_path, "gen01", output_path),
                link_for(first_attempt.reference_path, "gold", output_path),
            ]
            if item
        )
    return f"""
      <td class="mode-retrieval">
        <div class="score-row">
          <span class="score-text">{score_text(summary.best_score)}</span>
          <span class="score-bar"><span style="width: {css_score_width(summary.best_score)}"></span></span>
        </div>
        <div class="doc-id" title="{html.escape(summary.best_doc_id or '')}">{html.escape(summary.best_doc_id or "-")}</div>
        <div class="small muted">{summary.scored_doc_count} scored · {summary.missing_doc_count} missing · {html.escape(variant_text)}</div>
        <div class="doc-list" title="{html.escape(retrieved_title)}">{html.escape(compact_doc_list(summary.representative_doc_ids))}</div>
        <div class="links">{links}</div>
      </td>
    """


def render_html(scored_attempts: Sequence[ScoredAttempt], output_path: Path, index_dir: Path, doc_field: str) -> str:
    problem_rows = build_problem_summaries(scored_attempts)
    modes = mode_order(problem_rows)
    total_attempts = len(scored_attempts)
    total_for_rate = max(total_attempts, 1)
    passed = sum(1 for item in scored_attempts if item.attempt.passed is True)
    with_score = sum(1 for item in scored_attempts if item.best_score is not None)
    missing_docs = sum(item.missing_doc_count for item in scored_attempts)
    mode_cards = []
    for mode, mode_total, mode_passed, rate, avg_score in mode_summaries(scored_attempts):
        mode_cards.append(
            f"""
            <section class="metric">
              <div class="metric-label">{html.escape(mode)}</div>
              <div class="metric-value">{mode_passed}/{mode_total}</div>
              <div class="metric-note">attempt pass {rate:.1%} · avg best {score_text(avg_score)}</div>
            </section>
            """
        )

    header_cells = ["<th>Problem</th>"]
    for mode in modes:
        label = html.escape(mode)
        header_cells.append(f"<th>{label} Generated</th>")
        header_cells.append(f"<th>{label} Retrieval</th>")

    body_rows = []
    for problem_row in problem_rows:
        cells = [f'<td class="problem">{html.escape(problem_row.problem)}</td>']
        for mode in modes:
            summary = problem_row.modes.get(mode)
            cells.append(render_result_cell(summary))
            cells.append(render_retrieval_cell(summary, output_path))
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    index_label = index_dir.relative_to(REPO_ROOT).as_posix() if index_dir.is_relative_to(REPO_ROOT) else index_dir.as_posix()
    min_width = 260 + max(1, len(modes)) * 560
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Verilog-Eval Retrieval Similarity Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #697384;
      --line: #d9dee7;
      --pass: #166534;
      --pass-bg: #dcfce7;
      --fail: #9f1239;
      --fail-bg: #ffe4e6;
      --unknown: #57534e;
      --unknown-bg: #e7e5e4;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    code {{
      padding: 1px 4px;
      border-radius: 4px;
      background: #eef2f6;
    }}
    .subtle, .muted {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      padding: 16px 32px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric-value {{
      margin-top: 4px;
      font-size: 24px;
      font-weight: 700;
    }}
    .metric-note {{ color: var(--muted); }}
    main {{ padding: 0 32px 32px; }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    table {{
      width: 100%;
      min-width: {min_width}px;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef2f6;
      color: #334155;
      text-align: left;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .03em;
      white-space: nowrap;
    }}
    tr:hover {{ background: #f2f7fb; }}
    .problem {{
      width: 230px;
      min-width: 230px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .mode-result {{
      width: 230px;
      min-width: 230px;
    }}
    .mode-retrieval {{
      width: 330px;
      min-width: 330px;
    }}
    .primary-line {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 6px;
    }}
    .rate {{
      font-size: 20px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .samples {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-bottom: 6px;
    }}
    .sample {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      border-radius: 6px;
      font-weight: 800;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .sample.pass {{ color: var(--pass); background: var(--pass-bg); }}
    .sample.fail {{ color: var(--fail); background: var(--fail-bg); }}
    .sample.unknown {{ color: var(--unknown); background: var(--unknown-bg); }}
    .small {{
      font-size: 12px;
      line-height: 1.35;
    }}
    .score-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }}
    .score-text {{
      width: 44px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .score-bar {{
      display: inline-block;
      width: 112px;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #e5e7eb;
      vertical-align: middle;
    }}
    .score-bar span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .doc-id, .doc-list {{
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: nowrap;
    }}
    .doc-id {{
      max-width: 300px;
      margin-bottom: 3px;
    }}
    .doc-list {{
      max-width: 300px;
      margin-top: 4px;
      color: var(--muted);
    }}
    .links {{
      margin-top: 5px;
      min-height: 18px;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      margin-right: 8px;
    }}
    .links a:hover {{ text-decoration: underline; }}
    .empty {{
      color: var(--muted);
      text-align: center;
      vertical-align: middle;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Verilog-Eval Retrieval Similarity Report</h1>
    <div class="subtle">
      One row per problem. Each mode summarizes all samples, while retrieval shows the representative retrieved set and best similarity against the golden <code>*_ref.sv</code>.
      Compared retrieved document <strong>{html.escape(doc_field)}</strong> text from <code>{html.escape(index_label)}</code>.
    </div>
  </header>
  <section class="metrics">
    <section class="metric">
      <div class="metric-label">Problems</div>
      <div class="metric-value">{len(problem_rows)}</div>
      <div class="metric-note">{total_attempts} attempts included</div>
    </section>
    <section class="metric">
      <div class="metric-label">Attempts</div>
      <div class="metric-value">{passed}/{total_attempts}</div>
      <div class="metric-note">{passed / total_for_rate:.1%} pass rate</div>
    </section>
    <section class="metric">
      <div class="metric-label">Scored Attempts</div>
      <div class="metric-value">{with_score}</div>
      <div class="metric-note">{missing_docs} retrieved IDs missing from selected index</div>
    </section>
    {''.join(mode_cards)}
  </section>
  <main>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>{''.join(header_cells)}</tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = [resolve_repo_path(item) for item in args.run_dirs] if args.run_dirs else default_run_dirs()
    if not run_dirs:
        raise FileNotFoundError("No run directories were provided and no default rag/model directories exist.")

    index_dir = resolve_repo_path(args.index)
    output_path = resolve_repo_path(args.output)
    documents = load_documents(index_dir)
    attempts = sort_attempts(attempt for run_dir in run_dirs for attempt in load_attempts(run_dir))
    if args.limit is not None:
        attempts = attempts[: args.limit]
    scored_attempts = score_attempts(attempts, documents, args.embedder, args.doc_field)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(scored_attempts, output_path, index_dir, args.doc_field), encoding="utf-8")

    passed = sum(1 for item in scored_attempts if item.attempt.passed is True)
    print(f"Wrote {output_path}")
    print(f"Attempts: {len(scored_attempts)}")
    print(f"Passed: {passed}")
    print(f"Scored attempts: {sum(1 for item in scored_attempts if item.best_score is not None)}")
    print(f"Missing retrieved IDs: {sum(item.missing_doc_count for item in scored_attempts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
