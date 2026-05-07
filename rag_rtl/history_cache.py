from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from .embeddings import Embedder

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
JSON_ARRAY_RE = re.compile(r"\[[\s\S]*?\]")
JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "build",
    "code",
    "create",
    "for",
    "from",
    "hdl",
    "in",
    "input",
    "make",
    "module",
    "of",
    "output",
    "rtl",
    "that",
    "the",
    "to",
    "verilog",
    "with",
}

KEYWORD_EXTRACTION_PROMPT = """You are a Verilog specification keyword extraction assistant.

Your task is to extract concise, structured keywords from the given Verilog code specification.

You must identify the following fields:

1. direction:
   - The implementation direction or task type.
   - Choose one of:
     - "design"
     - "verify"
     - "testbench"
     - "debug"
     - "optimize"
     - "analyze"
     - "unknown"
   - Use "design" if the spec asks to implement, create, write, or generate a Verilog module.
   - Use "verify" if the spec asks to check correctness, prove behavior, or validate an existing module.
   - Use "testbench" if the spec asks to write simulation tests or stimulus.
   - Use "debug" if the spec asks to fix errors.
   - Use "optimize" if the spec asks to improve area, timing, power, or performance.
   - Use "analyze" if the spec asks to explain or inspect existing code.

2. module_name:
   - Extract the target Verilog module name.
   - If multiple modules are mentioned, return all module names as a list.
   - If no module name is given, return [].

3. type:
   - Determine the circuit type.
   - Choose one of:
     - "combinational"
     - "sequential"
     - "mixed"
     - "unknown"
   - Use "combinational" if behavior depends only on current inputs.
   - Use "sequential" if the spec mentions clock, reset, register, flip-flop, counter, FSM, memory, latch, or state.
   - Use "mixed" if both combinational and sequential parts are clearly required.
   - Use "unknown" if the spec does not provide enough information.

4. gate_usage:
   - Extract explicitly required gates, operators, or implementation primitives.
   - Examples:
     - "NAND"
     - "NOR"
     - "AND"
     - "OR"
     - "XOR"
     - "XNOR"
     - "Inverter"
     - "MUX"
     - "Adder"
     - "Subtractor"
     - "Comparator"
     - "Register"
     - "Flip-flop"
     - "Latch"
   - If the spec says "use only NAND gates", return ["NAND"].
   - If the spec says "not i", return ["Inverter"].
   - If the spec says "a and b", return ["AND"].
   - If no gate or operator is specified, return [].

5. input_signals:
   - Extract input signal names if mentioned.
   - Return [] if not specified.

6. output_signals:
   - Extract output signal names if mentioned.
   - Return [] if not specified.

7. behavior_keywords:
   - Extract short behavior-level keywords.
   - Examples:
     - "invert"
     - "logical and"
     - "logical or"
     - "multiplexer"
     - "counter"
     - "state transition"
     - "parity"
     - "comparison"
   - Keep each keyword short.

8. constraints:
   - Extract implementation constraints.
   - Examples:
     - "continuous assignment"
     - "structural Verilog"
     - "gate-level implementation"
     - "no always block"
     - "synchronous reset"
     - "active-low reset"
     - "use only NAND gates"
   - Return [] if no constraints are given.

Return only valid JSON.
Do not include explanation.
Do not include markdown.
Do not include comments.
Do not invent information not supported by the specification.
Return the JSON object in this shape:
{
  "direction": "design",
  "module_name": ["invert"],
  "type": "combinational",
  "gate_usage": ["Inverter"],
  "signals": {
    "input": ["i"],
    "output": ["o"]
  },
  "keywords": [
    "invert",
    "logical not",
    "continuous assignment",
    "combinational"
  ]
}
Map input_signals to signals.input and output_signals to signals.output.
Put short behavior keywords, constraints, and useful circuit-type terms in keywords.

Specification:
{SPEC}
"""


@dataclass
class CacheEntry:
    query: str
    result: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class CacheLookup:
    query: str
    mode: str
    reuse_threshold: float
    evidence_threshold: float
    decision: str
    score: Optional[float] = None
    entry: Optional[CacheEntry] = None
    query_keywords: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    candidate_count: int = 0
    best_history_match: Optional[Dict[str, Any]] = None

    @property
    def reusable_entry(self) -> Optional[CacheEntry]:
        return self.entry if self.decision == "reuse" else None

    @property
    def evidence_entry(self) -> Optional[CacheEntry]:
        return self.entry if self.decision == "evidence" else None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "reuse_threshold": self.reuse_threshold,
            "evidence_threshold": self.evidence_threshold,
            "decision": self.decision,
            "score": self.score,
            "matched_query": self.entry.query if self.entry else None,
            "query_keywords": self.query_keywords,
            "matched_keywords": self.matched_keywords,
            "candidate_count": self.candidate_count,
            "best_history_match": self.best_history_match,
        }


def extract_keywords(text: str) -> List[str]:
    keywords: List[str] = []
    seen: Set[str] = set()
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) < 2 or token in STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords


def _normalize_keyword(value: str, *, allow_short: bool = False, allow_stopwords: bool = False) -> Optional[str]:
    keyword = value.strip().lower()
    keyword = re.sub(r"[`'\"()\[\]{}<>]", "", keyword)
    keyword = re.sub(r"[\s\-/]+", "_", keyword)
    keyword = re.sub(r"[^a-z0-9_:$]", "", keyword)
    keyword = keyword.strip("_")
    if not keyword:
        return None
    if not allow_short and len(keyword) < 2:
        return None
    if not allow_stopwords and keyword in STOPWORDS:
        return None
    return keyword


def _dedupe_keywords(
    values: Sequence[str],
    *,
    allow_short: bool = False,
    allow_stopwords: bool = False,
) -> List[str]:
    keywords: List[str] = []
    seen: Set[str] = set()
    for value in values:
        keyword = _normalize_keyword(str(value), allow_short=allow_short, allow_stopwords=allow_stopwords)
        if keyword and keyword not in seen:
            seen.add(keyword)
            keywords.append(keyword)
    return keywords


class LlmKeywordExtractor:
    def __init__(self, llm_client: Any, fallback: Callable[[str], List[str]] = extract_keywords):
        self.llm_client = llm_client
        self.fallback = fallback

    def __call__(self, text: str) -> List[str]:
        prompt = KEYWORD_EXTRACTION_PROMPT.replace("{SPEC}", text.strip())
        try:
            response = self.llm_client.complete(prompt, temperature=0.0, max_tokens=320)
        except Exception:
            return self.fallback(text)
        keywords = self._parse_response(response)
        return keywords or self.fallback(text)

    def _parse_response(self, response: str) -> List[str]:
        response = response.strip()
        payload: Any = None
        if response.startswith("{"):
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            match = JSON_OBJECT_RE.search(response)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None
        if payload is None:
            match = JSON_ARRAY_RE.search(response)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None
        if isinstance(payload, dict):
            return _dedupe_keywords(
                _structured_keyword_values(payload),
                allow_short=True,
                allow_stopwords=True,
            )
        if isinstance(payload, list):
            return _dedupe_keywords([str(item) for item in payload])
        rough_values = re.split(r"[,;\n]", response)
        return _dedupe_keywords(rough_values)


def _extend_keyword_values(values: List[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        values.append(value)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _extend_keyword_values(values, nested)
        return
    if isinstance(value, list):
        for item in value:
            _extend_keyword_values(values, item)
        return
    values.append(str(value))


def _structured_keyword_values(payload: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in (
        "module_name",
        "type",
        "gate_usage",
        "input_signals",
        "output_signals",
        "behavior_keywords",
        "constraints",
        "signals",
        "keywords",
    ):
        _extend_keyword_values(values, payload.get(key))
    return [value for value in values if str(value).strip().lower() != "unknown"]


class HistorySemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        path: str | Path,
        threshold: Optional[float] = None,
        max_size: int = 1000,
        reuse_threshold: float = 0.95,
        evidence_threshold: float = 0.88,
        mode: str = "keywords",
        keyword_extractor: Optional[Callable[[str], List[str]]] = None,
    ):
        if mode not in {"keywords", "direct"}:
            raise ValueError("cache mode must be 'keywords' or 'direct'")
        self.embedder = embedder
        self.path = Path(path)
        self.reuse_threshold = reuse_threshold if threshold is None else threshold
        self.evidence_threshold = evidence_threshold
        self.threshold = self.reuse_threshold
        self.mode = mode
        self.max_size = max_size
        self.keyword_extractor = keyword_extractor or extract_keywords
        self.entries: List[CacheEntry] = []
        self._lock = threading.RLock()
        if self.path.exists():
            self.load()

    def get(self, query: str) -> Optional[CacheEntry]:
        return self.lookup(query).reusable_entry

    def lookup(self, query: str) -> CacheLookup:
        query_keywords = self.keyword_extractor(query) if self.mode == "keywords" else extract_keywords(query)
        with self._lock:
            if not self.entries:
                return CacheLookup(
                    query=query,
                    mode=self.mode,
                    reuse_threshold=self.reuse_threshold,
                    evidence_threshold=self.evidence_threshold,
                    decision="miss",
                    query_keywords=query_keywords,
                )

            candidates = self._keyword_candidates(query_keywords) if self.mode == "keywords" else list(self.entries)
            if not candidates:
                return CacheLookup(
                    query=query,
                    mode=self.mode,
                    reuse_threshold=self.reuse_threshold,
                    evidence_threshold=self.evidence_threshold,
                    decision="miss",
                    query_keywords=query_keywords,
                    candidate_count=0,
                )

            query_vector = self._normalize(self.embedder.encode([query])[0])
            scored = self._score_entries(query_vector, candidates)
            best_score, best_entry = scored[0]
            matched_keywords = self._matched_keywords(query_keywords, best_entry)
            decision = self._decision(best_score)
            lookup = CacheLookup(
                query=query,
                mode=self.mode,
                reuse_threshold=self.reuse_threshold,
                evidence_threshold=self.evidence_threshold,
                decision=decision,
                score=best_score,
                entry=best_entry if decision in {"reuse", "evidence"} else None,
                query_keywords=query_keywords,
                matched_keywords=matched_keywords,
                candidate_count=len(candidates),
                best_history_match=self._match_metadata(best_entry, best_score, query_keywords),
            )
            if decision == "reuse" and best_entry:
                best_entry.hit_count += 1
                best_entry.last_access_time = time.time()
                best_entry.metadata["last_score"] = best_score
                self._save_unlocked()
            return lookup

    def put(self, query: str, result: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        embedding = self._normalize(self.embedder.encode([query])[0]).astype(float).tolist()
        entry_metadata = dict(metadata or {})
        if self.mode == "keywords":
            entry_metadata.setdefault("keywords", self.keyword_extractor(query))
        else:
            entry_metadata.setdefault("keywords", extract_keywords(query))
        with self._lock:
            self.entries.append(CacheEntry(query=query, result=result, embedding=embedding, metadata=entry_metadata))
            if len(self.entries) > self.max_size:
                self.entries.sort(key=lambda entry: entry.last_access_time)
                self.entries = self.entries[-self.max_size :]
            self._save_unlocked()

    def load(self) -> None:
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.entries = [CacheEntry(**item) for item in payload]

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(entry) for entry in self.entries], handle, ensure_ascii=False, indent=2)

    def _keyword_candidates(self, query_keywords: Sequence[str]) -> List[CacheEntry]:
        query_set = set(query_keywords)
        if not query_set:
            return []
        candidates: List[CacheEntry] = []
        for entry in self.entries:
            if query_set & set(self._entry_keywords(entry)):
                candidates.append(entry)
        return candidates

    def _score_entries(self, query_vector: np.ndarray, entries: Sequence[CacheEntry]) -> List[tuple[float, CacheEntry]]:
        scored = [
            (float(np.dot(query_vector, self._normalize(np.asarray(entry.embedding, dtype=np.float32)))), entry)
            for entry in entries
        ]
        return sorted(scored, key=lambda item: item[0], reverse=True)

    def _decision(self, score: float) -> str:
        if score >= self.reuse_threshold:
            return "reuse"
        if score >= self.evidence_threshold:
            return "evidence"
        return "miss"

    def _match_metadata(self, entry: CacheEntry, score: float, query_keywords: Sequence[str]) -> Dict[str, Any]:
        return {
            "query": entry.query,
            "score": score,
            "metadata": entry.metadata,
            "matched_keywords": self._matched_keywords(query_keywords, entry),
        }

    def _matched_keywords(self, query_keywords: Sequence[str], entry: CacheEntry) -> List[str]:
        entry_keywords = set(self._entry_keywords(entry))
        return sorted(set(query_keywords) & entry_keywords)

    def _entry_keywords(self, entry: CacheEntry) -> List[str]:
        keywords = entry.metadata.get("keywords")
        if isinstance(keywords, list):
            return _dedupe_keywords([str(keyword) for keyword in keywords], allow_short=True, allow_stopwords=True)
        return extract_keywords(entry.query)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
