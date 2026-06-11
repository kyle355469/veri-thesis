from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..manifest import (
    group_manifest_payloads as _group_manifest_payloads,
    manifest_validation_errors as _manifest_validation_errors,
    markdown_chunks as _markdown_chunks,
    write_text as _write_text,
)
from ..prompts import (
    build_chunk_partition_prompt,
    build_manifest_correction_prompt,
    build_manifest_merge_prompt,
    build_spec_partition_prompt,
)
from ..serialization import parse_json_object as _parse_json_object
from ..types import LlmTrace


class PartitionStagesMixin:
    def _partition_large_spec(
        self,
        prompt: str,
        *,
        target: str,
        top_module: Optional[str],
        constraints: List[str],
        llm_traces: List[LlmTrace],
        workspace: Path,
        artifacts: Dict[str, str],
    ) -> Dict[str, Any]:
        errors_dir = workspace / "errors"
        if self.config.decomposition_mode == "original":
            self._stage("large_spec_split", "running", mode="full", prompt_chars=len(prompt))
            try:
                response = self._complete_text(
                    "large_spec_split_full",
                    build_spec_partition_prompt(prompt, target, top_module, constraints),
                    llm_traces,
                )
                response_path = errors_dir / "full_split_response.txt"
                _write_text(response_path, response)
                artifacts["split_response:full"] = str(response_path)
                payload = _parse_json_object(response) or {}
                validation_errors = _manifest_validation_errors(payload, top_module)
                if not validation_errors:
                    self._stage("large_spec_split", "complete", mode="full", module_count=len(payload["modules"]))
                    return payload
                _write_text(errors_dir / "full_split_validation.json", json.dumps(validation_errors, indent=2))
            except Exception as exc:  # noqa: BLE001 - the chunk fallback is intentional.
                _write_text(errors_dir / "full_split_error.txt", str(exc) + "\n")
        else:
            self._stage("large_spec_split", "skipped", mode="chunking", prompt_chars=len(prompt))

        chunks = _markdown_chunks(prompt, self.config.large_spec_chunk_chars)
        chunk_stage = "large_spec_chunking" if self.config.decomposition_mode == "chunking" else "large_spec_chunk_fallback"
        self._stage(chunk_stage, "running", chunk_count=len(chunks), mode=self.config.decomposition_mode)

        def _process_chunk(index_chunk: tuple[int, str]) -> Dict[str, Any]:
            index, chunk = index_chunk
            prompt = build_chunk_partition_prompt(chunk, index, len(chunks), target, top_module)
            last_error_path: Optional[Path] = None
            for attempt in range(self.config.max_generation_retries + 1):
                stage = f"large_spec_chunk:{index}" if attempt == 0 else f"large_spec_chunk:{index}:retry{attempt}"
                suffix = "" if attempt == 0 else f"_retry{attempt:03d}"
                try:
                    response = self._complete_text(stage, prompt, llm_traces)
                except Exception as exc:
                    error_path = errors_dir / f"chunk_{index:03d}{suffix}_error.txt"
                    _write_text(error_path, str(exc) + "\n")
                    artifacts[f"error:chunk:{index}{suffix}"] = str(error_path)
                    last_error_path = error_path
                    if attempt < self.config.max_generation_retries:
                        self._stage(chunk_stage, "retry", chunk=index, attempt=attempt + 1, reason=str(exc))
                        continue
                    raise RuntimeError(f"large-spec chunk {index} failed; see {error_path}") from exc
                response_path = errors_dir / f"chunk_{index:03d}{suffix}_response.txt"
                _write_text(response_path, response)
                artifacts[f"split_response:chunk:{index}{suffix}"] = str(response_path)
                payload = _parse_json_object(response)
                if payload is not None:
                    if attempt:
                        self._stage(chunk_stage, "recovered", chunk=index, attempt=attempt)
                    return payload
                last_error_path = response_path
                if attempt < self.config.max_generation_retries:
                    self._stage(chunk_stage, "retry", chunk=index, attempt=attempt + 1, reason="json_parse_failed")
            assert last_error_path is not None
            raise RuntimeError(f"large-spec chunk {index} returned invalid JSON; see {last_error_path}")

        with ThreadPoolExecutor() as executor:
            partials: List[Dict[str, Any]] = list(
                executor.map(_process_chunk, enumerate(chunks, start=1))
            )

        manifest = self._merge_partial_manifests(
            partials,
            target=target,
            top_module=top_module,
            constraints=constraints,
            llm_traces=llm_traces,
            errors_dir=errors_dir,
            artifacts=artifacts,
        )
        validation_errors = _manifest_validation_errors(manifest, top_module)
        if validation_errors:
            try:
                correction_response = self._complete_text(
                    "large_spec_manifest_correction",
                    build_manifest_correction_prompt(manifest, validation_errors, target, top_module),
                    llm_traces,
                )
            except Exception as exc:
                error_path = errors_dir / "manifest_correction_error.txt"
                _write_text(error_path, str(exc) + "\n")
                artifacts["error:manifest_correction"] = str(error_path)
                raise RuntimeError(f"large-spec manifest correction failed; see {error_path}") from exc
            correction_path = errors_dir / "corrected_split_response.txt"
            _write_text(correction_path, correction_response)
            artifacts["split_response:corrected"] = str(correction_path)
            manifest = _parse_json_object(correction_response) or {}
            validation_errors = _manifest_validation_errors(manifest, top_module)
        if validation_errors:
            validation_path = errors_dir / "manifest_validation.json"
            _write_text(validation_path, json.dumps(validation_errors, indent=2))
            artifacts["error:manifest_validation"] = str(validation_path)
            self._stage(chunk_stage, "error", errors=validation_errors)
            raise RuntimeError(f"large-spec manifest validation failed; see {validation_path}")
        self._stage(
            chunk_stage,
            "complete",
            chunk_count=len(chunks),
            module_count=len(manifest["modules"]),
            mode=self.config.decomposition_mode,
        )
        return manifest

    def _merge_partial_manifests(
        self,
        partials: List[Dict[str, Any]],
        *,
        target: str,
        top_module: Optional[str],
        constraints: List[str],
        llm_traces: List[LlmTrace],
        errors_dir: Path,
        artifacts: Dict[str, str],
    ) -> Dict[str, Any]:
        current = partials
        round_index = 1
        while True:
            groups = _group_manifest_payloads(current, max(self.config.large_spec_chunk_chars * 2, 4000))

            def _process_group(index_group: tuple[int, List[Dict[str, Any]]], _round: int = round_index) -> Dict[str, Any]:
                group_index, group = index_group
                prompt = build_manifest_merge_prompt(group, target, top_module, constraints)
                last_error_path: Optional[Path] = None
                for attempt in range(self.config.max_generation_retries + 1):
                    stage = (
                        f"large_spec_merge:{_round}:{group_index}"
                        if attempt == 0
                        else f"large_spec_merge:{_round}:{group_index}:retry{attempt}"
                    )
                    suffix = "" if attempt == 0 else f"_retry{attempt:03d}"
                    try:
                        response = self._complete_text(stage, prompt, llm_traces)
                    except Exception as exc:
                        error_path = errors_dir / f"merge_{_round:02d}_{group_index:03d}{suffix}_error.txt"
                        _write_text(error_path, str(exc) + "\n")
                        artifacts[f"error:merge:{_round}:{group_index}{suffix}"] = str(error_path)
                        last_error_path = error_path
                        if attempt < self.config.max_generation_retries:
                            self._stage("large_spec_merge", "retry", round=_round, group=group_index, attempt=attempt + 1, reason=str(exc))
                            continue
                        raise RuntimeError(f"large-spec merge failed; see {error_path}") from exc
                    response_path = errors_dir / f"merge_{_round:02d}_{group_index:03d}{suffix}_response.txt"
                    _write_text(response_path, response)
                    artifacts[f"split_response:merge:{_round}:{group_index}{suffix}"] = str(response_path)
                    payload = _parse_json_object(response)
                    if payload is not None:
                        if attempt:
                            self._stage("large_spec_merge", "recovered", round=_round, group=group_index, attempt=attempt)
                        return payload
                    last_error_path = response_path
                    if attempt < self.config.max_generation_retries:
                        self._stage("large_spec_merge", "retry", round=_round, group=group_index, attempt=attempt + 1, reason="json_parse_failed")
                assert last_error_path is not None
                raise RuntimeError(f"large-spec merge returned invalid JSON; see {last_error_path}")

            with ThreadPoolExecutor() as executor:
                merged: List[Dict[str, Any]] = list(
                    executor.map(_process_group, enumerate(groups, start=1))
                )
            if len(merged) == 1:
                return merged[0]
            current = merged
            round_index += 1

