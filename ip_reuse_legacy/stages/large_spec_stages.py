from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_rtl.llm import extract_code
from rag_rtl.types import Diagnostic, VerificationReport

from ..manifest import (
    dependency_order as _dependency_order,
    group_manifest_payloads as _group_manifest_payloads,
    manifest_validation_errors as _manifest_validation_errors,
    markdown_chunks as _markdown_chunks,
    prepare_workspace as _prepare_workspace,
    render_manifest_index as _render_manifest_index,
    render_module_spec as _render_module_spec,
    write_text as _write_text,
)
from ..planning import (
    module_from_manifest as _module_from_manifest,
    requirements_from_manifest as _requirements_from_manifest,
)
from ..prompts import (
    build_chunk_partition_prompt,
    build_manifest_correction_prompt,
    build_manifest_merge_prompt,
    build_scoped_module_generation_prompt,
    build_scoped_module_repair_prompt,
    build_spec_partition_prompt,
    build_top_wrapper_generation_prompt,
    build_top_wrapper_repair_prompt,
)
from ..rtl import (
    combine_dependency_rtl as _combine_dependency_rtl,
    combine_final_rtl as _combine_final_rtl,
    decision_generation_payload as _decision_generation_payload,
    module_signature as _module_signature,
    validate_single_module_rtl as _validate_single_module_rtl,
)
from ..serialization import parse_json_object as _parse_json_object
from ..types import AgenticIpReuseResult, IpReusePlan, LlmTrace


class LargeSpecStagesMixin:
    def _run_large_spec(
        self,
        prompt: str,
        *,
        target: str,
        top_module: Optional[str],
        constraints: List[str],
        llm_traces: List[LlmTrace],
        retrieval_traces: List[Dict[str, Any]],
        workspace_dir: Optional[str | Path],
    ) -> AgenticIpReuseResult:
        workspace = _prepare_workspace(workspace_dir)
        self._stage("large_spec_workspace", "complete", workspace_dir=str(workspace))
        artifacts: Dict[str, str] = {}
        original_spec_path = workspace / "original_spec.txt"
        _write_text(original_spec_path, prompt)
        artifacts["original_spec"] = str(original_spec_path)

        manifest = self._partition_large_spec(
            prompt,
            target=target,
            top_module=top_module,
            constraints=constraints,
            llm_traces=llm_traces,
            workspace=workspace,
            artifacts=artifacts,
        )
        manifest_path = workspace / "spec_manifest.json"
        _write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
        artifacts["spec_manifest"] = str(manifest_path)

        index_text = _render_manifest_index(manifest)
        index_path = workspace / "index.txt"
        _write_text(index_path, index_text)
        artifacts["index"] = str(index_path)

        for module_payload in manifest["modules"]:
            spec_path = workspace / "specs" / f"{module_payload['name']}.txt"
            _write_text(
                spec_path,
                _render_module_spec(
                    manifest,
                    module_payload,
                    {item["name"]: item for item in manifest["modules"]},
                ),
            )
            artifacts[f"spec:{module_payload['name']}"] = str(spec_path)

        if self.config.recursive_decomposition:
            return self._run_recursive_manifest(
                manifest,
                target=target,
                constraints=constraints,
                llm_traces=llm_traces,
                retrieval_traces=retrieval_traces,
                workspace=workspace,
                artifacts=artifacts,
                index_text=index_text,
            )

        requirements = _requirements_from_manifest(manifest)
        modules = [_module_from_manifest(item) for item in manifest["modules"]]
        self._stage(
            "decomposition",
            "complete",
            source="large_spec_manifest",
            module_count=len(modules),
            modules=[module.name for module in modules],
        )
        decisions = self._build_decisions(modules, llm_traces, retrieval_traces)
        plan = IpReusePlan(requirements=requirements, modules=modules, decisions=decisions)
        decisions_by_name = {decision.module.name: decision for decision in decisions}
        module_payloads = {item["name"]: item for item in manifest["modules"]}
        generation_order = _dependency_order(manifest)
        generated_modules: Dict[str, str] = {}
        module_generation: List[Dict[str, Any]] = []
        total_repair_attempts = 0
        failed_modules: Dict[str, str] = {}

        for module_name in generation_order:
            module_payload = module_payloads[module_name]

            failed_deps = [d for d in module_payload["dependencies"] if d in failed_modules]
            if failed_deps:
                error_msg = f"module {module_name} skipped: dependencies failed: {', '.join(failed_deps)}"
                failed_modules[module_name] = error_msg
                module_generation.append(
                    {
                        "module": module_name,
                        "dependencies": list(module_payload["dependencies"]),
                        "repair_attempts": 0,
                        "passed": False,
                        "error": error_msg,
                        "spec_path": str(workspace / "specs" / f"{module_name}.txt"),
                    }
                )
                self._stage("module_generation", "skipped", module=module_name, reason=error_msg)
                continue

            module_spec_text = (workspace / "specs" / f"{module_name}.txt").read_text(encoding="utf-8")
            dependency_interfaces = {
                name: module_payloads[name]["ports"]
                for name in module_payload["dependencies"]
            }
            self._stage("module_generation", "running", module=module_name)
            final_text = self._complete_text(
                f"module_generation:{module_name}",
                build_scoped_module_generation_prompt(
                    module_spec_text,
                    dependency_interfaces,
                    _decision_generation_payload(decisions_by_name[module_name]),
                    target,
                ),
                llm_traces,
            )
            module_rtl = extract_code(final_text).strip()
            try:
                _validate_single_module_rtl(module_rtl, module_name)
            except RuntimeError as exc:
                error_path = workspace / "errors" / f"{module_name}_generation_response.txt"
                _write_text(error_path, final_text)
                artifacts[f"error:{module_name}:generation"] = str(error_path)
                error_msg = str(exc)
                failed_modules[module_name] = error_msg
                module_generation.append(
                    {
                        "module": module_name,
                        "dependencies": list(module_payload["dependencies"]),
                        "repair_attempts": 0,
                        "passed": False,
                        "error": error_msg,
                        "spec_path": str(workspace / "specs" / f"{module_name}.txt"),
                    }
                )
                self._stage("module_generation", "error", module=module_name, error=error_msg)
                continue

            verification_input = _combine_dependency_rtl(
                generated_modules,
                module_payloads,
                module_payload["dependencies"],
                module_rtl,
            )
            self._stage("module_verification", "running", module=module_name)
            verification = self._verify_module(verification_input, module_name)
            attempts = 0
            repair_errors: List[str] = []
            while not verification.passed and attempts < self.config.max_repair_attempts:
                repair_errors.extend(d.stderr for d in verification.diagnostics if not d.passed)
                attempts += 1
                total_repair_attempts += 1
                errors = [d.stderr for d in verification.diagnostics if not d.passed]
                self._stage("module_repair", "running", module=module_name, attempt=attempts, errors=errors)
                final_text = self._complete_text(
                    f"module_repair:{module_name}:{attempts}",
                    build_scoped_module_repair_prompt(
                        module_spec_text,
                        dependency_interfaces,
                        module_rtl,
                        [asdict(item) for item in verification.diagnostics],
                        target,
                    ),
                    llm_traces,
                )
                module_rtl = extract_code(final_text).strip()
                try:
                    _validate_single_module_rtl(module_rtl, module_name)
                except RuntimeError as exc:
                    error_path = workspace / "errors" / f"{module_name}_repair_{attempts}_response.txt"
                    _write_text(error_path, final_text)
                    artifacts[f"error:{module_name}:repair:{attempts}"] = str(error_path)
                    repair_errors.append(str(exc))
                    verification = VerificationReport(
                        syntax_passed=False,
                        lint_passed=False,
                        diagnostics=[Diagnostic(tool="rtl_extraction", passed=False, stderr=str(exc))],
                    )
                    break
                verification_input = _combine_dependency_rtl(
                    generated_modules,
                    module_payloads,
                    module_payload["dependencies"],
                    module_rtl,
                )
                self._stage("module_verification", "running", module=module_name, repair_attempt=attempts)
                verification = self._verify_module(verification_input, module_name)
            if not verification.passed:
                repair_errors.extend(d.stderr for d in verification.diagnostics if not d.passed)
                error_msg = f"large-spec module {module_name} failed verification after {attempts} repair attempt(s)"
                error_path = workspace / "errors" / f"{module_name}_verification.json"
                _write_text(error_path, json.dumps(asdict(verification), ensure_ascii=False, indent=2))
                _write_text(workspace / "errors" / f"{module_name}_failed.v", (module_rtl or "") + "\n")
                artifacts[f"error:{module_name}"] = str(error_path)
                failed_modules[module_name] = error_msg
                module_generation.append(
                    {
                        "module": module_name,
                        "dependencies": list(module_payload["dependencies"]),
                        "repair_attempts": attempts,
                        "passed": False,
                        "error": error_msg,
                        "repair_errors": repair_errors,
                        "spec_path": str(workspace / "specs" / f"{module_name}.txt"),
                    }
                )
                self._stage("module_verification", "error", module=module_name, attempts=attempts, error=error_msg)
                continue

            rtl_path = workspace / "rtl" / f"{module_name}.v"
            _write_text(rtl_path, module_rtl + "\n")
            artifacts[f"rtl:{module_name}"] = str(rtl_path)
            generated_modules[module_name] = module_rtl
            module_generation.append(
                {
                    "module": module_name,
                    "dependencies": list(module_payload["dependencies"]),
                    "repair_attempts": attempts,
                    "passed": True,
                    "repair_errors": repair_errors,
                    "spec_path": str(workspace / "specs" / f"{module_name}.txt"),
                    "rtl_path": str(rtl_path),
                }
            )
            self._stage(
                "module_verification",
                "complete",
                module=module_name,
                attempts=attempts,
                rtl_chars=len(module_rtl),
                passed=True,
            )

        top_name = manifest["top_module"]["name"]
        if failed_modules:
            self._stage(
                "module_generation_summary",
                "partial",
                failed=list(failed_modules.keys()),
                succeeded=list(generated_modules.keys()),
            )
        signatures = {name: _module_signature(rtl) for name, rtl in generated_modules.items()}
        self._stage("top_integration", "running", top_module=top_name, failed_modules=list(failed_modules.keys()))
        final_text = self._complete_text(
            "top_integration",
            build_top_wrapper_generation_prompt(index_text, signatures, target, top_name),
            llm_traces,
        )
        wrapper_rtl = extract_code(final_text).strip()
        try:
            _validate_single_module_rtl(wrapper_rtl, top_name)
        except RuntimeError as exc:
            error_path = workspace / "errors" / "top_generation_response.txt"
            _write_text(error_path, final_text)
            artifacts["error:top:generation"] = str(error_path)
            self._stage("top_integration", "error", top_module=top_name, error=str(exc))
            combined_rtl = ""
            wrapper_rtl = ""
            verification = VerificationReport(
                syntax_passed=False,
                lint_passed=False,
                diagnostics=[Diagnostic(tool="rtl_extraction", passed=False, stderr=str(exc))],
            )
            top_attempts = 0
        else:
            combined_rtl = _combine_final_rtl(generated_modules, generation_order, wrapper_rtl)
            self._stage("combined_verification", "running", top_module=top_name)
            verification = self._verify_or_empty(combined_rtl, top_name)
            top_attempts = 0
            while not verification.passed and top_attempts < self.config.max_repair_attempts:
                top_attempts += 1
                total_repair_attempts += 1
                self._stage("top_repair", "running", top_module=top_name, attempt=top_attempts)
                final_text = self._complete_text(
                    f"top_repair:{top_attempts}",
                    build_top_wrapper_repair_prompt(
                        index_text,
                        signatures,
                        wrapper_rtl,
                        [asdict(item) for item in verification.diagnostics],
                        target,
                        top_name,
                    ),
                    llm_traces,
                )
                wrapper_rtl = extract_code(final_text).strip()
                try:
                    _validate_single_module_rtl(wrapper_rtl, top_name)
                except RuntimeError as exc:
                    error_path = workspace / "errors" / f"top_repair_{top_attempts}_response.txt"
                    _write_text(error_path, final_text)
                    artifacts[f"error:top:repair:{top_attempts}"] = str(error_path)
                    verification = VerificationReport(
                        syntax_passed=False,
                        lint_passed=False,
                        diagnostics=[Diagnostic(tool="rtl_extraction", passed=False, stderr=str(exc))],
                    )
                    break
                combined_rtl = _combine_final_rtl(generated_modules, generation_order, wrapper_rtl)
                self._stage("combined_verification", "running", top_module=top_name, repair_attempt=top_attempts)
                verification = self._verify_or_empty(combined_rtl, top_name)

        top_path = workspace / "rtl" / f"{top_name}.v"
        combined_path = workspace / "combined" / f"{top_name}.sv"
        _write_text(top_path, wrapper_rtl + "\n")
        _write_text(combined_path, combined_rtl + "\n")
        artifacts["rtl:top"] = str(top_path)
        artifacts["combined_rtl"] = str(combined_path)
        for path in sorted(workspace.rglob("*")):
            if path.is_file():
                artifacts.setdefault(f"workspace:{path.relative_to(workspace)}", str(path))
        self._stage(
            "combined_verification",
            "complete" if verification.passed else "error",
            top_module=top_name,
            passed=verification.passed,
            repair_attempts=top_attempts,
        )

        result = AgenticIpReuseResult(
            plan=plan,
            rtl=combined_rtl,
            final_text=final_text,
            verification=verification,
            repair_attempts=total_repair_attempts,
            llm_traces=llm_traces,
            retrieval_traces=retrieval_traces,
            large_spec_manifest=manifest,
            module_generation=module_generation,
            workspace_dir=str(workspace),
            artifacts=artifacts,
        )
        self._stage(
            "agent_complete",
            "complete",
            passed=verification.passed,
            repair_attempts=total_repair_attempts,
            workspace_dir=str(workspace),
        )
        return result

