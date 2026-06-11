from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from rag_rtl.llm import extract_code
from rag_rtl.types import Diagnostic, VerificationReport

from ..manifest import (
    dependency_order as _dependency_order,
    leaf_decomposition as _leaf_decomposition,
    recursive_decomposition_validation_errors as _recursive_decomposition_validation_errors,
    recursive_node_manifest as _recursive_node_manifest,
    render_manifest_index as _render_manifest_index,
    render_module_spec as _render_module_spec,
    write_text as _write_text,
)
from ..planning import (
    module_from_manifest as _module_from_manifest,
    requirements_from_manifest as _requirements_from_manifest,
)
from ..prompts import (
    build_recursive_decomposition_correction_prompt,
    build_recursive_decomposition_prompt,
    build_scoped_module_generation_prompt,
    build_scoped_module_repair_prompt,
    build_top_wrapper_generation_prompt,
    build_top_wrapper_repair_prompt,
)
from ..rtl import (
    combine_dependency_rtl as _combine_dependency_rtl,
    combine_final_rtl as _combine_final_rtl,
    combine_recursive_wrapper_rtl as _combine_recursive_wrapper_rtl,
    decision_generation_payload as _decision_generation_payload,
    module_signature as _module_signature,
    validate_single_module_rtl as _validate_single_module_rtl,
)
from ..serialization import parse_json_object as _parse_json_object
from ..types import AgenticIpReuseResult, IpReusePlan, LlmTrace, ModuleReuseDecision


class RecursiveStagesMixin:
    def _run_recursive_manifest(
        self,
        manifest: Dict[str, Any],
        *,
        target: str,
        constraints: List[str],
        llm_traces: List[LlmTrace],
        retrieval_traces: List[Dict[str, Any]],
        workspace: Path,
        artifacts: Dict[str, str],
        index_text: str,
    ) -> AgenticIpReuseResult:
        del constraints
        top_name = manifest["top_module"]["name"]
        if len(manifest["modules"]) > self.config.recursive_max_nodes:
            raise RuntimeError(
                f"root decomposition exceeded maximum node count {self.config.recursive_max_nodes}"
            )
        state: Dict[str, Any] = {
            "root_manifest": manifest,
            "workspace": workspace,
            "artifacts": artifacts,
            "llm_traces": llm_traces,
            "retrieval_traces": retrieval_traces,
            "module_payloads": {item["name"]: item for item in manifest["modules"]},
            "known_names": {top_name, *(item["name"] for item in manifest["modules"])},
            "node_count": len(manifest["modules"]),
            "generated_modules": {},
            "generation_order": [],
            "module_generation": [],
            "decisions": {},
            "repair_attempts": 0,
            "failed_modules": {},
        }
        self._stage(
            "recursive_decomposition",
            "running",
            top_module=top_name,
            max_depth=self.config.recursive_max_depth,
            max_nodes=self.config.recursive_max_nodes,
        )
        root_children: List[Dict[str, Any]] = []
        for module_name in _dependency_order(manifest):
            tree = self._build_recursive_module_tree(
                state["module_payloads"][module_name],
                depth=1,
                ancestors=[top_name],
                target=target,
                state=state,
            )
            root_children.append(tree)

        decomposition_tree = {
            "module": top_name,
            "kind": "root",
            "depth": 0,
            "children": root_children,
        }
        tree_path = workspace / "decomposition_tree.json"
        _write_text(tree_path, json.dumps(decomposition_tree, ensure_ascii=False, indent=2))
        artifacts["decomposition_tree"] = str(tree_path)
        self._stage(
            "recursive_decomposition",
            "complete",
            top_module=top_name,
            node_count=state["node_count"],
        )

        self._stage("recursive_rtl_generation", "running", top_module=top_name)
        tree_by_name = {tree["module"]: tree for tree in root_children}
        for module_name in _dependency_order(manifest):
            self._generate_recursive_tree_module(
                tree_by_name[module_name],
                target=target,
                state=state,
            )
        self._stage(
            "recursive_rtl_generation",
            "complete",
            top_module=top_name,
            generated=list(state["generated_modules"].keys()),
            failed=list(state["failed_modules"].keys()),
        )

        direct_names = [item["name"] for item in manifest["modules"]]
        if state["failed_modules"]:
            self._stage(
                "module_generation_summary",
                "partial",
                failed=list(state["failed_modules"].keys()),
                succeeded=list(state["generated_modules"].keys()),
            )
        signatures = {
            name: _module_signature(state["generated_modules"][name])
            for name in direct_names
            if name in state["generated_modules"]
        }
        self._stage("top_integration", "running", top_module=top_name, failed_modules=list(state["failed_modules"].keys()))
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
            combined_rtl = _combine_final_rtl(
                state["generated_modules"],
                state["generation_order"],
                wrapper_rtl,
            )
            self._stage("combined_verification", "running", top_module=top_name)
            verification = self._verify_or_empty(combined_rtl, top_name)
            top_attempts = 0
            while not verification.passed and top_attempts < self.config.max_repair_attempts:
                top_attempts += 1
                state["repair_attempts"] += 1
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
                combined_rtl = _combine_final_rtl(
                    state["generated_modules"],
                    state["generation_order"],
                    wrapper_rtl,
                )
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

        requirements = _requirements_from_manifest(manifest)
        modules = [_module_from_manifest(state["module_payloads"][name]) for name in state["generation_order"]]
        decisions = [state["decisions"][name] for name in state["generation_order"]]
        result = AgenticIpReuseResult(
            plan=IpReusePlan(requirements=requirements, modules=modules, decisions=decisions),
            rtl=combined_rtl,
            final_text=final_text,
            verification=verification,
            repair_attempts=state["repair_attempts"],
            llm_traces=llm_traces,
            retrieval_traces=retrieval_traces,
            large_spec_manifest=manifest,
            decomposition_tree=decomposition_tree,
            module_generation=state["module_generation"],
            workspace_dir=str(workspace),
            artifacts=artifacts,
        )
        self._stage(
            "agent_complete",
            "complete",
            passed=verification.passed,
            repair_attempts=state["repair_attempts"],
            workspace_dir=str(workspace),
        )
        return result

    def _build_recursive_module_tree(
        self,
        module_payload: Dict[str, Any],
        *,
        depth: int,
        ancestors: List[str],
        target: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        module_name = module_payload["name"]
        workspace: Path = state["workspace"]
        spec_path = workspace / "specs" / f"{module_name}.txt"
        if not spec_path.exists():
            _write_text(
                spec_path,
                _render_module_spec(state["root_manifest"], module_payload, state["module_payloads"]),
            )
        state["artifacts"][f"spec:{module_name}"] = str(spec_path)
        module_spec_text = spec_path.read_text(encoding="utf-8")

        decomposition = self._decompose_recursive_module(
            module_payload,
            module_spec_text=module_spec_text,
            depth=depth,
            ancestors=ancestors,
            target=target,
            state=state,
        )
        # A decomposition that yields ≤1 child produces a pointless wrapper; force leaf.
        if decomposition["decision"] == "decompose" and len(decomposition.get("children") or []) <= 1:
            decomposition = _leaf_decomposition(module_payload, "Forced leaf: decomposition yielded ≤1 child.")
        decomposition_path = workspace / "decompositions" / f"{module_name}.json"
        _write_text(decomposition_path, json.dumps(decomposition, ensure_ascii=False, indent=2))
        state["artifacts"][f"decomposition:{module_name}"] = str(decomposition_path)

        if decomposition["decision"] == "leaf":
            return {
                "module": module_name,
                "kind": "leaf",
                "depth": depth,
                "reason": decomposition.get("reason"),
                "children": [],
            }

        children = decomposition["children"]
        state["node_count"] += len(children)
        if state["node_count"] > self.config.recursive_max_nodes:
            raise RuntimeError(
                f"recursive decomposition exceeded maximum node count {self.config.recursive_max_nodes}"
            )
        for child in children:
            state["known_names"].add(child["name"])
            state["module_payloads"][child["name"]] = child
            child_spec_path = workspace / "specs" / f"{child['name']}.txt"
            _write_text(
                child_spec_path,
                _render_module_spec(state["root_manifest"], child, state["module_payloads"]),
            )
            state["artifacts"][f"spec:{child['name']}"] = str(child_spec_path)

        node_manifest = _recursive_node_manifest(
            state["root_manifest"],
            module_payload,
            decomposition["parent_module"],
            children,
        )
        node_index_text = _render_manifest_index(node_manifest)
        external_dependencies = {
            name: state["module_payloads"][name]["ports"]
            for name in module_payload["dependencies"]
            if name in state["module_payloads"]
        }
        if external_dependencies:
            node_index_text += (
                "\nEXTERNAL DEPENDENCY INTERFACES\n"
                + json.dumps(external_dependencies, ensure_ascii=False, indent=2)
                + "\n"
            )
        node_index_path = workspace / "indexes" / f"{module_name}.txt"
        _write_text(node_index_path, node_index_text)
        state["artifacts"][f"index:{module_name}"] = str(node_index_path)

        child_trees: List[Dict[str, Any]] = []
        for child_name in _dependency_order(node_manifest):
            tree = self._build_recursive_module_tree(
                state["module_payloads"][child_name],
                depth=depth + 1,
                ancestors=[*ancestors, module_name],
                target=target,
                state=state,
            )
            child_trees.append(tree)
        return {
            "module": module_name,
            "kind": "composite",
            "depth": depth,
            "reason": decomposition.get("reason"),
            "children": child_trees,
        }

    def _generate_recursive_tree_module(
        self,
        tree: Dict[str, Any],
        *,
        target: str,
        state: Dict[str, Any],
    ) -> List[str]:
        module_name = tree["module"]
        module_payload = state["module_payloads"][module_name]
        workspace: Path = state["workspace"]
        spec_path = workspace / "specs" / f"{module_name}.txt"
        module_spec_text = spec_path.read_text(encoding="utf-8")
        depth = int(tree.get("depth") or 0)

        if tree.get("kind") == "leaf":
            rtl, attempts, repair_errors = self._generate_recursive_leaf(
                module_payload,
                module_spec_text=module_spec_text,
                depth=depth,
                target=target,
                state=state,
            )
            if module_name in state["failed_modules"]:
                state["module_generation"].append(
                    {
                        "module": module_name,
                        "kind": "leaf",
                        "depth": depth,
                        "children": [],
                        "repair_attempts": attempts,
                        "passed": False,
                        "error": state["failed_modules"][module_name],
                        "repair_errors": repair_errors,
                        "spec_path": str(spec_path),
                    }
                )
                tree["failed"] = True
                tree["error"] = state["failed_modules"][module_name]
                return []
            state["generated_modules"][module_name] = rtl
            state["generation_order"].append(module_name)
            state["module_generation"].append(
                {
                    "module": module_name,
                    "kind": "leaf",
                    "depth": depth,
                    "children": [],
                    "repair_attempts": attempts,
                    "passed": True,
                    "repair_errors": repair_errors,
                    "spec_path": str(spec_path),
                    "rtl_path": state["artifacts"][f"rtl:{module_name}"],
                }
            )
            return [module_name]

        child_trees = list(tree.get("children") or [])
        children = [state["module_payloads"][child["module"]] for child in child_trees]
        parent_module = {
            "name": module_name,
            "ports": module_payload["ports"],
            "instances": [],
        }
        decomposition: Dict[str, Any] = {}
        decomposition_path = workspace / "decompositions" / f"{module_name}.json"
        if decomposition_path.exists():
            decomposition = json.loads(decomposition_path.read_text(encoding="utf-8"))
            parent_module = decomposition.get("parent_module") or parent_module
        node_manifest = _recursive_node_manifest(
            state["root_manifest"],
            module_payload,
            parent_module,
            children,
        )
        node_index_path = workspace / "indexes" / f"{module_name}.txt"
        node_index_text = node_index_path.read_text(encoding="utf-8")

        child_by_name = {child["module"]: child for child in child_trees}
        subtree_order: List[str] = []
        for child_name in _dependency_order(node_manifest):
            subtree_order.extend(
                self._generate_recursive_tree_module(
                    child_by_name[child_name],
                    target=target,
                    state=state,
                )
            )

        failed_children = [child["name"] for child in children if child["name"] in state["failed_modules"]]
        if failed_children:
            error_msg = f"composite {module_name} skipped: children failed: {', '.join(failed_children)}"
            state["failed_modules"][module_name] = error_msg
            state["module_generation"].append(
                {
                    "module": module_name,
                    "kind": "composite",
                    "depth": depth,
                    "children": [child["name"] for child in children],
                    "repair_attempts": 0,
                    "passed": False,
                    "error": error_msg,
                    "spec_path": str(spec_path),
                }
            )
            self._stage("recursive_wrapper_generation", "skipped", module=module_name, depth=depth, failed_children=failed_children)
            tree["failed"] = True
            tree["error"] = error_msg
            return []

        signatures = {
            child["name"]: _module_signature(state["generated_modules"][child["name"]])
            for child in children
        }
        signatures.update(
            {
                name: _module_signature(state["generated_modules"][name])
                for name in module_payload["dependencies"]
                if name in state["generated_modules"]
            }
        )
        self._stage("recursive_wrapper_generation", "running", module=module_name, depth=depth)
        try:
            wrapper_rtl = self._generate_module_rtl(
                f"recursive_wrapper_generation:{module_name}",
                build_top_wrapper_generation_prompt(node_index_text, signatures, target, module_name),
                module_name,
                state["llm_traces"],
            )
        except RuntimeError as exc:
            error_msg = f"recursive wrapper {module_name} RTL extraction failed: {exc}"
            state["failed_modules"][module_name] = error_msg
            state["module_generation"].append(
                {
                    "module": module_name,
                    "kind": "composite",
                    "depth": depth,
                    "children": [child["name"] for child in children],
                    "repair_attempts": 0,
                    "passed": False,
                    "error": error_msg,
                    "repair_errors": [str(exc)],
                    "spec_path": str(spec_path),
                }
            )
            self._stage("recursive_wrapper_generation", "error", module=module_name, depth=depth, error=error_msg)
            tree["failed"] = True
            tree["error"] = error_msg
            return []
        verification_input = _combine_recursive_wrapper_rtl(
            state["generated_modules"],
            state["module_payloads"],
            module_payload["dependencies"],
            subtree_order,
            wrapper_rtl,
        )
        verification = self._verify_module(verification_input, module_name)
        attempts = 0
        repair_errors: List[str] = []
        while not verification.passed and attempts < self.config.max_repair_attempts:
            repair_errors.extend(d.stderr for d in verification.diagnostics if not d.passed)
            attempts += 1
            state["repair_attempts"] += 1
            errors = [d.stderr for d in verification.diagnostics if not d.passed]
            self._stage("recursive_wrapper_repair", "running", module=module_name, depth=depth, attempt=attempts, errors=errors)
            try:
                wrapper_rtl = self._generate_module_rtl(
                    f"recursive_wrapper_repair:{module_name}:{attempts}",
                    build_top_wrapper_repair_prompt(
                        node_index_text,
                        signatures,
                        wrapper_rtl,
                        [asdict(item) for item in verification.diagnostics],
                        target,
                        module_name,
                    ),
                    module_name,
                    state["llm_traces"],
                )
            except RuntimeError as exc:
                repair_errors.append(str(exc))
                verification = VerificationReport(
                    syntax_passed=False,
                    lint_passed=False,
                    diagnostics=[Diagnostic(tool="rtl_extraction", passed=False, stderr=str(exc))],
                )
                break
            verification_input = _combine_recursive_wrapper_rtl(
                state["generated_modules"],
                state["module_payloads"],
                module_payload["dependencies"],
                subtree_order,
                wrapper_rtl,
            )
            verification = self._verify_module(verification_input, module_name)
        if not verification.passed:
            repair_errors.extend(d.stderr for d in verification.diagnostics if not d.passed)
            error_msg = f"recursive wrapper {module_name} failed verification after {attempts} repair attempt(s)"
            error_path = workspace / "errors" / f"{module_name}_recursive_wrapper_verification.json"
            _write_text(error_path, json.dumps(asdict(verification), ensure_ascii=False, indent=2))
            state["artifacts"][f"error:{module_name}:recursive_wrapper"] = str(error_path)
            state["failed_modules"][module_name] = error_msg
            state["module_generation"].append(
                {
                    "module": module_name,
                    "kind": "composite",
                    "depth": depth,
                    "children": [child["name"] for child in children],
                    "repair_attempts": attempts,
                    "passed": False,
                    "error": error_msg,
                    "repair_errors": repair_errors,
                    "spec_path": str(spec_path),
                }
            )
            self._stage("recursive_wrapper_generation", "error", module=module_name, depth=depth, attempts=attempts, passed=False, error=error_msg)
            tree["failed"] = True
            tree["error"] = error_msg
            return []

        rtl_path = workspace / "rtl" / f"{module_name}.v"
        _write_text(rtl_path, wrapper_rtl + "\n")
        state["artifacts"][f"rtl:{module_name}"] = str(rtl_path)
        state["generated_modules"][module_name] = wrapper_rtl
        state["generation_order"].append(module_name)
        state["decisions"][module_name] = ModuleReuseDecision(
            module=_module_from_manifest(module_payload),
            action="new",
            integration_notes="Generated as a recursive wrapper over immediate child modules.",
            rationale=str(decomposition.get("reason") or "Module was recursively decomposed."),
        )
        state["module_generation"].append(
            {
                "module": module_name,
                "kind": "composite",
                "depth": depth,
                "children": [child["name"] for child in children],
                "repair_attempts": attempts,
                "passed": True,
                "repair_errors": repair_errors,
                "spec_path": str(spec_path),
                "rtl_path": str(rtl_path),
            }
        )
        self._stage(
            "recursive_wrapper_generation",
            "complete",
            module=module_name,
            depth=depth,
            child_count=len(children),
            passed=True,
        )
        return [*subtree_order, module_name]

    # stashed
    def _generate_recursive_module(
        self,
        module_payload: Dict[str, Any],
        *,
        depth: int,
        ancestors: List[str],
        target: str,
        state: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[str]]:
        module_name = module_payload["name"]
        workspace: Path = state["workspace"]
        spec_path = workspace / "specs" / f"{module_name}.txt"
        if not spec_path.exists():
            _write_text(
                spec_path,
                _render_module_spec(state["root_manifest"], module_payload, state["module_payloads"]),
            )
        state["artifacts"][f"spec:{module_name}"] = str(spec_path)
        module_spec_text = spec_path.read_text(encoding="utf-8")

        decomposition = self._decompose_recursive_module(
            module_payload,
            module_spec_text=module_spec_text,
            depth=depth,
            ancestors=ancestors,
            target=target,
            state=state,
        )
        # A decomposition that yields ≤1 child produces a pointless wrapper; force leaf.
        if decomposition["decision"] == "decompose" and len(decomposition.get("children") or []) <= 1:
            decomposition = _leaf_decomposition(module_payload, "Forced leaf: decomposition yielded ≤1 child.")
        decomposition_path = workspace / "decompositions" / f"{module_name}.json"
        _write_text(decomposition_path, json.dumps(decomposition, ensure_ascii=False, indent=2))
        state["artifacts"][f"decomposition:{module_name}"] = str(decomposition_path)

        if decomposition["decision"] == "leaf":
            rtl, attempts, repair_errors = self._generate_recursive_leaf(
                module_payload,
                module_spec_text=module_spec_text,
                depth=depth,
                target=target,
                state=state,
            )
            if module_name in state["failed_modules"]:
                state["module_generation"].append(
                    {
                        "module": module_name,
                        "kind": "leaf",
                        "depth": depth,
                        "children": [],
                        "repair_attempts": attempts,
                        "passed": False,
                        "error": state["failed_modules"][module_name],
                        "repair_errors": repair_errors,
                        "spec_path": str(spec_path),
                    }
                )
                return {
                    "module": module_name,
                    "kind": "leaf",
                    "depth": depth,
                    "reason": decomposition.get("reason"),
                    "children": [],
                    "failed": True,
                    "error": state["failed_modules"][module_name],
                }, []
            state["generated_modules"][module_name] = rtl
            state["generation_order"].append(module_name)
            state["module_generation"].append(
                {
                    "module": module_name,
                    "kind": "leaf",
                    "depth": depth,
                    "children": [],
                    "repair_attempts": attempts,
                    "passed": True,
                    "repair_errors": repair_errors,
                    "spec_path": str(spec_path),
                    "rtl_path": state["artifacts"][f"rtl:{module_name}"],
                }
            )
            return {
                "module": module_name,
                "kind": "leaf",
                "depth": depth,
                "reason": decomposition.get("reason"),
                "children": [],
            }, [module_name]

        children = decomposition["children"]
        state["node_count"] += len(children)
        if state["node_count"] > self.config.recursive_max_nodes:
            raise RuntimeError(
                f"recursive decomposition exceeded maximum node count {self.config.recursive_max_nodes}"
            )
        for child in children:
            state["known_names"].add(child["name"])
            state["module_payloads"][child["name"]] = child
            child_spec_path = workspace / "specs" / f"{child['name']}.txt"
            _write_text(
                child_spec_path,
                _render_module_spec(state["root_manifest"], child, state["module_payloads"]),
            )
            state["artifacts"][f"spec:{child['name']}"] = str(child_spec_path)

        node_manifest = _recursive_node_manifest(
            state["root_manifest"],
            module_payload,
            decomposition["parent_module"],
            children,
        )
        node_index_text = _render_manifest_index(node_manifest)
        external_dependencies = {
            name: state["module_payloads"][name]["ports"]
            for name in module_payload["dependencies"]
            if name in state["module_payloads"]
        }
        if external_dependencies:
            node_index_text += (
                "\nEXTERNAL DEPENDENCY INTERFACES\n"
                + json.dumps(external_dependencies, ensure_ascii=False, indent=2)
                + "\n"
            )
        node_index_path = workspace / "indexes" / f"{module_name}.txt"
        _write_text(node_index_path, node_index_text)
        state["artifacts"][f"index:{module_name}"] = str(node_index_path)

        child_trees: List[Dict[str, Any]] = []
        subtree_order: List[str] = []
        for child_name in _dependency_order(node_manifest):
            tree, child_order = self._generate_recursive_module(
                state["module_payloads"][child_name],
                depth=depth + 1,
                ancestors=[*ancestors, module_name],
                target=target,
                state=state,
            )
            child_trees.append(tree)
            subtree_order.extend(child_order)

        failed_children = [child["name"] for child in children if child["name"] in state["failed_modules"]]
        if failed_children:
            error_msg = f"composite {module_name} skipped: children failed: {', '.join(failed_children)}"
            state["failed_modules"][module_name] = error_msg
            state["module_generation"].append(
                {
                    "module": module_name,
                    "kind": "composite",
                    "depth": depth,
                    "children": [child["name"] for child in children],
                    "repair_attempts": 0,
                    "passed": False,
                    "error": error_msg,
                    "spec_path": str(spec_path),
                }
            )
            self._stage("recursive_wrapper_generation", "skipped", module=module_name, depth=depth, failed_children=failed_children)
            return {
                "module": module_name,
                "kind": "composite",
                "depth": depth,
                "reason": decomposition.get("reason"),
                "children": child_trees,
                "failed": True,
                "error": error_msg,
            }, []
        return {
            "module": module_name,
            "kind": "composite",
            "depth": depth,
            "reason": decomposition.get("reason"),
            "children": child_trees,
        }, [*subtree_order, module_name]

    def _decompose_recursive_module(
        self,
        module_payload: Dict[str, Any],
        *,
        module_spec_text: str,
        depth: int,
        ancestors: List[str],
        target: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        module_name = module_payload["name"]
        if depth >= self.config.recursive_max_depth:
            self._stage(
                "recursive_decomposition",
                "complete",
                module=module_name,
                depth=depth,
                decision="leaf",
                reason="max_depth",
            )
            return _leaf_decomposition(module_payload, "Maximum recursive depth reached.")

        self._stage("recursive_decomposition", "running", module=module_name, depth=depth)
        response = self._complete_text(
            f"recursive_decomposition:{module_name}",
            build_recursive_decomposition_prompt(
                module_spec_text,
                target,
                depth,
                self.config.recursive_max_depth,
                sorted(state["known_names"]),
            ),
            state["llm_traces"],
        )
        response_path = state["workspace"] / "decompositions" / f"{module_name}_response.txt"
        _write_text(response_path, response)
        state["artifacts"][f"decomposition_response:{module_name}"] = str(response_path)
        decomposition = _parse_json_object(response) or {}
        errors = _recursive_decomposition_validation_errors(
            decomposition,
            module_payload,
            existing_names=state["known_names"],
            ancestors=ancestors,
        )
        if errors:
            correction = self._complete_text(
                f"recursive_decomposition_correction:{module_name}",
                build_recursive_decomposition_correction_prompt(
                    decomposition,
                    errors,
                    module_spec_text,
                    target,
                ),
                state["llm_traces"],
            )
            correction_path = state["workspace"] / "decompositions" / f"{module_name}_corrected_response.txt"
            _write_text(correction_path, correction)
            state["artifacts"][f"decomposition_correction:{module_name}"] = str(correction_path)
            decomposition = _parse_json_object(correction) or {}
            errors = _recursive_decomposition_validation_errors(
                decomposition,
                module_payload,
                existing_names=state["known_names"],
                ancestors=ancestors,
            )
        if errors:
            error_path = state["workspace"] / "errors" / f"{module_name}_decomposition_validation.json"
            _write_text(error_path, json.dumps(errors, ensure_ascii=False, indent=2))
            state["artifacts"][f"error:{module_name}:decomposition"] = str(error_path)
            raise RuntimeError(f"recursive decomposition for {module_name} failed validation; see {error_path}")
        self._stage(
            "recursive_decomposition",
            "complete",
            module=module_name,
            depth=depth,
            decision=decomposition["decision"],
            child_count=len(decomposition["children"]),
        )
        return decomposition

    def _generate_recursive_leaf(
        self,
        module_payload: Dict[str, Any],
        *,
        module_spec_text: str,
        depth: int,
        target: str,
        state: Dict[str, Any],
    ) -> tuple[str, int, List[str]]:
        module_name = module_payload["name"]
        module = _module_from_manifest(module_payload)
        decision = self._build_decisions(
            [module],
            state["llm_traces"],
            state["retrieval_traces"],
        )[0]
        state["decisions"][module_name] = decision
        dependency_interfaces = {
            name: state["module_payloads"][name]["ports"]
            for name in module_payload["dependencies"]
            if name in state["module_payloads"]
        }
        self._stage("module_generation", "running", module=module_name, depth=depth)
        try:
            rtl = self._generate_module_rtl(
                f"module_generation:{module_name}",
                build_scoped_module_generation_prompt(
                    module_spec_text,
                    dependency_interfaces,
                    _decision_generation_payload(decision),
                    target,
                ),
                module_name,
                state["llm_traces"],
            )
        except RuntimeError as exc:
            error_msg = f"recursive leaf {module_name} RTL extraction failed: {exc}"
            state["failed_modules"][module_name] = error_msg
            self._stage("module_generation", "error", module=module_name, depth=depth, error=error_msg)
            return "", 0, [str(exc)]
        # Resolve a testbench: prefer a matching RealBench file, else ask the LLM.
        leaf_verifier = self._make_leaf_verifier(module_name, module_spec_text, rtl, state)
        verification_input = _combine_dependency_rtl(
            state["generated_modules"],
            state["module_payloads"],
            module_payload["dependencies"],
            rtl,
        )
        verification = self._verify_with(leaf_verifier, verification_input, module_name)
        attempts = 0
        repair_errors: List[str] = []
        while not verification.passed and attempts < self.config.max_repair_attempts:
            repair_errors.extend(d.stderr for d in verification.diagnostics if not d.passed)
            attempts += 1
            state["repair_attempts"] += 1
            errors = [d.stderr for d in verification.diagnostics if not d.passed]
            self._stage("module_repair", "running", module=module_name, depth=depth, attempt=attempts, errors=errors)
            try:
                rtl = self._generate_module_rtl(
                    f"module_repair:{module_name}:{attempts}",
                    build_scoped_module_repair_prompt(
                        module_spec_text,
                        dependency_interfaces,
                        rtl,
                        [asdict(item) for item in verification.diagnostics],
                        target,
                    ),
                    module_name,
                    state["llm_traces"],
                )
            except RuntimeError as exc:
                repair_errors.append(str(exc))
                verification = VerificationReport(
                    syntax_passed=False,
                    lint_passed=False,
                    diagnostics=[Diagnostic(tool="rtl_extraction", passed=False, stderr=str(exc))],
                )
                break
            verification_input = _combine_dependency_rtl(
                state["generated_modules"],
                state["module_payloads"],
                module_payload["dependencies"],
                rtl,
            )
            verification = self._verify_with(leaf_verifier, verification_input, module_name)
        if not verification.passed:
            repair_errors.extend(d.stderr for d in verification.diagnostics if not d.passed)
            error_msg = f"recursive leaf {module_name} failed verification after {attempts} repair attempt(s)"
            error_path = state["workspace"] / "errors" / f"{module_name}_verification.json"
            _write_text(error_path, json.dumps(asdict(verification), ensure_ascii=False, indent=2))
            state["artifacts"][f"error:{module_name}"] = str(error_path)
            state["failed_modules"][module_name] = error_msg
            self._stage("module_verification", "error", module=module_name, depth=depth, attempts=attempts, passed=False, error=error_msg)
            return "", attempts, repair_errors
        rtl_path = state["workspace"] / "rtl" / f"{module_name}.v"
        _write_text(rtl_path, rtl + "\n")
        state["artifacts"][f"rtl:{module_name}"] = str(rtl_path)
        self._register_live_document(module_name, module_payload, rtl)
        self._stage(
            "module_verification",
            "complete",
            module=module_name,
            depth=depth,
            attempts=attempts,
            passed=True,
        )
        return rtl, attempts, repair_errors
