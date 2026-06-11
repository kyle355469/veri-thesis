# Improvements Roadmap

This document describes the 9 remaining technical improvements for making `agentic_ip_reuse` a production-grade Verilog generation pipeline. Improvements 2, 4, 5, and 6 have already been implemented.

---

## Improvement 1 — Embedding-Based Semantic IP Search

### Problem
The current `JsonIpRepository.search()` uses regex tokenization and token-set intersection (`repository.py:19–36`). "AXI4 burst master" and "high-bandwidth bus initiator" share zero tokens, causing silent false negatives on real-world IP catalogs like Realbench.

### Implementation Plan
1. **Embed catalog at load time.** In `JsonIpRepository.__init__`, call an embedding API (e.g., the same vLLM endpoint via `/v1/embeddings`) for each IP's concatenated `name + summary + tags + behavior` text. Cache embeddings in a sidecar `.embeddings.npy` file next to the catalog JSON so re-launch is instant.
2. **Add `cosine_search()` to `JsonIpRepository`.** Given a query string, embed it, then rank all IPs by cosine similarity. Return the top-k `IpCandidate` objects with `score` set to the cosine value.
3. **Keep token search as fallback.** If the embedding endpoint is unavailable (no `VLLM_BASE_URL`), fall back to the existing token-set search transparently.
4. **Add `--search-mode [token|embedding|hybrid]` CLI flag** in `cli.py`. Hybrid mode averages both scores.

### Key Files
- `agentic_ip_reuse/repository.py` — add `cosine_search()`, update `search()` to dispatch by mode
- `agentic_ip_reuse/llm.py` — add `VllmClient.embed(texts)` using `/v1/embeddings`
- `agentic_ip_reuse/cli.py` — add `--search-mode` arg

### Dependencies
- `numpy` (for cosine arithmetic and `.npy` caching)
- vLLM endpoint must expose `/v1/embeddings` (or swap to `sentence-transformers` for fully offline use)

### Complexity: M

---

## Improvement 3 — Long Spec Chunking and RAG Retrieval

### Problem
The entire design spec is injected as a single user message in `agent.py:30–34`. A 50-page Realbench spec will hit the model's context limit, dilute attention, and cause the agent to miss requirements from later sections.

### Implementation Plan
1. **Add a `SpecChunker` class** (`agentic_ip_reuse/spec_chunker.py`). Split the input spec by Markdown headings (`##`, `###`) or by fixed token window (e.g., 512 tokens) with 64-token overlap. Each chunk gets an index and a short heading label.
2. **Embed each chunk** using the same embedding endpoint as Improvement 1. Store in an in-memory list of `(embedding, chunk_text)` pairs. For very large specs (>200 chunks), use `chromadb` or `faiss` for the index.
3. **Replace the full-spec user message with a RAG retrieval step.** At the start of each module's decomposition sub-task, embed the module's name + role and retrieve the top-5 most relevant spec chunks. Inject them as a focused `system` context block.
4. **Add `--chunk-spec` CLI flag.** When set, `cmd_run` passes the spec through `SpecChunker` before building the `DesignTask`.

### Key Files
- `agentic_ip_reuse/spec_chunker.py` — new file: `SpecChunker`, `ChunkedSpec`
- `agentic_ip_reuse/prompts.py` — `build_user_prompt()` accepts optional `retrieved_chunks: list[str]`
- `agentic_ip_reuse/cli.py` — `--chunk-spec` flag
- `agentic_ip_reuse/hierarchical.py` — pass relevant chunks into `_build_sub_task()`

### Dependencies
- `numpy` for embedding arithmetic
- Optional: `chromadb` or `faiss-cpu` for large specs

### Complexity: M

---

## Improvement 7 — Multi-Agent Parallelism and Higher Step Budget

### Problem
The agent loop is serial: each module is searched, inspected, evaluated, and generated one at a time. For a 10-module Realbench design with 3 tool calls per module, that's 30+ sequential LLM calls. `max_steps=16` becomes the bottleneck.

### Implementation Plan
1. **Phase 1 — increase budget with progressive summarization.** After every 8 steps, summarize the tool call history into a single compressed `assistant` message (call the LLM with `tool_choice="none"` to produce the summary). This resets the effective context length without losing decisions. Implement in `agent.py` as a `_compress_history()` method called at every 8-step boundary.
2. **Phase 2 — parallel module sub-agents.** After the top-level agent produces the module list, spawn one `AgenticIpReuseAgent` per module simultaneously using `concurrent.futures.ThreadPoolExecutor`. Each sub-agent handles its own IP search + evaluation + RTL generation, writing to `output_dir/modules/<name>/`. Merge results in `HierarchicalAgent.run()`.
3. **Add `--parallel-modules N` CLI flag** to control the thread pool size. Default `N=1` (serial) to stay safe on rate-limited endpoints.

### Key Files
- `agentic_ip_reuse/agent.py` — add `_compress_history()` and an `AgentConfig.compress_every: int = 8` field
- `agentic_ip_reuse/hierarchical.py` — add parallel execution path inside `run()` using `ThreadPoolExecutor`
- `agentic_ip_reuse/cli.py` — add `--parallel-modules` flag

### Dependencies
- `concurrent.futures` (stdlib, no new deps)

### Complexity: M

---

## Improvement 8 — Structured Spec-to-Constraint Extraction Stage

### Problem
Free-text requirements are passed directly to the LLM, which must re-derive concrete values (clock frequency, bus protocol version, data width, reset polarity) on every tool call. Misinterpretations compound through 16 steps and are invisible until the final plan.

### Implementation Plan
1. **Add a `ConstraintExtractor` pre-pass** (`agentic_ip_reuse/constraint_extractor.py`). Call the LLM once with `tool_choice="none"` and a focused prompt asking it to return a JSON object with these mandatory fields:
   ```json
   { "clock_mhz": null, "bus_protocol": null, "data_width_bits": null,
     "reset_polarity": "active_low", "pipeline_stages": null,
     "target_technology": null, "additional": [] }
   ```
2. **Attach the extracted `ConstraintRecord` to `DesignTask`** as a new optional field `extracted_constraints: Dict[str, Any]`.
3. **Inject the constraint record as a compact JSON block** at the top of every subsequent `build_user_prompt()` call so each step has the resolved values in context without re-inference.
4. **Use constraints as hard filters** in `AgentToolExecutor.search_reuse_ip()`: if `data_width_bits=32`, filter out IPs that only support `DATA_WIDTH: "8..8"`.

### Key Files
- `agentic_ip_reuse/constraint_extractor.py` — new: `ConstraintExtractor`, `ConstraintRecord`
- `agentic_ip_reuse/types.py` — add `extracted_constraints` field to `DesignTask`
- `agentic_ip_reuse/prompts.py` — `build_user_prompt()` injects `ConstraintRecord` if present
- `agentic_ip_reuse/cli.py` — run extractor before `agent.run()` when `--extract-constraints` flag is set

### Dependencies
None beyond existing LLM client.

### Complexity: S

---

## Improvement 9 — Adapter/Glue Wrapper Auto-Generation

### Problem
`ReuseDecision.required_adapters` is a plain text list (`types.py:84–90`). Every detected interface mismatch (e.g., valid-ready stream into push/pop FIFO) results in a prose note that becomes manual RTL work, defeating the reuse benefit.

### Implementation Plan
1. **Add an adapter template library** (`agentic_ip_reuse/adapter_templates/`). Provide parameterized Jinja2 (or plain Python f-string) templates for the most common glue patterns:
   - `valid_ready_to_push_pop.sv.j2` — wraps a push/pop FIFO in a valid-ready interface
   - `width_pad.sv.j2` — zero-pads or truncates a signal to match widths
   - `active_low_rst.sv.j2` — inverts reset polarity
   - `axi_lite_to_regfile.sv.j2` — bridges AXI-lite to a flat register array
2. **Add a `generate_adapter` tool** in `tools.py` that takes `adapter_type`, `parameters` (dict), and `file_path`, renders the template, and writes the `.sv` file.
3. **Update `evaluate_ip_candidate`** in `repository.py`: when a criterion fails, classify which adapter type would fix it and set `required_adapter_type` in the `IpAssessment`.
4. **Update the system prompt** to instruct the agent to call `generate_adapter` for each `required_adapter_type` before writing the integration plan.

### Key Files
- `agentic_ip_reuse/adapter_templates/` — new directory with `.sv.j2` templates
- `agentic_ip_reuse/tools.py` — new `generate_adapter` tool schema + executor method
- `agentic_ip_reuse/repository.py` — `score()` adds `required_adapter_type` to `IpAssessment`
- `agentic_ip_reuse/types.py` — add `required_adapter_type: Optional[str]` to `IpAssessment`

### Dependencies
- `jinja2` for template rendering (or use Python f-strings for zero extra deps)

### Complexity: M

---

## Improvement 10 — PPA-Aware Scoring with Synthesis Estimates

### Problem
`criteria_scores["synthesis_support"]` is currently a keyword-match score on a prose string (`repository.py:66`). No actual area, timing, or power data informs IP selection. For Realbench, choosing a 200 MHz IP when the spec demands 500 MHz is silently wrong.

### Implementation Plan
1. **Add a catalog ingestion pipeline** (`scripts/ingest_catalog.py`). For each IP in the catalog:
   - Run `yosys -p "synth; stat" <ip>.sv` to get gate-count and maximum frequency estimate.
   - Store `{ "lut_count": 128, "max_freq_mhz": 350, "slack_ns": 1.2 }` in the catalog JSON under a new `synthesis_metrics` field.
2. **Extend `IpCandidate`** with `synthesis_metrics: Dict[str, float]` in `types.py`.
3. **Update `score()`** in `repository.py`: compare `synthesis_metrics["max_freq_mhz"]` against the `ppa_targets` from `module_requirements`. Award 1.0 for meeting the target, 0.5 for within 20%, 0.0 for failing by >20%.
4. **Update `evaluate_ip_candidate` tool result** to include the PPA comparison values so the agent can reason about them explicitly.

### Key Files
- `scripts/ingest_catalog.py` — new: runs Yosys on each IP, writes metrics back to catalog JSON
- `agentic_ip_reuse/types.py` — add `synthesis_metrics` to `IpCandidate`
- `agentic_ip_reuse/repository.py` — `score()` uses `synthesis_metrics` for `synthesis_support` criterion

### Dependencies
- `yosys` (open-source synthesis, `apt install yosys`) for the ingestion script
- Optional: OpenROAD for more accurate timing on ASIC flows

### Complexity: L

---

## Improvement 11 — Incremental Checkpointing and Partial Resume

### Problem
Long runs (20+ minutes for Realbench) fail at step 14 and all intermediate decisions are lost. The LLM conversation history (`messages` list in `agent.py`) is only kept in memory.

### Implementation Plan
1. **Add a `CheckpointStore` class** (`agentic_ip_reuse/checkpoint.py`). After every tool call, serialize `messages` + `structured_plan_so_far` to `<output_dir>/checkpoint.json`. Use atomic write (write to `.tmp`, then rename) to avoid corruption on crash.
2. **Add a `--resume` CLI flag.** When set, `cmd_run` loads `checkpoint.json`, restores the `messages` list, and passes it to the agent's loop which resumes from step `len(previous_tool_calls) + 1`.
3. **Add `AgentConfig.checkpoint_every: int = 1` field** (checkpoint after every tool call by default; set to 0 to disable).
4. **Integrate with hierarchical mode:** each sub-agent writes its own `checkpoint.json` in its sub-output-directory.

### Key Files
- `agentic_ip_reuse/checkpoint.py` — new: `CheckpointStore`, `save()`, `load()`
- `agentic_ip_reuse/agent.py` — call `checkpoint.save()` after each tool result in the main loop; accept `initial_messages` parameter for resume
- `agentic_ip_reuse/cli.py` — `--resume` flag loads checkpoint before calling `agent.run()`

### Dependencies
None (stdlib `json` + `os.replace` for atomic writes).

### Complexity: S

---

## Improvement 12 — Testbench Skeleton and SVA Assertion Generation

### Problem
`_verification_md()` in `artifacts.py` produces prose bullets. No runnable testbench or assertion files are generated, so verification planning never connects to actual simulation infrastructure.

### Implementation Plan
1. **Add a `generate_testbench` tool** in `tools.py`. Given a `module_path` (the generated `.sv` file) and an optional `interface_type` (`valid_ready`, `axi_lite`, `push_pop`), it:
   - Parses the module's ports using `verilog_tools._parse_ports()`
   - Fills a testbench template that drives all inputs with clock-aligned stimulus, monitors outputs, and includes a `$finish` after a timeout
   - Writes `<module_name>_tb.sv`
2. **Add SVA assertion generation.** For known interfaces, emit a separate `<module_name>_assertions.sv` with:
   - Valid-ready: `assert property (@(posedge clk) valid |-> ##[1:$] ready);`
   - AXI-lite: standard AW/W/B channel handshake assertions
3. **Add a `run_simulation` tool** (optional, low priority) that shells out to `iverilog + vvp` on the tb file, captures pass/fail, and returns stdout.
4. **Update the system prompt** to call `generate_testbench` for every module whose RTL was generated by `generate_rtl_module`.

### Key Files
- `agentic_ip_reuse/verilog_tools.py` — add `generate_testbench()` and `generate_assertions()`
- `agentic_ip_reuse/tools.py` — new `generate_testbench` tool schema + executor method
- `agentic_ip_reuse/prompts.py` — add step 9 to system prompt for testbench generation

### Dependencies
- Optional: `iverilog + vvp` for simulation execution

### Complexity: M

---

## Improvement 13 — Design Hierarchy Memory Store

### Problem
With 20 modules and 16 steps per agent, early design decisions (clock domain, chosen data width, top-level bus protocol) drift out of the LLM's effective attention window. Later modules may contradict earlier choices. This is especially acute in hierarchical mode where each sub-agent starts with no knowledge of sibling decisions.

### Implementation Plan
1. **Add a `DesignStateStore` class** (`agentic_ip_reuse/design_state.py`). Backed by a `design_state.json` file in `output_dir`, it stores:
   ```json
   {
     "confirmed_clock_mhz": 500,
     "confirmed_data_width": 32,
     "selected_ips": { "Buffer": "sync_fifo", "Control": "axi_lite_reg_bank" },
     "interface_widths": { "data_bus": 32 },
     "unresolved": ["reset polarity TBD"]
   }
   ```
2. **Inject the state store summary** as a compact JSON block at the top of every `build_user_prompt()` call. The agent reads the current confirmed decisions before generating new ones.
3. **Add a `commit_design_decision` tool** in `tools.py`. The agent calls this after each confirmed decision (IP selection, parameter binding) to persist it to the store. This makes state updates explicit in the agent's reasoning trace.
4. **Share the store across sub-agents** in hierarchical mode by passing `output_dir` of the root agent to all sub-executors, so all levels read and write the same `design_state.json`.

### Key Files
- `agentic_ip_reuse/design_state.py` — new: `DesignStateStore`, `commit()`, `snapshot()`
- `agentic_ip_reuse/tools.py` — new `commit_design_decision` tool schema + executor method
- `agentic_ip_reuse/prompts.py` — `build_user_prompt()` accepts optional `state_snapshot: str`
- `agentic_ip_reuse/hierarchical.py` — `_make_executor()` shares root `output_dir` for state

### Dependencies
None.

### Complexity: S

---

## Priority and Sequencing

| # | Improvement | Complexity | Recommended Order |
|---|---|---|---|
| 11 | Incremental Checkpointing | S | 1 — enables safe iteration on long runs |
| 8 | Constraint Extraction Stage | S | 2 — improves every downstream decision |
| 13 | Design Hierarchy Memory Store | S | 3 — prevents incoherence in hierarchical runs |
| 3 | Long Spec Chunking + RAG | M | 4 — enables Realbench-scale input |
| 1 | Embedding-Based Search | M | 5 — improves IP discovery quality |
| 9 | Adapter Auto-Generation | M | 6 — reduces manual integration work |
| 12 | Testbench Skeleton Generation | M | 7 — closes the verification loop |
| 7 | Multi-Agent Parallelism | M | 8 — speed optimization after correctness is solid |
| 10 | PPA-Aware Scoring | L | 9 — requires Yosys setup and catalog re-ingestion |

Start with the three S-complexity items (11, 8, 13) as they are self-contained, have no new dependencies, and immediately harden the existing pipeline before adding new capabilities.
