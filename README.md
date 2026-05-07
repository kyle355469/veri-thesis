# RAG RTL Generation Prototype

This repository contains a thesis-oriented prototype for RAG-assisted RTL code generation. It follows the deployment diagram in `thesis-Deployment.png`: CPU-side retrieval, reranking, context parsing, semantic history cache, syntax/lint verification, and GPU-side local vLLM inference.

## Components

- `rag_rtl/config.py`: small dataclass configuration objects for cache, runtime paths, tool calling, and the fixed-pipe Yosys stage.
- `rag_rtl/dataset.py`: normalizes `merged.jsonl`, removes private reasoning traces, and extracts final RTL code.
- `rag_rtl/vector_store.py`: stores embedded RTL documents as `vectors.npy` plus `documents.jsonl`.
- `rag_rtl/datapath.py`: runs Yosys during preprocessing, converts RTL into module-level datapath graphs, and builds an extra graph-wise VectorDB.
- `rag_rtl/retrieval.py`: vector retrieval plus a lexical RTL reranker.
- `rag_rtl/prompting.py`: builds answer-only RTL prompts with retrieved examples and verification diagnostics.
- `rag_rtl/llm.py`: OpenAI-compatible vLLM client using `VLLM_BASE_URL`, `VLLM_MODEL`, and `VLLM_API_KEY`.
- `rag_rtl/verifier.py`: runs `yosys` and `verilator --lint-only` when installed.
- `rag_rtl/pipeline.py`: end-to-end keyword semantic cache, RAG, generation, verification, repair, failed-attempt logging, and monitoring flow.
- `rag_rtl/json_utils.py`: shared JSON serialization, JSONL append, and text-preview helpers used by reports, monitoring, tool calls, and scripts.

The core pipelines still accept their original keyword arguments, but new modules should prefer `CacheConfig`, `RuntimeConfig`, `ToolCallingConfig`, and `FixedPipeConfig`. That keeps constructors stable as more retrieval, verification, or graph-processing stages are added.

## Setup

Use Python 3. The local `python` command may point to Python 2.7, so prefer `python3`.

```bash
python3 -m pip install -r requirements.txt
```

For full generation and verification, install:

- `vllm` and a local RTL-capable model served with an OpenAI-compatible endpoint.
- `yosys`
- `verilator`

## Website Setup

The local web UI wraps the same generation and fixed-pipe pipelines as the CLI, but makes the prompt, settings, tag filters, and per-attempt model outputs visible in one browser page.

Start the vLLM server first, or point the client at an existing OpenAI-compatible endpoint:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=siliconmind-server
export VLLM_API_KEY=EMPTY
```

Then run the website from the repository root:

```bash
python3 scripts/web_app.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The website automatically discovers indexes under `indexes/` that contain both `documents.jsonl` and `vectors.npy`. It also reads all document tags from the selected index and shows them as selectable options. Selected tags filter the retrieval store before generation; use `Any` to keep documents matching at least one selected tag, or `All` to require every selected tag.

Useful controls:

| Control | Meaning |
| --- | --- |
| `Pipeline` | Choose normal `generate` or thesis `fixed_pipe`. |
| `Index` | VectorDB used by normal generation. |
| `Spec index` | First-stage spec/text VectorDB for fixed-pipe mode. |
| `Structure index` | Graph/code-structure VectorDB for fixed-pipe mode. |
| `Tags` | Retrieval filter options loaded from the selected index. |
| `Tag target` | Choose whether tags filter the primary store, structure store, both stores, or neither. |
| `Repair attempts` | Number of verification-feedback retries after the first generated RTL. |
| `Enable tool calling` | Lets a tool-compatible vLLM endpoint call local retrieval and RTL verification tools. |

Each run displays:

- The raw model text returned in each generation attempt.
- The extracted RTL used for verification.
- Pass/fail status for each attempt.
- The full JSON report, including cache, retrieval, verification, timings, and final RTL.

## CLI Reference

Run all commands from the repository root. The main entrypoint is:

```bash
python3 -m rag_rtl.cli {index,datapath-index,fixed-pipe,generate,evaluate} [options]
```

### `index`

Build a vector index from a JSONL corpus.

| Option | Default | Meaning |
| --- | --- | --- |
| `--corpus CORPUS` | `merged.jsonl` | Input JSONL corpus. |
| `--output OUTPUT` | `indexes/rtl_hash` | Output directory for `vectors.npy` and `documents.jsonl`. |
| `--embedder EMBEDDER` | `hash` | Embedder name, either `hash` or a supported sentence-transformers model name. |
| `--limit LIMIT` | none | Index only the first `LIMIT` records. |

Example:

```bash
python3 -m rag_rtl.cli index \
  --corpus merged.jsonl \
  --output indexes/rtl_hash \
  --embedder hash \
  --limit 1000
```

### `datapath-index`

Pre-serve graph retrievable data by synthesizing each Verilog solution through Yosys, storing structured datapath graphs, and embedding graph summaries into a second VectorDB.

| Option | Default | Meaning |
| --- | --- | --- |
| `--corpus CORPUS` | `merged.jsonl` | Input JSONL corpus. |
| `--output OUTPUT` | `indexes/rtl_datapath_hash` | Output directory for `datapaths.jsonl`, `documents.jsonl`, `vectors.npy`, and `failures.jsonl`. |
| `--embedder EMBEDDER` | `hash` | Embedder name, either `hash` or a supported sentence-transformers model name. |
| `--limit LIMIT` | none | Process only the first `LIMIT` records. |
| `--yosys-bin YOSYS_BIN` | `yosys` | Yosys executable name or path. |
| `--timeout-s TIMEOUT_S` | `30` | Per-document Yosys timeout in seconds. |

Example:

```bash
python3 -m rag_rtl.cli datapath-index \
  --corpus merged.jsonl \
  --output indexes/rtl_datapath_hash \
  --embedder hash \
  --limit 1000
```

### `fixed-pipe`

Run the thesis fixed pipeline shown in `thesis-fixed-pipe-flow.png`:

1. Retrieve specification examples from `VectorDB (Spec)`.
2. Ask the reasoning LLM for first-edition code.
3. Verify with Yosys/Verilator, feeding diagnostics back to the LLM when verification fails.
4. If verification passes, build a Yosys datapath graph from the first edition.
5. Retrieve graph-wise examples from `VectorDB (Code Structure)`.
6. Ask the reasoning LLM for second-edition code and verify it.

| Option | Default | Meaning |
| --- | --- | --- |
| `--spec-index SPEC_INDEX` | `indexes/rtl_hash` | Spec/text VectorDB built by `index`. |
| `--code-structure-index CODE_STRUCTURE_INDEX` | `indexes/rtl_datapath_hash` | Graph-wise VectorDB built by `datapath-index`. |
| `--structure-retrieve-k STRUCTURE_RETRIEVE_K` | `8` | Number of graph VectorDB hits before reranking. |
| `--structure-context-k STRUCTURE_CONTEXT_K` | `4` | Number of reranked graph examples sent to the LLM. |
| `--second-edition-repair-attempts SECOND_EDITION_REPAIR_ATTEMPTS` | `1` | Repair attempts for second-edition verification failures. |
| `--yosys-bin YOSYS_BIN` | `yosys` | Yosys executable used for the graph build. |
| `--yosys-timeout-s YOSYS_TIMEOUT_S` | `30` | Yosys graph-build timeout in seconds. |

Example:

```bash
python3 -m rag_rtl.cli fixed-pipe \
  --spec-index indexes/rtl_hash \
  --code-structure-index indexes/rtl_datapath_hash \
  --prompt "Design a Verilog inverter module named invert with input i and output o." \
  --max-repair-attempts 2 \
  --second-edition-repair-attempts 1 \
  --json-report runs/latest_report.json
```

### `generate`

Generate one RTL answer with retrieval, semantic cache, verification, repair, and reporting.

| Option | Default | Meaning |
| --- | --- | --- |
| `--index INDEX` | `indexes/rtl_hash` | Vector index directory to load. |
| `--embedder EMBEDDER` | `hash` | Embedder used for retrieval and cache lookup. |
| `--prompt PROMPT` | none | Inline user prompt. Required unless `--prompt-file` is used. |
| `--prompt-file PROMPT_FILE` | none | File containing the user prompt. Required unless `--prompt` is used. |
| `--target-hdl TARGET_HDL` | `verilog` | HDL language tag used in prompts. |
| `--module-signature MODULE_SIGNATURE` | none | Optional expected module signature/interface. |
| `--constraint CONSTRAINT` | `[]` | Extra constraint. Can be repeated. |
| `--retrieve-k RETRIEVE_K` | `8` | Number of VectorDB hits before reranking. |
| `--context-k CONTEXT_K` | `4` | Number of reranked examples passed to the LLM. |
| `--max-repair-attempts MAX_REPAIR_ATTEMPTS` | `1` | Number of verification-feedback repair attempts after the first generation. |
| `--cache CACHE` | `data/history_cache.json` | History semantic cache JSON path. |
| `--monitor MONITOR` | `runs/monitor.jsonl` | Monitor/verbose event JSONL path. |
| `--cache-mode {keywords,direct}` | `keywords` | `keywords` asks the LLM for Verilog-focused keywords, filters history by keyword overlap, then scores only those candidates; `direct` scores every cached prompt. |
| `--cache-reuse-threshold CACHE_REUSE_THRESHOLD` | `0.95` | Similarity needed to directly reuse cached RTL. |
| `--cache-evidence-threshold CACHE_EVIDENCE_THRESHOLD` | `0.88` | Similarity needed to pass a cache entry as LLM evidence. |
| `--failed-log FAILED_LOG` | `runs/failed_attempts.jsonl` | JSONL file for generated attempts that fail verification. |
| `--verbose-generation` | off | Print and log prompts, raw model text, extracted RTL, and diagnostics. |
| `--enable-tool-calling` | off | Let vLLM call local RTL tools during generation. Available tools are retrieval, Yosys, Verilator, and full verification. |
| `--tool-choice TOOL_CHOICE` | `auto` | vLLM `tool_choice` value, such as `auto` or `required`. |
| `--max-tool-rounds MAX_TOOL_ROUNDS` | `4` | Maximum tool-call/result turns before forcing a final answer. |
| `--testbench TESTBENCH` | none | Optional external testbench path. |
| `--top-module TOP_MODULE` | none | Optional top module for Yosys hierarchy and test command placeholder. |
| `--test-command TEST_COMMAND` | none | External verifier command template using `{rtl}`, `{testbench}`, and `{top}`. |
| `--json-report JSON_REPORT` | none | Write a readable JSON report with `summary`, `task`, `llm_actions`, `cache`, `retrieval`, `verification`, `timings`, and `rtl` sections. |

Generation uses these vLLM environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible vLLM endpoint. |
| `VLLM_MODEL` | `siliconmind-server` | Served model name. |
| `VLLM_API_KEY` | `EMPTY` | API key sent to the endpoint. |

For automatic tool calling, start vLLM with a tool-compatible parser and chat template. The bundled server script accepts:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENABLE_TOOL_CALLING` | `0` | Set to `1` to add vLLM auto tool-choice flags. |
| `TOOL_CALL_PARSER` | `hermes` | Parser passed to `--tool-call-parser`; choose the parser that matches your model. |
| `CHAT_TEMPLATE` | none | Optional explicit `--chat-template` value. Leave unset unless the model requires one. |

Example:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=siliconmind-server

python3 -m rag_rtl.cli generate \
  --index indexes/rtl_hash \
  --prompt 'The Verilog code implements a hardware module for generating the propagate (`o_p`) and generate (`o_g`) signals in a binary adder. It computes `o_p` as the XOR of the inputs (`i_a` and `i_b`) and `o_g` as the AND of the same inputs.' \
  --target-hdl verilog \
  --module-signature "module invert(input i, output o);" \
  --constraint "Use a continuous assignment." \
  --retrieve-k 8 \
  --context-k 4 \
  --max-repair-attempts 3 \
  --cache-mode keywords \
  --cache-reuse-threshold 0.95 \
  --cache-evidence-threshold 0.88 \
  --json-report runs/latest_report.json
```

Example with vLLM tool calling:

```bash
ENABLE_TOOL_CALLING=1 TOOL_CALL_PARSER=hermes bash vllm_deploy.sh

python3 -m rag_rtl.cli generate \
  --index indexes/rtl_hash \
  --prompt 'The Verilog code defines a module named `VCC` that outputs a constant high logic level (1). The output `V` is always set to 1.' \
  --target-hdl verilog \
  --max-repair-attempts 3 \
  --cache-mode keywords \
  --enable-tool-calling \
  --json-report runs/latest_report.json
```

Example with an external testbench:

```bash
python3 -m rag_rtl.cli generate \
  --index indexes/rtl_hash \
  --prompt "Design module invert with input i and output o where o is not i." \
  --top-module invert \
  --testbench tests/testbenches/invert_tb.v \
  --test-command "iverilog -o /tmp/invert_tb.out {rtl} {testbench}" \
  --json-report runs/latest_report.json
```

### `evaluate`

Run a JSONL prompt set through one baseline mode.

| Option | Default | Meaning |
| --- | --- | --- |
| `--tasks TASKS` | required | JSONL file with at least a `prompt` field per row. |
| `--index INDEX` | `indexes/rtl_hash` | Vector index directory to load. |
| `--embedder EMBEDDER` | `hash` | Embedder used by retrieval/cache. |
| `--mode {llm_only,rag,rag_cache_verify}` | `rag_cache_verify` | Baseline mode. |
| `--output OUTPUT` | `runs/evaluation.json` | Summary and per-task records JSON. |
| `--cache-mode {keywords,direct}` | `keywords` | Cache workflow for `rag_cache_verify`; keyword mode uses LLM-extracted Verilog keywords as the candidate gate before similarity. |
| `--cache-reuse-threshold CACHE_REUSE_THRESHOLD` | `0.95` | Similarity needed to directly reuse cached RTL. |
| `--cache-evidence-threshold CACHE_EVIDENCE_THRESHOLD` | `0.88` | Similarity needed to pass a cache entry as evidence. |

Example:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=siliconmind-server

printf '%s\n' \
  '{"prompt":"Design a Verilog inverter module named invert with input i and output o."}' \
  '{"prompt":"Design a 2-input and gate module named and2."}' \
  > /tmp/rtl_eval_prompts.jsonl

python3 -m rag_rtl.cli evaluate \
  --tasks /tmp/rtl_eval_prompts.jsonl \
  --index indexes/rtl_hash \
  --mode rag_cache_verify \
  --cache-mode keywords \
  --output runs/evaluation.json
```

## Script Examples

The files under `scripts/` are thin wrappers around the same CLI, plus a concurrent request driver. They are not marked executable in this checkout, so the examples call them with `python3`. Run `bash vllm_deploy.sh` for the server script.

### `vllm_deploy.sh`

Start the local OpenAI-compatible vLLM server on port `8000`:

```bash
bash vllm_deploy.sh
```

### `scripts/index_corpus.py`

Equivalent to `python3 -m rag_rtl.cli index ...`:

```bash
python3 scripts/index_corpus.py \
  --corpus merged.jsonl \
  --output indexes/rtl_hash \
  --embedder hash \
  --limit 1000
```

### `scripts/build_datapath_index.py`

Equivalent to `python3 -m rag_rtl.cli datapath-index ...`:

```bash
python3 scripts/build_datapath_index.py \
  --corpus merged.jsonl \
  --output indexes/rtl_datapath_hash \
  --embedder hash \
  --limit 1000
```

### `scripts/generate_rtl.py`

Equivalent to `python3 -m rag_rtl.cli generate ...`:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=siliconmind-server

python3 scripts/generate_rtl.py \
  --index indexes/rtl_hash \
  --prompt "Design a Verilog module named invert with input i and output o where o is not i." \
  --max-repair-attempts 1 \
  --cache-mode keywords \
  --json-report runs/latest_report.json
```

### `scripts/evaluate_rtl.py`

Equivalent to `python3 -m rag_rtl.cli evaluate ...`:

```bash
printf '%s\n' \
  '{"prompt":"Design a Verilog inverter module named invert with input i and output o."}' \
  > /tmp/rtl_eval_prompts.jsonl

python3 scripts/evaluate_rtl.py \
  --tasks /tmp/rtl_eval_prompts.jsonl \
  --index indexes/rtl_hash \
  --mode rag_cache_verify \
  --output runs/evaluation.json
```

### `scripts/concurrent_generate.py`

Keep vLLM busy with multiple simultaneous generation requests:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=siliconmind-server

printf '%s\n' \
  '{"prompt":"Design a Verilog inverter module named invert with input i and output o."}' \
  '{"prompt":"Design a 2-input xor gate named xor2."}' \
  '{"prompt":"Design a 4-bit ripple-carry adder named add4."}' \
  > /tmp/rtl_concurrent_prompts.jsonl

python3 scripts/concurrent_generate.py \
  --tasks /tmp/rtl_concurrent_prompts.jsonl \
  --index indexes/rtl_hash \
  --concurrency 3 \
  --output runs/concurrent_generation.jsonl \
  --cache-mode keywords \
  --failed-log runs/failed_attempts.jsonl
```

## Fixed-Pipe Correctness Smoke

The deterministic unit test covers the complete diagram path without a model server:

```bash
python3 -m unittest tests.test_fixed_pipe -v
```

To run the real vLLM-backed smoke:

```bash
python3 -m rag_rtl.cli index \
  --corpus merged.jsonl \
  --output indexes/rtl_hash \
  --embedder hash \
  --limit 1000

python3 -m rag_rtl.cli datapath-index \
  --corpus merged.jsonl \
  --output indexes/rtl_datapath_hash \
  --embedder hash \
  --limit 1000

bash vllm_deploy.sh
```

In another shell:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=siliconmind-server

python3 -m rag_rtl.cli fixed-pipe \
  --spec-index indexes/rtl_hash \
  --code-structure-index indexes/rtl_datapath_hash \
  --prompt "Design a Verilog module named invert with input i and output o where o is not i." \
  --top-module invert \
  --max-repair-attempts 2 \
  --second-edition-repair-attempts 1 \
  --json-report runs/latest_report.json
```

## Evaluation Baselines

Use the same prompt set across these modes:

- LLM only: run with an empty or disabled retrieval store.
- RAG + LLM: use retrieval/reranking, disable cache reuse.
- RAG + semantic cache + verification feedback: default pipeline.

Recommended metrics: syntax pass rate, lint pass rate, repair success rate, retrieval relevance, cache hit accuracy, and latency by stage.

Run a baseline evaluation with a JSONL file containing one prompt per row:

```json
{"prompt": "Design a Verilog inverter module named invert with input i and output o."}
```

```bash
python3 -m rag_rtl.cli evaluate \
  --tasks data/eval_prompts.jsonl \
  --index indexes/rtl_hash \
  --mode rag_cache_verify \
  --output runs/evaluation.json
```

Supported modes are `llm_only`, `rag`, and `rag_cache_verify`. The semantic cache uses LLM keyword prefiltering by default, so cosine similarity is computed only for keyword-matched history entries; pass `--cache-mode direct` to evaluate direct cosine matching over the full cache.
