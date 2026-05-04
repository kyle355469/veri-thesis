# RAG RTL Generation Prototype

This repository contains a thesis-oriented prototype for RAG-assisted RTL code generation. It follows the deployment diagram in `thiese-Deployment.png`: CPU-side retrieval, reranking, context parsing, semantic history cache, syntax/lint verification, and GPU-side local vLLM inference.

## Components

- `rag_rtl/dataset.py`: normalizes `merged.jsonl`, removes private reasoning traces, and extracts final RTL code.
- `rag_rtl/vector_store.py`: stores embedded RTL documents as `vectors.npy` plus `documents.jsonl`.
- `rag_rtl/retrieval.py`: vector retrieval plus a lexical RTL reranker.
- `rag_rtl/prompting.py`: builds answer-only RTL prompts with retrieved examples and verification diagnostics.
- `rag_rtl/llm.py`: OpenAI-compatible vLLM client using `VLLM_BASE_URL`, `VLLM_MODEL`, and `VLLM_API_KEY`.
- `rag_rtl/verifier.py`: runs `yosys` and `verilator --lint-only` when installed.
- `rag_rtl/pipeline.py`: end-to-end cache, RAG, generation, verification, repair, and monitoring flow.

## Setup

Use Python 3. The local `python` command may point to Python 2.7, so prefer `python3`.

```bash
python3 -m pip install -r requirements.txt
```

For full generation and verification, install:

- `vllm` and a local RTL-capable model served with an OpenAI-compatible endpoint.
- `yosys`
- `verilator`

## Build An Index

For a small smoke-test index:

```bash
python3 -m rag_rtl.cli index --corpus merged.jsonl --output indexes/rtl_hash --limit 1000
```

For the full corpus, omit `--limit`.

## Run Generation

Start vLLM separately, then configure the client:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=your-local-model-name
python3 -m rag_rtl.cli generate \
  --index indexes/rtl_hash \
  --prompt "Design a Verilog module named invert with input i and output o where o is not i." \
  --max-repair-attempts 1 \
  --json-report runs/latest_report.json
```

The generated RTL is printed to stdout. Logs are appended to `runs/monitor.jsonl`, and verified or explicitly failed attempts are saved in `data/history_cache.json`.

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

Supported modes are `llm_only`, `rag`, and `rag_cache_verify`.
