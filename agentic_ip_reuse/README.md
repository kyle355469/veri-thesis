# Agentic IP Reuse

`agentic_ip_reuse` is a dependency-free Python framework for planning IC circuit implementations around reusable IP blocks. It focuses on the agentic module: requirements capture, system decomposition, IP search/evaluation, interface understanding, parameterization, and simulation/synthesis/debug planning.

The future reused-IP description module is represented by the `IpRepository` protocol. This package ships a JSON-backed repository and a small example catalog so the flow can run now.

## Quick Start

```bash
cd /home/kai/veri-thesis/agentic_ip_reuse
python3 -m unittest discover tests
python3 -m agentic_ip_reuse.cli run \
  --prompt "Build a simple streaming FIR accelerator with reusable FIFO and AXI-lite control" \
  --mock-llm \
  --output-dir /tmp/agentic_ip_reuse_demo
```

If installed, the console entrypoint is:

```bash
agentic-ip-reuse run --prompt "Build a DMA subsystem with reusable FIFOs" --mock-llm
```

For a live model, configure an OpenAI-compatible vLLM endpoint:

```bash
VLLM_BASE_URL=http://localhost:8000/v1 \
VLLM_MODEL=siliconmind-server \
VLLM_API_KEY=EMPTY \
python3 -m agentic_ip_reuse.cli run --prompt-file spec.txt
```

## Remote vLLM Helper

To start vLLM on a remote GPU server without Slurm and forward it to your local machine:

```bash
cd /home/kai/veri-thesis/agentic_ip_reuse
bash scripts/deploy_vllm_remote.sh user@gpu-server 8000 8000
```

The script starts the remote model server, writes a remote PID/log file, waits for `/v1/models`, and keeps an SSH tunnel open. In another local terminal:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=siliconmind-server
export VLLM_API_KEY=EMPTY
```

Useful overrides:

```bash
MODEL=AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507 \
VLLM_VENV='$HOME/.venvs/vllm' \
CUDA_VISIBLE_DEVICES=0 \
TENSOR_PARALLEL_SIZE=1 \
bash scripts/deploy_vllm_remote.sh user@gpu-server 18000 8000
```

## Outputs

Each run writes:

- `requirements.md`
- `module_decomposition.md`
- `ip_reuse_matrix.md`
- `integration_plan.md`
- `verification_plan.md`
- `result.json`

The final JSON includes selected IPs, rejected alternatives, adapter/wrapper needs, unresolved assumptions, and simulation/synthesis/debug steps.
