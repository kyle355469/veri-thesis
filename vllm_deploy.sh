#!/usr/bin/env bash
set -e

MODEL="AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507"
SERVED_NAME="siliconmind-server"
PORT=8000

# Optional: choose GPU
export CUDA_VISIBLE_DEVICES=0

# Optional: Hugging Face cache path
export HF_HOME="$HOME/.cache/huggingface"


# Start OpenAI-compatible vLLM server
vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768