#!/usr/bin/env bash
set -e

MODEL="AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507"
SERVED_NAME="siliconmind-server"
PORT=8000
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

# Optional: choose GPU
export CUDA_VISIBLE_DEVICES=0

# Optional: Hugging Face cache path
export HF_HOME="$HOME/.cache/huggingface"


TOOL_ARGS=()
if [ "$ENABLE_TOOL_CALLING" = "1" ]; then
  TOOL_ARGS=(
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
  )
  if [ -n "$CHAT_TEMPLATE" ]; then
    TOOL_ARGS+=(--chat-template "$CHAT_TEMPLATE")
  fi
fi

# Start OpenAI-compatible vLLM server.
# For tool calling, run with ENABLE_TOOL_CALLING=1 and set TOOL_CALL_PARSER
# to the parser recommended by your model/chat template. Set CHAT_TEMPLATE
# only when your model needs an explicit tool-use template.
vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization 0.93 \
  --max-model-len 131072 \
  "${TOOL_ARGS[@]}"
