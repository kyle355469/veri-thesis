#!/usr/bin/env bash
set -e
MODEL="${MODEL:-openai/gpt-oss-20b}"
# MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
# MODEL="${MODEL:-zhuyaoyu/CodeV-R1-RL-Qwen-7B}"
# MODEL="${MODEL:-AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507}"
SERVED_NAME="${SERVED_NAME:-siliconmind-server}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
if [ -z "${TOOL_CALL_PARSER:-}" ]; then
  case "$MODEL" in
    Qwen/*|*Qwen*|*qwen*) TOOL_CALL_PARSER=qwen3_xml ;;
    *) TOOL_CALL_PARSER=hermes ;;
  esac
fi
if [ -z "${TENSOR_PARALLEL_SIZE:-}" ]; then
  case "$MODEL" in
    Qwen/Qwen3-4B-Thinking-2507-FP8) TENSOR_PARALLEL_SIZE=4 ;;
    *) TENSOR_PARALLEL_SIZE=4 ;;
  esac
fi

# Optional: Hugging Face cache path
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"


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
echo "Starting vLLM on ${HOST}:${PORT}"
echo "Model: ${MODEL}"
echo "Served model name: ${SERVED_NAME}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
vllm serve "$MODEL" \
  --max-model-len 131072 \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${TOOL_ARGS[@]}"
