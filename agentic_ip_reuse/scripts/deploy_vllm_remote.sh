#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/deploy_vllm_remote.sh <ssh-host> [local-port] [remote-port]

Run this on your local computer. The script starts an OpenAI-compatible vLLM
server on the remote host with SSH, waits for it to become ready, and then keeps
an SSH tunnel open from local-port to the remote vLLM port.

Examples:
  bash scripts/deploy_vllm_remote.sh user@gpu-server
  bash scripts/deploy_vllm_remote.sh user@gpu-server 18000 8000

Common environment variables:
  MODEL=AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507
  SERVED_NAME=siliconmind-server
  VLLM_VENV=$HOME/.venvs/vllm
  REMOTE_LOG_DIR=$HOME/agentic_ip_reuse_runs/vllm
  CUDA_VISIBLE_DEVICES=0
  TENSOR_PARALLEL_SIZE=1
  MAX_MODEL_LEN=32768
  GPU_MEMORY_UTILIZATION=0.93
  ENABLE_TOOL_CALLING=1
  TOOL_CALL_PARSER=qwen3_xml
  CHAT_TEMPLATE=/path/to/template.jinja

After the tunnel is open:
  export VLLM_BASE_URL=http://localhost:8000/v1
  export VLLM_MODEL=siliconmind-server
  export VLLM_API_KEY=EMPTY
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
  usage
  exit 0
fi

SSH_HOST="$1"
LOCAL_PORT="${2:-8000}"
REMOTE_PORT="${3:-8000}"

MODEL="${MODEL:-AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507}"
SERVED_NAME="${SERVED_NAME:-siliconmind-server}"
VLLM_VENV="${VLLM_VENV:-}"
REMOTE_BIND_HOST="${REMOTE_BIND_HOST:-127.0.0.1}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
REMOTE_READY_TIMEOUT_S="${REMOTE_READY_TIMEOUT_S:-240}"

if [ -z "$TOOL_CALL_PARSER" ]; then
  case "$MODEL" in
    Qwen/*|*Qwen*|*qwen*) TOOL_CALL_PARSER=qwen3_xml ;;
    *) TOOL_CALL_PARSER=hermes ;;
  esac
fi

CONTROL_DIR="${TMPDIR:-/tmp}/agentic-ip-reuse-ssh"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"

SSH_OPTS=(
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath="${CONTROL_DIR}/%r@%h:%p"
)

if ( : >"/dev/tcp/127.0.0.1/${LOCAL_PORT}" ) >/dev/null 2>&1; then
  echo "Local port ${LOCAL_PORT} is already in use." >&2
  echo "Use another local port, for example:" >&2
  echo "  bash scripts/deploy_vllm_remote.sh ${SSH_HOST} 18000 ${REMOTE_PORT}" >&2
  echo "Then set:" >&2
  echo "  export VLLM_BASE_URL=http://localhost:18000/v1" >&2
  exit 1
fi

REMOTE_ENV=(
  "MODEL=$MODEL"
  "SERVED_NAME=$SERVED_NAME"
  "VLLM_VENV=$VLLM_VENV"
  "REMOTE_BIND_HOST=$REMOTE_BIND_HOST"
  "REMOTE_PORT=$REMOTE_PORT"
  "REMOTE_LOG_DIR=$REMOTE_LOG_DIR"
  "RUN_NAME=$RUN_NAME"
  "DTYPE=$DTYPE"
  "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
  "ENABLE_TOOL_CALLING=$ENABLE_TOOL_CALLING"
  "TOOL_CALL_PARSER=$TOOL_CALL_PARSER"
  "CHAT_TEMPLATE=$CHAT_TEMPLATE"
  "VLLM_API_KEY=$VLLM_API_KEY"
  "REMOTE_READY_TIMEOUT_S=$REMOTE_READY_TIMEOUT_S"
)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  REMOTE_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

quote_env() {
  local item
  for item in "${REMOTE_ENV[@]}"; do
    printf "%q " "$item"
  done
}

REMOTE_SCRIPT='
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$HOME/agentic_ip_reuse_runs/vllm}"
mkdir -p "$REMOTE_LOG_DIR"
LOG_FILE="$REMOTE_LOG_DIR/vllm-${RUN_NAME}.log"
PID_FILE="$REMOTE_LOG_DIR/vllm-${RUN_NAME}.pid"

if [ ! -x "$VLLM_VENV/bin/python" ]; then
  echo "Missing vLLM virtualenv python: $VLLM_VENV/bin/python" >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :$REMOTE_PORT )" | grep -q ":$REMOTE_PORT"; then
  echo "Remote port $REMOTE_PORT already appears to be in use." >&2
  exit 1
fi

TOOL_ARGS=()
if [ "$ENABLE_TOOL_CALLING" = "1" ]; then
  TOOL_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
  if [ -n "$CHAT_TEMPLATE" ]; then
    TOOL_ARGS+=(--chat-template "$CHAT_TEMPLATE")
  fi
fi

TP_ARGS=()
if [ -n "$TENSOR_PARALLEL_SIZE" ] && [ "$TENSOR_PARALLEL_SIZE" != "1" ]; then
  TP_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi

echo "Starting vLLM on ${REMOTE_BIND_HOST}:${REMOTE_PORT}"
echo "Model: $MODEL"
echo "Served model name: $SERVED_NAME"
echo "Log: $LOG_FILE"
echo "PID: $PID_FILE"

nohup "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host "$REMOTE_BIND_HOST" \
  --port "$REMOTE_PORT" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${TP_ARGS[@]}" \
  "${TOOL_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" >"$PID_FILE"

deadline=$((SECONDS + REMOTE_READY_TIMEOUT_S))
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "vLLM process exited early. Last log lines:" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if "$VLLM_VENV/bin/python" - "$REMOTE_BIND_HOST" "$REMOTE_PORT" <<'"'"'PY'"'"' >/dev/null 2>&1
import json
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=3) as response:
    payload = json.loads(response.read().decode("utf-8"))
if "data" not in payload:
    raise SystemExit(1)
PY
  then
    echo "READY $PID $LOG_FILE $PID_FILE"
    exit 0
  fi
  sleep 5
done

echo "Timed out waiting for vLLM readiness. Last log lines:" >&2
tail -n 80 "$LOG_FILE" >&2 || true
exit 1
'

echo "Starting remote vLLM on ${SSH_HOST}:${REMOTE_PORT}"
REMOTE_OUTPUT="$(
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$(quote_env) bash -s" <<<"$REMOTE_SCRIPT"
)"
echo "$REMOTE_OUTPUT"

PID="$(printf "%s\n" "$REMOTE_OUTPUT" | awk "/^READY / {print \$2; exit}")"
LOG_FILE="$(printf "%s\n" "$REMOTE_OUTPUT" | awk "/^READY / {print \$3; exit}")"
PID_FILE="$(printf "%s\n" "$REMOTE_OUTPUT" | awk "/^READY / {print \$4; exit}")"

cat <<EOF

Forwarding local port ${LOCAL_PORT} to vLLM on ${SSH_HOST}:${REMOTE_PORT}

Keep this terminal open. In another local terminal, use:
  export VLLM_BASE_URL=http://localhost:${LOCAL_PORT}/v1
  export VLLM_MODEL=${SERVED_NAME}
  export VLLM_API_KEY=${VLLM_API_KEY}

Remote process:
  pid file : ${PID_FILE:-unknown}
  log file : ${LOG_FILE:-unknown}
  stop     : ssh ${SSH_HOST} 'kill ${PID:-<pid>}'
EOF

exec ssh "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes -N -L "${LOCAL_PORT}:${REMOTE_BIND_HOST}:${REMOTE_PORT}" "$SSH_HOST"
