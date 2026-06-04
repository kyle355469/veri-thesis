#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/auto_vllm_deploy.sh [options]

Launch an OpenAI-compatible vLLM server in the background, auto-pick an open
port when needed, wait for readiness, and print the endpoint for clients.

Options:
  --model MODEL              Hugging Face/local model path.
  --served-name NAME         Model name clients should request.
  --host HOST                Bind host for vLLM. Default: 0.0.0.0
  --request-host HOST        Host printed for client requests. Default: 127.0.0.1
  --port PORT                Port to use. Default: first open port from 8000.
  --port-start PORT          First port to try when --port is omitted. Default: 8000
  --port-search-count N      Number of ports to scan. Default: 100
  --venv PATH                Optional venv whose bin/activate should be sourced.
  --log-dir DIR              Directory for logs and PID files. Default: runs/vllm
  --wait-timeout-s SECONDS   Readiness timeout. Default: 900
  --no-tool-calling          Disable vLLM auto tool-choice flags.
  --extra-arg ARG            Extra argument passed to vLLM; repeatable.
  -h, --help                 Show this help.

Environment defaults:
  MODEL, SERVED_NAME, HOST, REQUEST_HOST, PORT, VLLM_VENV, HF_HOME,
  ENABLE_TOOL_CALLING, TOOL_CALL_PARSER, CHAT_TEMPLATE, DTYPE,
  GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, TENSOR_PARALLEL_SIZE

Example:
  scripts/auto_vllm_deploy.sh --model AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507
EOF
}

MODEL="${MODEL:-AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507}"
SERVED_NAME="${SERVED_NAME:-siliconmind-server}"
HOST="${HOST:-0.0.0.0}"
REQUEST_HOST="${REQUEST_HOST:-127.0.0.1}"
PORT="${PORT:-}"
PORT_START="${PORT_START:-8000}"
PORT_SEARCH_COUNT="${PORT_SEARCH_COUNT:-100}"
VLLM_VENV="${VLLM_VENV:-}"
LOG_DIR="${LOG_DIR:-runs/vllm}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-900}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --served-name)
      SERVED_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --request-host)
      REQUEST_HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --port-start)
      PORT_START="$2"
      shift 2
      ;;
    --port-search-count)
      PORT_SEARCH_COUNT="$2"
      shift 2
      ;;
    --venv)
      VLLM_VENV="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --wait-timeout-s)
      WAIT_TIMEOUT_S="$2"
      shift 2
      ;;
    --no-tool-calling)
      ENABLE_TOOL_CALLING=0
      shift
      ;;
    --extra-arg)
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -n "$VLLM_VENV" ]; then
  if [ ! -f "$VLLM_VENV/bin/activate" ]; then
    echo "ERROR: venv activate script not found: $VLLM_VENV/bin/activate" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VLLM_VENV/bin/activate"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not available on PATH." >&2
  echo "Set --venv /path/to/vllm-env or activate an environment with vllm installed." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to find an open port and check readiness." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

if [ -z "${TOOL_CALL_PARSER:-}" ]; then
  case "$MODEL" in
    Qwen/*|*Qwen*|*qwen*) TOOL_CALL_PARSER=qwen3_xml ;;
    *) TOOL_CALL_PARSER=hermes ;;
  esac
fi

if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
  TENSOR_PARALLEL_SIZE=1
fi

if [ -z "$PORT" ]; then
  PORT="$(
    PORT_START="$PORT_START" PORT_SEARCH_COUNT="$PORT_SEARCH_COUNT" python3 - <<'PY'
import os
import socket
import sys

start = int(os.environ["PORT_START"])
count = int(os.environ["PORT_SEARCH_COUNT"])
for port in range(start, start + count):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            continue
        print(port)
        sys.exit(0)
raise SystemExit(f"no open port found from {start} to {start + count - 1}")
PY
  )"
fi

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/vllm-${PORT}-${STAMP}.log"
PID_FILE="$LOG_DIR/vllm-${PORT}.pid"
BASE_URL="http://${REQUEST_HOST}:${PORT}/v1"

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: $PID_FILE points to a running process: $OLD_PID" >&2
    echo "Stop it first or choose another --port." >&2
    exit 1
  fi
fi

TOOL_ARGS=()
if [ "$ENABLE_TOOL_CALLING" = "1" ]; then
  TOOL_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
  if [ -n "$CHAT_TEMPLATE" ]; then
    TOOL_ARGS+=(--chat-template "$CHAT_TEMPLATE")
  fi
fi

TP_ARGS=()
if [ "$TENSOR_PARALLEL_SIZE" != "1" ]; then
  TP_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi

echo "Starting vLLM..."
echo "  model       : $MODEL"
echo "  served name : $SERVED_NAME"
echo "  bind        : $HOST:$PORT"
echo "  endpoint    : $BASE_URL"
echo "  log         : $LOG_FILE"

nohup vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${TP_ARGS[@]}" \
  "${TOOL_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"

cleanup_on_failure() {
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
  fi
}
trap cleanup_on_failure ERR

echo "Waiting for vLLM readiness..."
if ! BASE_URL="$BASE_URL" WAIT_TIMEOUT_S="$WAIT_TIMEOUT_S" python3 - <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

base_url = os.environ["BASE_URL"].rstrip("/")
deadline = time.time() + int(os.environ["WAIT_TIMEOUT_S"])
url = f"{base_url}/models"
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [item.get("id") for item in payload.get("data", []) if item.get("id")]
        print(",".join(names))
        sys.exit(0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        last_error = str(exc)
        time.sleep(5)

print(f"server did not become ready before timeout: {last_error}", file=sys.stderr)
sys.exit(1)
PY
then
  echo "ERROR: vLLM did not become ready. Last log lines:" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi

trap - ERR

cat <<EOF

vLLM is ready.

Endpoint:
  $BASE_URL

Client environment:
  export VLLM_BASE_URL=$BASE_URL
  export VLLM_MODEL=$SERVED_NAME
  export VLLM_API_KEY=$VLLM_API_KEY

Process:
  pid: $PID
  pid file: $PID_FILE
  log: $LOG_FILE

Stop server:
  kill $PID
EOF
