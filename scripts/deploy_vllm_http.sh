#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_vllm_http.sh [options]

Start an OpenAI-compatible vLLM HTTP server directly on this machine, or over
SSH on a remote GPU machine. For remote runs, the script keeps an SSH tunnel
open so clients can use http://localhost:<local-port>/v1.

Common options:
  --ssh-host HOST             Optional SSH host for the GPU machine.
  --model MODEL               Hugging Face model or local model path.
  --served-name NAME          Model name clients request. Default: siliconmind-server.
  --local-port PORT           Local forwarded/client port. Default: 8000.
  --remote-port PORT          Port used by vLLM on the GPU machine. Default: 8000.
  --bind-host HOST            Host vLLM binds on the GPU machine.
                              Default: 0.0.0.0 locally, 127.0.0.1 remotely.
  --vllm-venv DIR             Optional venv containing vLLM.
  --log-dir DIR               Log/PID directory. Default: runs/vllm locally,
                              $HOME/vllm_runs remotely.
  --attach                    Do not start vLLM; only wait and forward.
  --ready-timeout-s SECONDS   Time to wait for /v1/models. Default: 900.
  --no-tool-calling           Disable vLLM auto tool-choice flags.
  -h, --help                  Show this help.

Environment defaults:
  MODEL, SERVED_NAME, LOCAL_PORT, PORT, BIND_HOST, VLLM_VENV, LOG_DIR,
  CUDA_VISIBLE_DEVICES, TENSOR_PARALLEL_SIZE, MAX_MODEL_LEN,
  GPU_MEMORY_UTILIZATION, DTYPE, ENABLE_TOOL_CALLING, TOOL_CALL_PARSER,
  CHAT_TEMPLATE, HF_HOME, VLLM_API_KEY

Examples:
  # Start vLLM on the current GPU machine.
  scripts/deploy_vllm_http.sh --model openai/gpt-oss-20b

  # Start vLLM on a remote GPU machine and forward it locally.
  scripts/deploy_vllm_http.sh \
    --ssh-host user@gpu-machine \
    --model openai/gpt-oss-20b \
    --local-port 18000 \
    --remote-port 8000

  # Forward an already-running remote vLLM server.
  scripts/deploy_vllm_http.sh \
    --ssh-host user@gpu-machine \
    --attach \
    --local-port 18000 \
    --remote-port 8000
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SSH_HOST="${SSH_HOST:-}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
SERVED_NAME="${SERVED_NAME:-siliconmind-server}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-${PORT:-8000}}"
BIND_HOST="${BIND_HOST:-}"
VLLM_VENV="${VLLM_VENV:-}"
LOG_DIR="${LOG_DIR:-}"
ATTACH=0
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"

require_value() {
  if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
    echo "ERROR: $1 requires a value." >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ssh-host)
      require_value "$@"
      SSH_HOST="$2"
      shift 2
      ;;
    --model)
      require_value "$@"
      MODEL="$2"
      shift 2
      ;;
    --served-name)
      require_value "$@"
      SERVED_NAME="$2"
      shift 2
      ;;
    --local-port)
      require_value "$@"
      LOCAL_PORT="$2"
      shift 2
      ;;
    --remote-port)
      require_value "$@"
      REMOTE_PORT="$2"
      shift 2
      ;;
    --bind-host)
      require_value "$@"
      BIND_HOST="$2"
      shift 2
      ;;
    --vllm-venv)
      require_value "$@"
      VLLM_VENV="$2"
      shift 2
      ;;
    --log-dir)
      require_value "$@"
      LOG_DIR="$2"
      shift 2
      ;;
    --attach)
      ATTACH=1
      shift
      ;;
    --ready-timeout-s)
      require_value "$@"
      READY_TIMEOUT_S="$2"
      shift 2
      ;;
    --no-tool-calling)
      ENABLE_TOOL_CALLING=0
      shift
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

if ! [[ "$LOCAL_PORT" =~ ^[0-9]+$ ]] || ! [[ "$REMOTE_PORT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ports must be numeric." >&2
  exit 2
fi

if ! [[ "$READY_TIMEOUT_S" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --ready-timeout-s must be numeric seconds." >&2
  exit 2
fi

if [ -z "$BIND_HOST" ]; then
  if [ -n "$SSH_HOST" ]; then
    BIND_HOST="127.0.0.1"
  else
    BIND_HOST="0.0.0.0"
  fi
fi

if [ -z "$LOG_DIR" ]; then
  if [ -n "$SSH_HOST" ]; then
    LOG_DIR=""
  else
    LOG_DIR="$REPO_ROOT/runs/vllm"
  fi
fi

if [ -z "${TOOL_CALL_PARSER:-}" ]; then
  case "$MODEL" in
    Qwen/*|*Qwen*|*qwen*) TOOL_CALL_PARSER=qwen3_xml ;;
    *) TOOL_CALL_PARSER=hermes ;;
  esac
fi

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

tunnel_target_host() {
  if [ "$BIND_HOST" = "0.0.0.0" ] || [ "$BIND_HOST" = "::" ]; then
    printf "127.0.0.1"
  else
    printf "%s" "$BIND_HOST"
  fi
}

client_base_url() {
  local port="$1"
  printf "http://localhost:%s/v1" "$port"
}

check_local_port_free() {
  local port="$1"
  if ( : >"/dev/tcp/127.0.0.1/${port}" ) >/dev/null 2>&1; then
    echo "ERROR: local port ${port} is already in use." >&2
    echo "Use --local-port with another value, for example: --local-port 18000" >&2
    exit 1
  fi
}

wait_for_http() {
  local url="$1"
  local timeout_s="$2"
  python3 - "$url" "$timeout_s" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/") + "/models"
deadline = time.time() + int(sys.argv[2])
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [item.get("id") for item in payload.get("data", []) if item.get("id")]
        print("ready models: " + (", ".join(names) if names else "<none listed>"))
        raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        last_error = str(exc)
        time.sleep(5)

print(f"timed out waiting for {url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

start_local_vllm() {
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
    echo "Activate a vLLM environment or pass --vllm-venv /path/to/env." >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  local stamp log_file pid_file pid
  stamp="$(date +%Y%m%d-%H%M%S)"
  log_file="$LOG_DIR/vllm-${REMOTE_PORT}-${stamp}.log"
  pid_file="$LOG_DIR/vllm-${REMOTE_PORT}.pid"

  local tool_args=()
  if [ "$ENABLE_TOOL_CALLING" = "1" ]; then
    tool_args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
    if [ -n "$CHAT_TEMPLATE" ]; then
      tool_args+=(--chat-template "$CHAT_TEMPLATE")
    fi
  fi

  local tp_args=()
  if [ "$TENSOR_PARALLEL_SIZE" != "1" ]; then
    tp_args+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
  fi

  echo "Starting local vLLM..."
  echo "  model       : $MODEL"
  echo "  served name : $SERVED_NAME"
  echo "  bind        : $BIND_HOST:$REMOTE_PORT"
  echo "  log         : $log_file"

  nohup vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$BIND_HOST" \
    --port "$REMOTE_PORT" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${tp_args[@]}" \
    "${tool_args[@]}" \
    >"$log_file" 2>&1 &

  pid="$!"
  echo "$pid" > "$pid_file"
  echo "  pid         : $pid"
  echo "  pid file    : $pid_file"
}

run_local() {
  if [ "$ATTACH" != "1" ]; then
    if ( : >"/dev/tcp/127.0.0.1/${REMOTE_PORT}" ) >/dev/null 2>&1; then
      echo "ERROR: port ${REMOTE_PORT} already has a listening service." >&2
      echo "Use --attach if this is an existing vLLM server, or choose --remote-port." >&2
      exit 1
    fi
    start_local_vllm
  else
    echo "Attaching to local vLLM on ${BIND_HOST}:${REMOTE_PORT}..."
  fi

  local base_url
  base_url="http://$(tunnel_target_host):${REMOTE_PORT}/v1"
  echo "Waiting for vLLM HTTP readiness at $base_url..."
  wait_for_http "$base_url" "$READY_TIMEOUT_S"

  cat <<EOF

vLLM HTTP is ready.

Client environment:
  export VLLM_BASE_URL=$(client_base_url "$REMOTE_PORT")
  export VLLM_MODEL=${SERVED_NAME}
  export VLLM_API_KEY=${VLLM_API_KEY}
EOF
}

quote_words() {
  local word
  for word in "$@"; do
    printf "%q " "$word"
  done
}

run_remote() {
  check_local_port_free "$LOCAL_PORT"

  local control_dir
  control_dir="${TMPDIR:-/tmp}/veri-thesis-ssh"
  mkdir -p "$control_dir"
  chmod 700 "$control_dir"

  local ssh_opts=(
    -o ControlMaster=auto
    -o ControlPersist=10m
    -o ControlPath="${control_dir}/%r@%h:%p"
  )

  local remote_env=(
    "MODEL=$MODEL"
    "SERVED_NAME=$SERVED_NAME"
    "REMOTE_PORT=$REMOTE_PORT"
    "BIND_HOST=$BIND_HOST"
    "VLLM_VENV=$VLLM_VENV"
    "LOG_DIR=$LOG_DIR"
    "ATTACH=$ATTACH"
    "READY_TIMEOUT_S=$READY_TIMEOUT_S"
    "ENABLE_TOOL_CALLING=$ENABLE_TOOL_CALLING"
    "TOOL_CALL_PARSER=$TOOL_CALL_PARSER"
    "CHAT_TEMPLATE=$CHAT_TEMPLATE"
    "DTYPE=$DTYPE"
    "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
    "MAX_MODEL_LEN=$MAX_MODEL_LEN"
    "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
    "VLLM_API_KEY=$VLLM_API_KEY"
  )

  local name
  for name in CUDA_VISIBLE_DEVICES HF_HOME; do
    if [ -n "${!name:-}" ]; then
      remote_env+=("${name}=${!name}")
    fi
  done

  echo "Deploying vLLM on $SSH_HOST..."
  ssh "${ssh_opts[@]}" "$SSH_HOST" "$(quote_words "${remote_env[@]}") bash -s" <<'REMOTE'
set -euo pipefail

if [ -n "$VLLM_VENV" ]; then
  if [ ! -f "$VLLM_VENV/bin/activate" ]; then
    echo "ERROR: venv activate script not found: $VLLM_VENV/bin/activate" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VLLM_VENV/bin/activate"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not available on PATH on the remote host." >&2
  echo "Pass --vllm-venv /path/to/env or install vLLM there." >&2
  exit 1
fi

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-$HOME/vllm_runs}"

target_host="$BIND_HOST"
if [ "$target_host" = "0.0.0.0" ] || [ "$target_host" = "::" ]; then
  target_host="127.0.0.1"
fi

if [ "$ATTACH" != "1" ]; then
  if python3 - "$target_host" "$REMOTE_PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=1):
    pass
PY
  then
    echo "ERROR: remote port ${REMOTE_PORT} already has a listening service." >&2
    echo "Use --attach if this is an existing vLLM server, or choose --remote-port." >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  stamp="$(date +%Y%m%d-%H%M%S)"
  log_file="$LOG_DIR/vllm-${REMOTE_PORT}-${stamp}.log"
  pid_file="$LOG_DIR/vllm-${REMOTE_PORT}.pid"

  tool_args=()
  if [ "$ENABLE_TOOL_CALLING" = "1" ]; then
    tool_args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
    if [ -n "$CHAT_TEMPLATE" ]; then
      tool_args+=(--chat-template "$CHAT_TEMPLATE")
    fi
  fi

  tp_args=()
  if [ "$TENSOR_PARALLEL_SIZE" != "1" ]; then
    tp_args+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
  fi

  echo "Starting remote vLLM..."
  echo "  model       : $MODEL"
  echo "  served name : $SERVED_NAME"
  echo "  bind        : $BIND_HOST:$REMOTE_PORT"
  echo "  log         : $log_file"

  nohup vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$BIND_HOST" \
    --port "$REMOTE_PORT" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${tp_args[@]}" \
    "${tool_args[@]}" \
    >"$log_file" 2>&1 &

  pid="$!"
  echo "$pid" > "$pid_file"
  echo "  pid         : $pid"
  echo "  pid file    : $pid_file"
else
  echo "Attaching to remote vLLM on ${target_host}:${REMOTE_PORT}..."
fi

python3 - "$target_host" "$REMOTE_PORT" "$READY_TIMEOUT_S" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

host, port, timeout_s = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + timeout_s
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [item.get("id") for item in payload.get("data", []) if item.get("id")]
        print("ready models: " + (", ".join(names) if names else "<none listed>"))
        raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        last_error = str(exc)
        time.sleep(5)

print(f"timed out waiting for {url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
REMOTE

  local target_host
  target_host="$(tunnel_target_host)"

  cat <<EOF

Forwarding vLLM HTTP:
  local:  http://localhost:${LOCAL_PORT}/v1
  remote: http://${target_host}:${REMOTE_PORT}/v1 on ${SSH_HOST}

Client environment:
  export VLLM_BASE_URL=$(client_base_url "$LOCAL_PORT")
  export VLLM_MODEL=${SERVED_NAME}
  export VLLM_API_KEY=${VLLM_API_KEY}

Keep this terminal open. Stop the tunnel with Ctrl-C.
EOF

  exec ssh "${ssh_opts[@]}" -o ExitOnForwardFailure=yes -N -L "${LOCAL_PORT}:${target_host}:${REMOTE_PORT}" "$SSH_HOST"
}

if [ -n "$SSH_HOST" ]; then
  run_remote
else
  run_local
fi
