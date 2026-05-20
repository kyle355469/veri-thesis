#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash forward.sh <nchc-login-host> <slurm-job-id> [local-port] [remote-port]

Run this on your local computer after the Slurm job is RUNNING.

Examples:
  bash forward.sh user@alogin.nchc.org.tw 123456
  bash forward.sh user@alogin.nchc.org.tw 123456 18000 8000

After the tunnel is open:
  export VLLM_BASE_URL=http://localhost:8000/v1
  export VLLM_MODEL=siliconmind-server
  export VLLM_API_KEY=EMPTY
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 2 ]; then
  usage
  exit 0
fi

LOGIN_HOST="$1"
JOB_ID="$2"
LOCAL_PORT="${3:-8000}"
REMOTE_PORT="${4:-8000}"

CONTROL_DIR="${TMPDIR:-/tmp}/veri-thesis-ssh"
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
  echo "  bash forward.sh ${LOGIN_HOST} ${JOB_ID} 18000 ${REMOTE_PORT}" >&2
  echo "Then set:" >&2
  echo "  export VLLM_BASE_URL=http://localhost:18000/v1" >&2
  exit 1
fi

JOB_INFO="$(
  ssh "${SSH_OPTS[@]}" "$LOGIN_HOST" "squeue -h -j '$JOB_ID' -o '%T %N' | head -n 1"
)"

STATE="${JOB_INFO%% *}"
NODE="${JOB_INFO#* }"

if [ -z "$JOB_INFO" ]; then
  echo "Could not find Slurm job $JOB_ID." >&2
  echo "Check it with: ssh $LOGIN_HOST squeue -j $JOB_ID" >&2
  exit 1
fi

if [ "$STATE" != "RUNNING" ]; then
  echo "Slurm job $JOB_ID is $STATE, not RUNNING." >&2
  echo "Check it with: ssh $LOGIN_HOST squeue -j $JOB_ID" >&2
  exit 1
fi

if [ -z "$NODE" ] || [ "$NODE" = "(null)" ] || [ "${NODE#(}" != "$NODE" ]; then
  echo "Could not find a running node for Slurm job $JOB_ID." >&2
  echo "Check it with: ssh $LOGIN_HOST squeue -j $JOB_ID" >&2
  exit 1
fi

cat <<EOF
Forwarding local port ${LOCAL_PORT} to vLLM on ${NODE}:${REMOTE_PORT}

Keep this terminal open. In another local terminal, use:
  export VLLM_BASE_URL=http://localhost:${LOCAL_PORT}/v1
  export VLLM_MODEL=siliconmind-server
  export VLLM_API_KEY=EMPTY
EOF

exec ssh "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes -N -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" "$LOGIN_HOST"
