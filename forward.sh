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

NODE="$(
  ssh "$LOGIN_HOST" "squeue -h -j '$JOB_ID' -o '%N' | head -n 1"
)"

if [ -z "$NODE" ] || [ "$NODE" = "(null)" ]; then
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

exec ssh -N -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" "$LOGIN_HOST"
