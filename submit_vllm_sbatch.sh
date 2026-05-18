#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm}"

mkdir -p "$SCRIPT_DIR/runs/slurm"

JOB_ID="$(
  sbatch \
    --parsable \
    "$@" \
    "$SCRIPT_DIR/scripts/vllm_server.sh"
)"

cat <<EOF
Submitted vLLM Slurm job: $JOB_ID

Runtime environment:
  VLLM_VENV=$VLLM_VENV

Check status:
  squeue -j $JOB_ID

Watch logs:
  tail -f $SCRIPT_DIR/runs/slurm/vllm-${JOB_ID}.out

After the job is RUNNING, create the tunnel from your local computer:
  bash forward.sh <nchc-login-host> $JOB_ID

Then send requests to:
  http://localhost:8000/v1

Client environment:
  export VLLM_BASE_URL=http://localhost:8000/v1
  export VLLM_MODEL=${SERVED_NAME:-siliconmind-server}
  export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
EOF
