#!/usr/bin/env bash
#SBATCH --account=GOV113121
#SBATCH --partition=large
#SBATCH --job-name=vllm-rtl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=200G
#SBATCH --gpus=8
#SBATCH --time=08:30:00
#SBATCH --output=runs/slurm/vllm-%j.out
#SBATCH --error=runs/slurm/vllm-%j.err

# NCHC partitions/accounts differ by project. If your queue requires them,
# uncomment and edit these lines before submitting:
##SBATCH --partition=gpu
##SBATCH --account=YOUR_PROJECT_ID

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm}"

mkdir -p "$REPO_DIR/runs/slurm"
cd "$REPO_DIR"

echo "Job id: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_JOB_NODELIST:-unknown}"
echo "Resolved node: $(hostname -f 2>/dev/null || hostname)"
echo "Repository: $PWD"
echo "PORT=$PORT"
echo "HOST=$HOST"

# Optional NCHC environment hooks. Set these in the sbatch command or uncomment
# the module/venv lines that match your allocation.
if [ -f /etc/profile ]; then
  # Some Slurm shells do not initialize the environment modules function.
  # shellcheck disable=SC1091
  source /etc/profile
fi

if [ -n "${VLLM_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $VLLM_MODULES
fi

if [ ! -f "$VLLM_VENV/bin/activate" ]; then
  echo "ERROR: VLLM_VENV does not contain an activate script: $VLLM_VENV/bin/activate" >&2
  echo "Create it with uv, or submit with VLLM_VENV=/path/to/vllm-env." >&2
  echo "Recommended: uv venv --managed-python --python 3.11.15 $VLLM_VENV && uv pip --python $VLLM_VENV/bin/python install vllm" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VLLM_VENV/bin/activate"

echo "Python executable: $(command -v python || true)"
python --version
echo "vLLM executable: $(command -v vllm || true)"

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not available in VLLM_VENV=$VLLM_VENV" >&2
  exit 1
fi

if ! python - <<'PY'
import os
import sys
import sysconfig

header = os.path.join(sysconfig.get_path("include"), "Python.h")
print(f"Python.h: {header} exists={os.path.exists(header)}")
if not os.path.exists(header):
    sys.exit(1)
PY
then
  echo "ERROR: Python.h is missing for this Python runtime." >&2
  echo "Torch/Triton compilation needs Python development headers; use a uv-managed Python venv rather than the system Python." >&2
  exit 1
fi

export HOST
export PORT
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

echo "OpenAI-compatible endpoint inside NCHC: http://$(hostname -f 2>/dev/null || hostname):${PORT}/v1"
srun ENABLE_TOOL_CALLING=1 TOOL_CALL_PARSER=hermes bash ./vllm_deploy.sh
