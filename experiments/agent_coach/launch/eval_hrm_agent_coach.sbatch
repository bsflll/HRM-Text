#!/bin/bash
#SBATCH --job-name=hrm-coach-eval
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=experiments/agent_coach/logs/%x-%j.out
#SBATCH --error=experiments/agent_coach/logs/%x-%j.err

set -euo pipefail

HRM_DIR="${HRM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
EXP_DIR="${EXP_DIR:-${HRM_DIR}/experiments/agent_coach}"
MODEL_PATH="${MODEL_PATH:-${EXP_DIR}/outputs/latest/final}"
DATA_PATH="${DATA_PATH:-${EXP_DIR}/data/open_agent_traces/val.jsonl}"
OUT="${OUT:-${EXP_DIR}/outputs/eval_${SLURM_JOB_ID}/predictions.jsonl}"
LIMIT="${LIMIT:-200}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "${EXP_DIR}/logs" "$(dirname "${OUT}")"

cd "${HRM_DIR}"

args=(
  "${EXP_DIR}/scripts/eval_hrm_coach.py"
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --out "${OUT}"
  --limit "${LIMIT}"
  --max-prompt-len 1800
  --max-new-tokens 384
)

if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  args+=(--local-files-only)
fi

"${PYTHON_BIN}" "${args[@]}"
