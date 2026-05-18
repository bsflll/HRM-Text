#!/usr/bin/env bash
#SBATCH --job-name=hrm-coach-codex-ab
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-79
#SBATCH --output=experiments/agent_coach/logs/hrm-coach-codex-ab-%A_%a.out
#SBATCH --error=experiments/agent_coach/logs/hrm-coach-codex-ab-%A_%a.err

set -euo pipefail

HRM_DIR="${HRM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS:?Set TASKS to a JSONL file with live Codex tasks}"
COACH_MODEL_PATH="${COACH_MODEL_PATH:?Set COACH_MODEL_PATH to the trained HRM-Coach checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HRM_DIR/experiments/agent_coach/outputs/codex_ab_${SLURM_JOB_ID:-local}}"

ARMS="${ARMS:-baseline,coach}"
MAX_PASSES="${MAX_PASSES:-2}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_MODEL="${CODEX_MODEL:-}"
CODEX_SANDBOX="${CODEX_SANDBOX:-danger-full-access}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-900}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-300}"
SETUP_TIMEOUT="${SETUP_TIMEOUT:-600}"
COACH_MAX_PROMPT_LEN="${COACH_MAX_PROMPT_LEN:-1800}"
COACH_MAX_NEW_TOKENS="${COACH_MAX_NEW_TOKENS:-384}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
OUTPUT_DIR="$OUTPUT_ROOT/shard_${SHARD_INDEX}"

mkdir -p "$HRM_DIR/experiments/agent_coach/logs" "$OUTPUT_DIR"
cd "$HRM_DIR"

ARGS=(
  experiments/agent_coach/scripts/run_codex_ab.py
  --tasks "$TASKS"
  --output-dir "$OUTPUT_DIR"
  --coach-model-path "$COACH_MODEL_PATH"
  --arms "$ARMS"
  --max-passes "$MAX_PASSES"
  --task-shard-index "$SHARD_INDEX"
  --task-shard-count "$SHARD_COUNT"
  --codex-bin "$CODEX_BIN"
  --codex-sandbox "$CODEX_SANDBOX"
  --codex-timeout "$CODEX_TIMEOUT"
  --verify-timeout "$VERIFY_TIMEOUT"
  --setup-timeout "$SETUP_TIMEOUT"
  --coach-max-prompt-len "$COACH_MAX_PROMPT_LEN"
  --coach-max-new-tokens "$COACH_MAX_NEW_TOKENS"
)

if [[ -n "$CODEX_MODEL" ]]; then
  ARGS+=(--codex-model "$CODEX_MODEL")
fi
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  ARGS+=(--local-files-only)
fi

"$PYTHON_BIN" "${ARGS[@]}"
