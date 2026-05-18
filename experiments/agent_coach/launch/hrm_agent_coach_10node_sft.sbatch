#!/bin/bash
#SBATCH --job-name=hrm-agent-coach
#SBATCH --partition=cluster
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=experiments/agent_coach/logs/%x-%j.out
#SBATCH --error=experiments/agent_coach/logs/%x-%j.err

set -euo pipefail

HRM_DIR="${HRM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
EXP_DIR="${EXP_DIR:-${HRM_DIR}/experiments/agent_coach}"
MODEL_PATH="${MODEL_PATH:-sapientinc/HRM-Text-1B}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}/data/open_agent_traces}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/outputs/run_${SLURM_JOB_ID}}"
MAX_PER_SOURCE="${MAX_PER_SOURCE:-30000}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "${EXP_DIR}/logs" "${DATA_DIR}" "${OUTPUT_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHON_BIN
export LOG_DIR="${EXP_DIR}/logs"

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${SLURM_JOB_NODELIST}"
echo "hrm_dir=${HRM_DIR}"
echo "model_path=${MODEL_PATH}"
echo "data_dir=${DATA_DIR}"
echo "output_dir=${OUTPUT_DIR}"

cd "${HRM_DIR}"

if [ -s "${DATA_DIR}/train.jsonl" ] && [ -s "${DATA_DIR}/val.jsonl" ] && [ -s "${DATA_DIR}/stats.json" ]; then
  echo "Reusing prepared real open agent traces from ${DATA_DIR}"
  cat "${DATA_DIR}/stats.json"
else
  echo "Preparing real open agent traces..."
  /data/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
    "${PYTHON_BIN}" "${EXP_DIR}/scripts/prepare_open_agent_traces.py" \
      --output-dir "${DATA_DIR}" \
      --max-per-source "${MAX_PER_SOURCE}" \
      --seed 17 \
      --max-turns 36 \
      --max-trace-chars 14000
fi

export MASTER_ADDR="$(/data/slurm/bin/scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((${SLURM_JOB_ID} % 16384 + 49152))"

trainer_args=(
  "${EXP_DIR}/scripts/train_hrm_coach_sft.py"
  --model-path "${MODEL_PATH}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --per-device-batch-size 1
  --grad-accum-steps 1
  --max-steps "${MAX_STEPS}"
  --stop-after-seconds 19800
  --lr 2.0e-5
  --warmup-steps 20
  --log-steps 5
  --eval-steps 100
  --eval-batches 4
  --save-steps 150
  --seed 17
)

if [ "${LOCAL_FILES_ONLY}" = "1" ]; then
  trainer_args+=(--local-files-only)
fi

echo "Starting HRM-Text agent-coach SFT on ${SLURM_JOB_NUM_NODES} nodes..."
/data/slurm/bin/srun --ntasks="${SLURM_NTASKS}" --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=8 \
  "${EXP_DIR}/launch/run_rank_python.sh" "${trainer_args[@]}"

echo "Finished HRM-Text agent-coach SFT. Output: ${OUTPUT_DIR}"
