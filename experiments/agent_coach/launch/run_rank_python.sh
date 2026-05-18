#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export WORLD_SIZE="${SLURM_NTASKS}"
export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE}"

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"

if [ "${SLURM_STEP_NUM_TASKS:-1}" -gt "${SLURM_STEP_NUM_NODES:-1}" ]; then
  export OMP_NUM_THREADS=1
fi

LOG_DIR="${LOG_DIR:-experiments/agent_coach/logs}"
mkdir -p "${LOG_DIR}"

"${PYTHON_BIN:-python3}" "$@" \
  2>&1 | tee "${LOG_DIR}/job_${SLURM_JOB_ID}_rank_${SLURM_PROCID}.log"
