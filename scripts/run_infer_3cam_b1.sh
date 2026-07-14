#!/usr/bin/env bash
# B1 离散权重实机 infer：chunk=50, batch=32, lr=2e-5
# 用法: ./run_infer_3cam_b1.sh [dry-run|run|attach|kill]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BRUSH_CKPT_DIR="${BRUSH_CKPT_DIR:-/media/shugen/LcxDisk/checkpoint/brush_policy_3cam_task0_46_discrete_b1}"
export INFER_CHUNK_SIZE="${INFER_CHUNK_SIZE:-50}"
export INFER_MAX_TIMESTEPS="${INFER_MAX_TIMESTEPS:-50}"
export INFER_TARGET_DELTA_GAIN="${INFER_TARGET_DELTA_GAIN:-1}"
export INFER_SLEEP_SEC="${INFER_SLEEP_SEC:-0.18}"
export INFER_TASK_ID="${INFER_TASK_ID:-0}"

echo "B1 离散权重: ${BRUSH_CKPT_DIR}"
echo "  chunk=${INFER_CHUNK_SIZE}  timesteps=${INFER_MAX_TIMESTEPS}  gain=${INFER_TARGET_DELTA_GAIN}  task_id=${INFER_TASK_ID}"
exec "${SCRIPT_DIR}/run_infer_3cam.sh" "${1:-run}"
