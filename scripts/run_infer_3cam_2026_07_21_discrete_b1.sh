#!/usr/bin/env bash
# 2026-07-21 三目离散 B1 权重实机 infer
# 训练: chunk=50, bins=256, num_tasks=16, best@epoch2120
# 用法:
#   ./run_infer_3cam_2026_07_21_discrete_b1.sh dry-run   # 机械臂不动，只验链路
#   ./run_infer_3cam_2026_07_21_discrete_b1.sh run       # 正式执行
#   ./run_infer_3cam_2026_07_21_discrete_b1.sh attach
#   ./run_infer_3cam_2026_07_21_discrete_b1.sh kill
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BRUSH_CKPT_DIR="${BRUSH_CKPT_DIR:-/media/shugen/LcxDisk/checkpoint/brush_policy_3cam_2026_07_21_task_discrete_b1}"
export INFER_CHUNK_SIZE="${INFER_CHUNK_SIZE:-50}"
export INFER_MAX_TIMESTEPS="${INFER_MAX_TIMESTEPS:-50}"
export INFER_TARGET_DELTA_GAIN="${INFER_TARGET_DELTA_GAIN:-1}"
export INFER_SLEEP_SEC="${INFER_SLEEP_SEC:-0.18}"
export INFER_TASK_ID="${INFER_TASK_ID:-0}"

echo "2026-07-21 离散 B1: ${BRUSH_CKPT_DIR}"
echo "  chunk=${INFER_CHUNK_SIZE}  timesteps=${INFER_MAX_TIMESTEPS}  gain=${INFER_TARGET_DELTA_GAIN}  task_id=${INFER_TASK_ID}"
echo "  (action_type 会从 dataset_stats.pkl 自动识别为 discrete)"
exec "${SCRIPT_DIR}/run_infer_3cam.sh" "${1:-run}"
