#!/usr/bin/env bash
# 2026-07-24..27 四相机双熔池离散 ACT B1 权重真机推理
# 相机顺序必须与训练一致: pool, pool1, scan_2d, paper_aruco
#
# 用法:
#   ./run_infer_4cam_2026_07_24_27_dualpool_discrete_b1.sh dry-run
#   ./run_infer_4cam_2026_07_24_27_dualpool_discrete_b1.sh run
#   ./run_infer_4cam_2026_07_24_27_dualpool_discrete_b1.sh attach
#   ./run_infer_4cam_2026_07_24_27_dualpool_discrete_b1.sh kill
#
# 解码方式:
#   INFER_DISCRETE_DECODE=argmax|sample|expectation
#   INFER_DISCRETE_TEMPERATURE=1.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SESSION="${SESSION:-brush_infer_4cam_dualpool}"
export BRUSH_CKPT_DIR="${BRUSH_CKPT_DIR:-/media/shugen/LcxDisk/checkpoint/brush_policy_4cam_2026_07_24_27_dualpool_discrete_b1}"
export BRUSH_TASK_NAME="${BRUSH_TASK_NAME:-brush_tool_pose_2026_07_24_27_dualpool}"
export INFER_CAMERA_NAMES="${INFER_CAMERA_NAMES:-pool pool1 scan_2d paper_aruco}"
export POOL1_CAMERA_TOPIC="${POOL1_CAMERA_TOPIC:-/pool_camera1/image_raw}"
export INFER_CHUNK_SIZE="${INFER_CHUNK_SIZE:-50}"
export INFER_MAX_TIMESTEPS="${INFER_MAX_TIMESTEPS:-50}"
export INFER_TARGET_DELTA_GAIN="${INFER_TARGET_DELTA_GAIN:-1}"
export INFER_SLEEP_SEC="${INFER_SLEEP_SEC:-0.18}"
export INFER_TASK_ID="${INFER_TASK_ID:-0}"
export INFER_DISCRETE_DECODE="${INFER_DISCRETE_DECODE:-argmax}"
export INFER_DISCRETE_TEMPERATURE="${INFER_DISCRETE_TEMPERATURE:-1.0}"
export PAPER_CAMERA_DEVICE="${PAPER_CAMERA_DEVICE:-/dev/video0}"
export INFER_RECORD_CSV="${INFER_RECORD_CSV:-/tmp/infer_4cam_dualpool.csv}"

echo "2026-07-24..27 四相机双熔池离散 B1"
echo "  task=${BRUSH_TASK_NAME}"
echo "  checkpoint=${BRUSH_CKPT_DIR}"
echo "  cameras=${INFER_CAMERA_NAMES}"
echo "  chunk=${INFER_CHUNK_SIZE} timesteps=${INFER_MAX_TIMESTEPS} task_id=${INFER_TASK_ID}"
echo "  decode=${INFER_DISCRETE_DECODE} temperature=${INFER_DISCRETE_TEMPERATURE}"

exec "${SCRIPT_DIR}/run_infer_3cam.sh" "${1:-run}"
