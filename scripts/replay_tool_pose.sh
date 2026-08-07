#!/usr/bin/env bash
# 回放 tool_pose.csv 轨迹（绝对 TCP 位姿 → /mov_jog）
#
# 用法:
#   bash scripts/replay_tool_pose.sh /path/to/tool_pose.csv
#   bash scripts/replay_tool_pose.sh /path/to/session_dir          # 自动找 robot_state/tool_pose.csv
#   STEP_M=0.02 bash scripts/replay_tool_pose.sh xxx.csv          # 按 xyz 路程每 2cm 取点
#   MODE=all bash scripts/replay_tool_pose.sh xxx.csv             # 几乎逐帧（仍会去掉重复点）
#   MAX_STEPS=20 bash scripts/replay_tool_pose.sh xxx.csv         # 只发前 20 个路点
#   DRY_RUN=1 bash scripts/replay_tool_pose.sh xxx.csv            # 只打印，不发运动
#
# 依赖: robot_driver_bridge_node 已启动，/mov_jog 可用。
# 注意: 请先确认轨迹安全；建议先 DRY_RUN=1 再实机。

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTO_WELDING_SETUP="${AUTO_WELDING_SETUP:-${HOME}/Documents/auto_welding/install/local_setup.bash}"

TARGET="${1:-}"
STEP_M="${STEP_M:-0.03}"
MODE="${MODE:-distance}"          # distance | all
MAX_STEPS="${MAX_STEPS:-0}"
SERVICE="${SERVICE:-/mov_jog}"
SRV_TYPE="${SRV_TYPE:-common_interface/srv/Move}"
DRY_RUN="${DRY_RUN:-0}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-9}"
MIN_STEP_M="${MIN_STEP_M:-0.001}" # MODE=all 时忽略小于该位移的重复点

if [[ -z "${TARGET}" ]]; then
  cat <<EOF
用法: $0 <tool_pose.csv|session目录>

示例:
  $0 /home/shugen/yanjie/ros2_ws/data_collect/2026-07-27/10-29-50/robot_state/tool_pose.csv
  $0 /home/shugen/yanjie/ros2_ws/data_collect/2026-07-27/10-29-50
  STEP_M=0.02 $0 path/to/tool_pose.csv
  MODE=all MAX_STEPS=50 $0 path/to/tool_pose.csv
  DRY_RUN=1 $0 path/to/tool_pose.csv
EOF
  exit 1
fi

# 避免 conda python 污染 ros2
if command -v conda >/dev/null 2>&1; then
  conda deactivate >/dev/null 2>&1 || true
fi
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/miniconda3/\|/anaconda3/' | paste -sd: -)"
export ROS_DOMAIN_ID
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${AUTO_WELDING_SETUP}" ]]; then
  source "${AUTO_WELDING_SETUP}"
fi
source "${WS_ROOT}/install/local_setup.bash"
set -u

resolve_csv() {
  local target="$1"
  if [[ -f "${target}" ]]; then
    echo "${target}"
    return
  fi
  if [[ -d "${target}" ]]; then
    if [[ -f "${target}/robot_state/tool_pose.csv" ]]; then
      echo "${target}/robot_state/tool_pose.csv"
      return
    fi
    if [[ -f "${target}/tool_pose.csv" ]]; then
      echo "${target}/tool_pose.csv"
      return
    fi
  fi
  echo "未找到 tool_pose.csv: ${target}" >&2
  exit 1
}

subsample_csv() {
  local csv="$1"
  MODE="${MODE}" STEP_M="${STEP_M}" MIN_STEP_M="${MIN_STEP_M}" "${PYTHON_BIN}" - "$csv" <<'PY'
import math
import os
import sys

mode = os.environ.get("MODE", "distance")
step_m = float(os.environ.get("STEP_M", "0.03"))
min_step_m = float(os.environ.get("MIN_STEP_M", "0.001"))
csv_path = sys.argv[1]
poses = []
with open(csv_path, encoding="utf-8", errors="replace") as handle:
    for raw in handle:
        line = raw.strip()
        if not line or line.lower().startswith("timestamp"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            values = [float(parts[i]) for i in range(1, 7)]
        except ValueError:
            continue
        poses.append(values)

if not poses:
    raise SystemExit(f"未解析到有效位姿: {csv_path}")

if mode == "all":
    selected = [poses[0]]
    for pose in poses[1:]:
        if math.dist(selected[-1][:3], pose[:3]) >= min_step_m:
            selected.append(pose)
    if selected[-1] != poses[-1]:
        selected.append(poses[-1])
else:
    selected = [poses[0]]
    acc = 0.0
    anchor = poses[0][:3]
    for pose in poses[1:]:
        acc += math.dist(anchor, pose[:3])
        if acc >= step_m:
            selected.append(pose)
            anchor = pose[:3]
            acc = 0.0
    if selected[-1] != poses[-1]:
        selected.append(poses[-1])

print(
    f"# raw={len(poses)} selected={len(selected)} mode={mode} step_m={step_m}",
    file=sys.stderr,
)
for pose in selected:
    print(",".join(f"{v:.10f}" for v in pose))
PY
}

play_csv() {
  local csv="$1"
  local count=0

  echo ""
  echo "=== 回放: ${csv} ==="
  echo "服务: ${SERVICE} (${SRV_TYPE})"
  echo "模式: MODE=${MODE} STEP_M=${STEP_M} MAX_STEPS=${MAX_STEPS} DRY_RUN=${DRY_RUN}"

  if ! ros2 service list | grep -qx "${SERVICE}"; then
    echo "错误: 服务不存在 ${SERVICE}"
    echo "请先启动: ros2 run welding_runtime robot_driver_bridge_node --ros-args -p robot_type:=duco"
    exit 1
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    echo "将开始实机运动。3 秒内 Ctrl+C 可取消..."
    sleep 3
  fi

  while IFS= read -r line; do
    [[ "${line}" == \#* || -z "${line}" ]] && continue
    IFS=',' read -r x y z rx ry rz <<< "${line}"
    count=$((count + 1))
    echo "[${count}] xyz=(${x}, ${y}, ${z}) rpy=(${rx}, ${ry}, ${rz})"
    if [[ "${DRY_RUN}" != "1" ]]; then
      ros2 service call "${SERVICE}" "${SRV_TYPE}" \
        "{a: ${x}, b: ${y}, c: ${z}, d: ${rx}, e: ${ry}, f: ${rz}, block: true, name: ''}"
    fi

    if [[ "${MAX_STEPS}" -gt 0 && "${count}" -ge "${MAX_STEPS}" ]]; then
      echo "已达 MAX_STEPS=${MAX_STEPS}，提前结束"
      break
    fi
  done < <(subsample_csv "${csv}")

  echo "完成: ${count} 个路点 (DRY_RUN=${DRY_RUN})"
}

CSV_PATH="$(resolve_csv "${TARGET}")"
play_csv "${CSV_PATH}"
