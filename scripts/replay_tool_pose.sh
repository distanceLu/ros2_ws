#!/usr/bin/env bash
# 回放 tool_pose.csv：沿路径每隔约 STEP_M 取一个点，直到轨迹结束
#
# 用法:
#   bash scripts/replay_tool_pose.sh path/to/tool_pose.csv
#   bash scripts/replay_tool_pose.sh path/to/2026-06-17/19-26-57
#   STEP_M=0.05 bash scripts/replay_tool_pose.sh ...    # 每 5cm 一个点
#   SERVICE=/mov_jog bash scripts/replay_tool_pose.sh ...
#   MAX_STEPS=10 bash scripts/replay_tool_pose.sh ...   # 最多发 10 个点
#
# 先启动: ros2 run robot_control robot_control_node

set -eo pipefail

TARGET="${1:-/home/shugen/yanjie/ros2_ws/data_collect/2026-06-17/19-26-57/robot_state/tool_pose.csv}"
STEP_M="${STEP_M:-0.03}"
MAX_STEPS="${MAX_STEPS:-0}"
SERVICE="${SERVICE:-/mov_tcp}"
SRV_TYPE="${SRV_TYPE:-robot_control/srv/Move}"

set +u
source /opt/ros/jazzy/setup.bash
source /home/shugen/yanjie/ros2_ws/install/local_setup.bash
set -u

subsample_csv() {
  local csv="$1"
  STEP_M="$STEP_M" python3 - "$csv" <<'PY'
import math
import os
import sys

step_m = float(os.environ.get("STEP_M", "0.03"))
csv_path = sys.argv[1]
poses = []
with open(csv_path, encoding="utf-8", errors="replace") as handle:
    for raw in handle:
        line = raw.strip()
        if not line or line.startswith("timestamp"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            values = [float(parts[i]) for i in range(1, 7)]
        except ValueError:
            continue
        poses.append(values)

if not poses:
    raise SystemExit(f"未解析到有效位姿: {csv_path}")

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

print(f"# raw={len(poses)} selected={len(selected)} step_m={step_m}", file=sys.stderr)
for pose in selected:
    print(",".join(f"{v:.10f}" for v in pose))
PY
}

play_csv() {
  local csv="$1"
  local count=0
  local total=0

  echo ""
  echo "=== 回放: $csv (每隔约 ${STEP_M}m 取点, 服务 ${SERVICE}) ==="

  while IFS= read -r line; do
    [[ "$line" == \#* || -z "$line" ]] && continue
    IFS=',' read -r x y z rx ry rz <<< "$line"
    total=$((total + 1))
    count=$((count + 1))
    echo "[$count] ${SERVICE}"
    ros2 service call "$SERVICE" "$SRV_TYPE" \
      "{a: ${x}, b: ${y}, c: ${z}, d: ${rx}, e: ${ry}, f: ${rz}, block: true, name: ''}"

    if [[ "$MAX_STEPS" -gt 0 && "$count" -ge "$MAX_STEPS" ]]; then
      echo "已达 MAX_STEPS=${MAX_STEPS}，提前结束"
      break
    fi
  done < <(subsample_csv "$csv")

  echo "完成: 发送 ${count} 个路点"
}

if [[ -f "$TARGET" ]]; then
  play_csv "$TARGET"
elif [[ -d "$TARGET" ]]; then
  mapfile -t files < <(find "$TARGET" -path '*/robot_state/tool_pose.csv' | sort)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "未找到 tool_pose.csv: $TARGET"
    exit 1
  fi
  for csv in "${files[@]}"; do
    play_csv "$csv"
  done
else
  echo "用法: $0 <tool_pose.csv 或 session/日期目录>"
  exit 1
fi
