#!/usr/bin/env bash
# 一键拍摄：3D 相机纯 2D（无激光）+ 熔池相机
#
# 用法:
#   ./scripts/snap_pool_3d2d.sh              # 若 camera_capture_node 未运行则自动后台启动
#   ./scripts/snap_pool_3d2d.sh --no-start   # 仅拍照，不自动启动 camera_capture_node
#   ./scripts/snap_pool_3d2d.sh --daemon     # 后台启动相机节点后退出（不拍照）
#
# 图片默认保存到: /home/shugen/yanjie/ros2_ws/camera_images/pool/... 与 /home/shugen/yanjie/ros2_ws/camera_images/3d_2d/...
#
# 环境变量:
#   AUTO_WELDING_SETUP, ROS2_WS_SETUP  同 training_collect.sh
#   CAMERA_KEYS  默认 pool,3d（不含 2d，不启机械臂）

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AUTO_WELDING_SETUP="${AUTO_WELDING_SETUP:-${HOME}/Documents/auto_welding/install/local_setup.bash}"
ROS2_WS_SETUP="${ROS2_WS_SETUP:-${WS_ROOT}/install/local_setup.bash}"
CAMERA_SCRIPT="${CAMERA_SCRIPT:-${SCRIPT_DIR}/camera_capture_node.py}"
CAMERA_KEYS="${CAMERA_KEYS:-pool,3d}"
SERVICE_WAIT_SEC="${SERVICE_WAIT_SEC:-90}"
CAPTURE_TIMEOUT_SEC="${CAPTURE_TIMEOUT_SEC:-120}"

AUTO_START=true
DAEMON_ONLY=false

usage() {
  cat <<'EOF'
用法: snap_pool_3d2d.sh [选项]

  一键调用:
    /camera_capture/capture_once_3d_2d  — 3D 相机纯 2D
    /camera_capture/capture_once_pool   — 熔池相机

选项:
  --no-start   不自动启动 camera_capture_node（须已手动启动）
  --daemon     仅后台启动 camera_capture_node（pool+3d），不拍照
  -h, --help   显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start)
      AUTO_START=false
      shift
      ;;
    --daemon)
      DAEMON_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

source_setup() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

if [[ -f "${AUTO_WELDING_SETUP}" ]]; then
  source_setup "${AUTO_WELDING_SETUP}"
fi
if [[ -f "${ROS2_WS_SETUP}" ]]; then
  source_setup "${ROS2_WS_SETUP}"
fi

set -u

CAMERA_PID=""
CAMERA_STARTED_BY_SCRIPT=false
LOG_FILE="${TMPDIR:-/tmp}/snap_pool_3d2d_camera.log"

service_exists() {
  ros2 service list 2>/dev/null | grep -qx "$1"
}

wait_for_service() {
  local name="$1"
  local deadline=$((SECONDS + SERVICE_WAIT_SEC))
  echo "等待服务: ${name} (最多 ${SERVICE_WAIT_SEC}s)..."
  while (( SECONDS < deadline )); do
    if service_exists "${name}"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_camera_node() {
  echo "后台启动 camera_capture_node (keys=${CAMERA_KEYS})..."
  echo "日志: ${LOG_FILE}"
  python3 "${CAMERA_SCRIPT}" --ros-args \
    -p "auto_start_camera_keys:=${CAMERA_KEYS}" \
    -p keep_launched_drivers_on_exit:=true \
    >"${LOG_FILE}" 2>&1 &
  CAMERA_PID=$!
  CAMERA_STARTED_BY_SCRIPT=true
}

call_capture() {
  local label="$1"
  local service="$2"
  local output
  echo ""
  echo "=== ${label} ==="
  echo "调用 ${service} ..."
  if ! output=$(timeout "${CAPTURE_TIMEOUT_SEC}" ros2 service call "${service}" std_srvs/srv/Trigger {} 2>&1); then
    echo "${output}"
    echo "失败: ${label} (${service})" >&2
    return 1
  fi
  echo "${output}"
  if ! grep -q "success=True" <<<"${output}"; then
    echo "失败: ${label} (${service}) 返回 success=False" >&2
    return 1
  fi
  return 0
}

cleanup() {
  if [[ "${CAMERA_STARTED_BY_SCRIPT}" == true ]] && [[ -n "${CAMERA_PID}" ]]; then
  if kill -0 "${CAMERA_PID}" 2>/dev/null; then
      echo "停止由本脚本启动的 camera_capture_node (pid=${CAMERA_PID})"
      kill "${CAMERA_PID}" 2>/dev/null || true
      wait "${CAMERA_PID}" 2>/dev/null || true
    fi
  fi
}

# --daemon 模式：只起相机节点，不拍照、不自动清理
if [[ "${DAEMON_ONLY}" == true ]]; then
  if service_exists "/camera_capture/capture_once_pool"; then
    echo "camera_capture_node 已在运行，无需重复启动。"
    exit 0
  fi
  start_camera_node
  if wait_for_service "/camera_capture/capture_once_pool" \
    && wait_for_service "/camera_capture/capture_once_3d_2d"; then
    echo "camera_capture_node 已就绪 (pid=${CAMERA_PID})"
    echo "拍照: ${SCRIPT_DIR}/snap_pool_3d2d.sh --no-start"
    echo "日志: ${LOG_FILE}"
    exit 0
  fi
  echo "camera_capture_node 启动超时，查看日志: ${LOG_FILE}" >&2
  kill "${CAMERA_PID}" 2>/dev/null || true
  exit 1
fi

trap cleanup EXIT

if ! service_exists "/camera_capture/capture_once_pool"; then
  if [[ "${AUTO_START}" == true ]]; then
    start_camera_node
  else
    echo "未找到 /camera_capture/capture_once_pool。" >&2
    echo "请先运行 camera_capture_node 或去掉 --no-start。" >&2
    exit 1
  fi
else
  echo "检测到 camera_capture_node 已在运行。"
fi

FAIL=0
if ! wait_for_service "/camera_capture/capture_once_pool"; then
  echo "超时: /camera_capture/capture_once_pool" >&2
  echo "查看日志: ${LOG_FILE}" >&2
  exit 1
fi
if ! wait_for_service "/camera_capture/capture_once_3d_2d"; then
  echo "超时: /camera_capture/capture_once_3d_2d" >&2
  echo "查看日志: ${LOG_FILE}" >&2
  exit 1
fi

echo "相机服务已就绪，开始拍照..."

if ! call_capture "3D 相机纯 2D" "/camera_capture/capture_once_3d_2d"; then
  FAIL=1
fi
if ! call_capture "熔池相机" "/camera_capture/capture_once_pool"; then
  FAIL=1
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  echo "全部完成。图片目录根路径: ${WS_ROOT}/camera_images"
  echo "  pool/   — 熔池"
  echo "  3d_2d/  — 3D 纯 2D"
else
  echo "部分拍照失败。" >&2
  if [[ "${CAMERA_STARTED_BY_SCRIPT}" == true ]]; then
    echo "camera 节点日志: ${LOG_FILE}" >&2
  fi
  exit 1
fi

# 成功且由本脚本拉起相机节点时，默认保留节点运行（便于连续拍多张）
if [[ "${CAMERA_STARTED_BY_SCRIPT}" == true ]]; then
  trap - EXIT
  echo ""
  echo "camera_capture_node 仍在后台运行 (pid=${CAMERA_PID})，可再次执行:"
  echo "  ${SCRIPT_DIR}/snap_pool_3d2d.sh --no-start"
  echo "停止相机节点: kill ${CAMERA_PID}"
fi
