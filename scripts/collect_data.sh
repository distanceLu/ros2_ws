#!/usr/bin/env bash
# 一键启动训练数据采集环境（tmux 多窗口）
#
# task_id（轮廓任务 id，供后期 task-conditioned 训练）:
#   ./collect_data.sh              # 启动环境，初始 task_id=0
#   ./collect_data.sh start 3      # 可选：指定初始 task_id=3
#   TASK_ID=3 ./collect_data.sh    # 可选：环境变量指定初始值
#
# 采集节点会把 task_id 写入每个 session 的 session_meta.json；
# session 窗口每次按 [s] 开始轨迹前都会询问本条轨迹的 task_id。

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION="${SESSION:-welding_collect}"

AUTO_WELDING_SETUP="${AUTO_WELDING_SETUP:-${HOME}/Documents/auto_welding/install/local_setup.bash}"
ROS2_WS_SETUP="${ROS2_WS_SETUP:-${WS_ROOT}/install/local_setup.bash}"
CAMERA_SCRIPT="${CAMERA_SCRIPT:-${SCRIPT_DIR}/camera_capture_node.py}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-${SCRIPT_DIR}/training_data_collect.py}"
SESSION_SCRIPT="${SESSION_SCRIPT:-${SCRIPT_DIR}/training_collect.sh}"
CAMERA_KEYS="${CAMERA_KEYS:-pool}"
# 默认关闭 3D→2D 采集；若要恢复：CAMERA_KEYS=3d,pool ENABLE_SCAN_CAMERA=1
ENABLE_SCAN_CAMERA="${ENABLE_SCAN_CAMERA:-0}"
# 4K HD Camera: by-id …-video-index0 → Capture（当前机 /dev/video1）；勿用 Metadata 节点
PAPER_CAMERA_DEVICE="${PAPER_CAMERA_DEVICE:-/dev/v4l/by-id/usb-Image+_4K_HD_Camera_YL-001-video-index0}"
# 熔池约 17–20 Hz；纸面默认 20Hz + 3840x2160 + 中心 3x 裁切（相对原 2x 再放大 1.5）
PAPER_CAMERA_HZ="${PAPER_CAMERA_HZ:-20.0}"
PAPER_CAMERA_ZOOM="${PAPER_CAMERA_ZOOM:-2.25}"
PAPER_CAMERA_WIDTH="${PAPER_CAMERA_WIDTH:-3840}"
PAPER_CAMERA_HEIGHT="${PAPER_CAMERA_HEIGHT:-2160}"
PAPER_CAMERA_SAVE_WIDTH="${PAPER_CAMERA_SAVE_WIDTH:-1920}"
PAPER_CAMERA_SAVE_HEIGHT="${PAPER_CAMERA_SAVE_HEIGHT:-1080}"
PAPER_CAMERA_FOCUS="${PAPER_CAMERA_FOCUS:-360}"
PAPER_CAMERA_SHARPNESS="${PAPER_CAMERA_SHARPNESS:-48}"
SAVE_DIR_ROOT="${SAVE_DIR_ROOT:-/data/yanjie/data_collect}"

build_env_prefix() {
  local parts=()
  parts+=("unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH")
  if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
    parts+=("export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
  elif [[ -f "${HOME}/.bashrc" ]] && grep -q '^export ROS_DOMAIN_ID=' "${HOME}/.bashrc"; then
    parts+=("source '${HOME}/.bashrc'")
  fi
  if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
    parts+=("source /opt/ros/jazzy/setup.bash")
  fi
  if [[ -f "${AUTO_WELDING_SETUP}" ]]; then
    parts+=("source '${AUTO_WELDING_SETUP}'")
  fi
  if [[ -f "${ROS2_WS_SETUP}" ]]; then
    parts+=("source '${ROS2_WS_SETUP}'")
  fi
  # 让 session / collect 子进程都能读到当前轮廓 id 和实际采集目录。
  # 删除最近轨迹时必须使用与采集节点完全相同的 SAVE_DIR_ROOT。
  parts+=("export TASK_ID=$(printf '%q' "${TASK_ID:-0}")")
  parts+=("export SAVE_DIR_ROOT=$(printf '%q' "${SAVE_DIR_ROOT}")")
  printf '%s; ' "${parts[@]}"
}

resolve_task_id() {
  # 这里只设置启动初值；每次按 s 时会再次询问该条轨迹的 task_id。
  local from_arg="${1-}"
  if [[ -n "${from_arg}" ]]; then
    TASK_ID="${from_arg}"
  elif [[ -z "${TASK_ID:-}" ]]; then
    TASK_ID=0
  fi

  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "错误: task_id 必须是非负整数，收到: '${TASK_ID}'" >&2
    exit 1
  fi
  export TASK_ID
  echo "初始 task_id=${TASK_ID}；每次在 session 窗口按 s 时可为该条轨迹重新输入。"
}

cmd_attach() {
  tmux attach -t "${SESSION}"
}

cmd_kill() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}"
    echo "已停止 tmux 会话: ${SESSION}"
  else
    echo "会话不存在: ${SESSION}"
  fi
}

cmd_start() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "会话已存在，进入: tmux attach -t ${SESSION}"
    echo "提示: 已有会话不会自动改 task_id；在 session 窗口按 [t] 切换，或先 kill 再 start。"
    cmd_attach
    return
  fi

  if pgrep -x 'robot_driver_br' >/dev/null 2>&1; then
    echo "错误: 已有 robot_driver_bridge_node 正在运行，不能再启动第二个 SDK 采集连接。" >&2
    pgrep -ax 'robot_driver_br' >&2 || true
    echo "请先停止旧推理/采集会话中的 robot bridge，再重新运行本脚本。" >&2
    exit 1
  fi

  resolve_task_id "${1-}"

  local env_prefix
  env_prefix="$(build_env_prefix)"

  local scan_params="-p enable_scan_camera:=false -p restart_3d_on_timeout:=false"
  if [[ "${ENABLE_SCAN_CAMERA}" == "1" ]]; then
    scan_params="-p enable_scan_camera:=true"
  fi

  local camera_cmd="${env_prefix} python3 '${CAMERA_SCRIPT}' --ros-args -p auto_start_camera_keys:=${CAMERA_KEYS} -p keep_launched_drivers_on_exit:=true; echo camera 窗口已退出; read"
  local robot_cmd="${env_prefix} ros2 run welding_runtime robot_driver_bridge_node --ros-args -p robot_type:=duco; echo robot 窗口已退出; read"
  local collect_cmd="${env_prefix} sleep 10; python3 '${COLLECT_SCRIPT}' --ros-args -p save_dir_root:=${SAVE_DIR_ROOT} -p task_id:=${TASK_ID} ${scan_params} -p paper_camera_hz:=${PAPER_CAMERA_HZ} -p paper_camera_device:=${PAPER_CAMERA_DEVICE} -p paper_camera_zoom:=${PAPER_CAMERA_ZOOM} -p paper_camera_width:=${PAPER_CAMERA_WIDTH} -p paper_camera_height:=${PAPER_CAMERA_HEIGHT} -p paper_camera_save_width:=${PAPER_CAMERA_SAVE_WIDTH} -p paper_camera_save_height:=${PAPER_CAMERA_SAVE_HEIGHT} -p paper_camera_autofocus:=false -p paper_camera_focus_absolute:=${PAPER_CAMERA_FOCUS} -p paper_camera_sharpness:=${PAPER_CAMERA_SHARPNESS}; echo collect 窗口已退出; read"
  local session_cmd="${env_prefix} sleep 15; '${SESSION_SCRIPT}'; echo session 窗口已退出; read"
  local monitor_cmd="${env_prefix} echo '相机监控命令'; echo '熔池0: ros2 run image_view image_view --ros-args -r image:=/pool_camera/image_raw'; echo '熔池1: ros2 run image_view image_view --ros-args -r image:=/pool_camera1/image_raw'; echo '纸面: v4l2-ctl --list-devices  # 默认不采 3D；ENABLE_SCAN_CAMERA=1 可恢复'; echo '示教命令: ros2 topic hz /robot/command_state'; echo '当前 TASK_ID='\"\${TASK_ID}\" CAMERA_KEYS=${CAMERA_KEYS} ENABLE_SCAN_CAMERA=${ENABLE_SCAN_CAMERA}; echo '服务检查: ros2 service list | grep -E mov_jog\|training_data\|pool_camera'; exec bash"

  tmux new-session -d -s "${SESSION}" -n camera "bash -lc $(printf '%q' "${camera_cmd}")"
  tmux new-window -t "${SESSION}" -n robot "bash -lc $(printf '%q' "${robot_cmd}")"
  tmux new-window -t "${SESSION}" -n collect "bash -lc $(printf '%q' "${collect_cmd}")"
  tmux new-window -t "${SESSION}" -n session "bash -lc $(printf '%q' "${session_cmd}")"
  tmux new-window -t "${SESSION}" -n monitor "bash -lc $(printf '%q' "${monitor_cmd}")"

  echo "已创建 tmux 会话: ${SESSION}"
  echo "窗口: camera | robot | collect | session | monitor"
  echo "初始 task_id=${TASK_ID}；CAMERA_KEYS=${CAMERA_KEYS} ENABLE_SCAN_CAMERA=${ENABLE_SCAN_CAMERA}"
  echo "默认只采双熔池+纸面（无 3D）；恢复 3D: CAMERA_KEYS=3d,pool ENABLE_SCAN_CAMERA=1 $0"
  echo "在 session 窗口输入 r，然后填 2，即可连续采集 2 条轨迹（双熔池会写入 camera_pool/ 与 camera_pool1/）。"
  echo "在 session 窗口输入 p，可删除最近一条采集轨迹；会先二次确认。"
  echo "退出但不停止: Ctrl+B 然后 D"
  echo "停止全部: ${SCRIPT_DIR}/collect_data.sh kill"
  cmd_attach
}

cmd_photo() {
  local env_prefix
  env_prefix="$(build_env_prefix)"
  exec bash -lc "${env_prefix} python3 '${CAMERA_SCRIPT}' --ros-args -p auto_start_camera_keys:=${CAMERA_KEYS} -p keep_launched_drivers_on_exit:=true"
}

print_usage() {
  cat <<EOF
用法: $0 [start [task_id]|attach|kill|photo|<task_id>]

  start [task_id]  启动采集环境；task_id 仅作为初始值，默认 0
  <task_id>        等同于 start <task_id>
  attach           进入已有 tmux 会话
  kill|stop        停止 tmux 会话
  photo            仅启动相机节点

环境变量 TASK_ID 可指定初始值；每次按 s 开始轨迹前会再次询问。
EOF
}

case "${1:-start}" in
  start|"")
    cmd_start "${2-}"
    ;;
  attach)
    cmd_attach
    ;;
  kill|stop)
    cmd_kill
    ;;
  photo)
    cmd_photo
    ;;
  help|-h|--help)
    print_usage
    ;;
  *)
    if [[ "${1}" =~ ^[0-9]+$ ]]; then
      cmd_start "${1}"
    else
      print_usage
      exit 1
    fi
    ;;
esac
