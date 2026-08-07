#!/usr/bin/env bash
# 一键启动 ACT 多相机实机推理环境（默认旧三相机，支持四相机双熔池）
#
# 用法:
#   ./run_infer_3cam.sh              # 正式执行（机械臂会动）
#   ./run_infer_3cam.sh dry-run      # dry-run（机械臂不动，只验证链路）
#   ./run_infer_3cam.sh attach       # 重新进入已存在的会话
#   ./run_infer_3cam.sh kill         # 停止全部
#
# 环境变量（可选）:
#   BRUSH_CKPT_DIR           三目权重目录，默认 2026_07_21_task_discrete_b1
#   BRUSH_TASK_NAME          constants.py 中的任务名
#   INFER_CAMERA_NAMES       bridge 相机名（空格分隔），默认 pool scan_2d paper_aruco
#   ROS_DOMAIN_ID            ROS 域，默认 9
#   INFER_MAX_TIMESTEPS      推理步数，默认 150
#   INFER_TARGET_DELTA_GAIN  xyz 位移放大倍数，默认 1
#   INFER_CHUNK_SIZE         ACT chunk_size，须与训练一致，默认 50
#   INFER_MAX_AMPLIFIED_STEP 放大后单步最大位移(m)，默认 0.01
#   INFER_RECORD_CSV         target 记录文件，默认 /tmp/infer_3cam.csv
#   INFER_PANEL_PORT         推理控制页端口，默认 8765
#   INFER_SLEEP_SEC          每步推理间隔(s)，防 safety dropped，默认 0.18
#   INFER_AUTO_START=1       恢复旧行为：infer 窗口 sleep 12 秒后自动启动（默认用手动控制页）
#   PAPER_CAMERA_DEVICE      纸面 USB 相机，默认 /dev/video1（4K HD Camera 采集节点）
#   PAPER_CAMERA_HZ          纸面相机频率，默认 20.0（贴近熔池）
#   PAPER_CAMERA_ZOOM        纸面中心数字变焦，默认 2.0
#   SAFETY_MAX_STEP_M        安全盒单步上限(m)，默认 0.15
#   INFER_RECORD_IMAGES      是否自动保存三目观测图，默认 1（开启）
#   INFER_RECORD_ROOT        录制根目录，默认 ros2_ws/data_collect
#   INFER_TASK_ID            控制页默认 task_id

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION="${SESSION:-brush_infer_3cam}"

MODE="${1:-run}"
ACT_ROOT="/home/shugen/yanjie/act"
ROS2_WS_ROOT="/home/shugen/yanjie/ros2_ws"
AUTO_WELDING_SETUP="${AUTO_WELDING_SETUP:-${HOME}/Documents/auto_welding/install/local_setup.bash}"
ROS2_WS_SETUP="${ROS2_WS_SETUP:-${ROS2_WS_ROOT}/install/local_setup.bash}"

BRUSH_CKPT_DIR="${BRUSH_CKPT_DIR:-/media/shugen/LcxDisk/checkpoint/brush_policy_3cam_2026_07_21_task_discrete_b1}"
BRUSH_TASK_NAME="${BRUSH_TASK_NAME:-brush_tool_pose_2026_07_21}"
INFER_CAMERA_NAMES="${INFER_CAMERA_NAMES:-pool scan_2d paper_aruco}"
POOL1_CAMERA_TOPIC="${POOL1_CAMERA_TOPIC:-/pool_camera1/image_raw}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-9}"
INFER_MAX_TIMESTEPS="${INFER_MAX_TIMESTEPS:-150}"
INFER_TARGET_DELTA_GAIN="${INFER_TARGET_DELTA_GAIN:-1}"
INFER_CHUNK_SIZE="${INFER_CHUNK_SIZE:-50}"
INFER_MAX_AMPLIFIED_STEP="${INFER_MAX_AMPLIFIED_STEP:-0.1}"
INFER_RECORD_CSV="${INFER_RECORD_CSV:-/tmp/infer_3cam.csv}"
INFER_PANEL_PORT="${INFER_PANEL_PORT:-8765}"
INFER_OBSERVATION_TIMEOUT_SEC="${INFER_OBSERVATION_TIMEOUT_SEC:-20}"
INFER_SLEEP_SEC="${INFER_SLEEP_SEC:-0.18}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
INFER_AUTO_START="${INFER_AUTO_START:-0}"
PAPER_CAMERA_DEVICE="${PAPER_CAMERA_DEVICE:-/dev/video0}"
PAPER_CAMERA_HZ="${PAPER_CAMERA_HZ:-20.0}"
PAPER_CAMERA_ZOOM="${PAPER_CAMERA_ZOOM:-2.25}"
SAFETY_MAX_STEP_M="${SAFETY_MAX_STEP_M:-0.15}"
INFER_RECORD_IMAGES="${INFER_RECORD_IMAGES:-1}"
INFER_RECORD_ROOT="${INFER_RECORD_ROOT:-${ROS2_WS_ROOT}/data_collect}"
INFER_TASK_ID="${INFER_TASK_ID:-0}"
INFER_DISCRETE_DECODE="${INFER_DISCRETE_DECODE:-argmax}"
INFER_DISCRETE_TEMPERATURE="${INFER_DISCRETE_TEMPERATURE:-1.0}"
# 0=不给网络喂 TCP/qpos；1=喂当前 TCP。默认跟随上游 export；未设则为 0。
USE_QPOS="${USE_QPOS:-0}"

# 构建 ROS2 环境前缀（必须用系统 Python 3.12；禁止 conda/aloha 的 python3.8）
build_ros_env_prefix() {
  local parts=()
  parts+=("unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH")
  parts+=("export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
  # 若从已 activate 的 conda 环境启动 tmux，PATH 会污染 python3 → rclpy 崩溃
  parts+=("if command -v conda >/dev/null 2>&1; then conda deactivate >/dev/null 2>&1 || true; fi")
  parts+=("export PATH=\"\$(echo \"\$PATH\" | tr ':' '\\n' | grep -v '/miniconda3/\\|anaconda3/' | paste -sd: -)\"")
  parts+=("export PYTHON_BIN=/usr/bin/python3")
  if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
    parts+=("source /opt/ros/jazzy/setup.bash")
  fi
  if [[ -f "${AUTO_WELDING_SETUP}" ]]; then
    parts+=("source '${AUTO_WELDING_SETUP}'")
  fi
  if [[ -f "${ROS2_WS_SETUP}" ]]; then
    parts+=("source '${ROS2_WS_SETUP}'")
  fi
  printf '%s; ' "${parts[@]}"
}

# 构建 aloha 环境前缀（conda，用于 infer）
build_aloha_env_prefix() {
  local parts=()
  parts+=("source \"\$(conda info --base)/etc/profile.d/conda.sh\"")
  parts+=("conda activate aloha")
  parts+=("export BRUSH_CKPT_DIR='${BRUSH_CKPT_DIR}'")
  parts+=("export BRUSH_TASK_NAME='${BRUSH_TASK_NAME}'")
  parts+=("export BRUSH_CKPT_NAME='${BRUSH_CKPT_NAME:-policy_best.ckpt}'")
  parts+=("export INFER_TASK_ID='${INFER_TASK_ID}'")
  parts+=("export PAPER_ARUCO_COLLECT=0")
  printf '%s; ' "${parts[@]}"
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
    cmd_attach
    return
  fi

  local dry_run_flag=""
  local mode_label="正式执行（机械臂会动）"
  if [[ "${MODE}" == "dry-run" ]]; then
    dry_run_flag="--dry-run"
    mode_label="dry-run（机械臂不动）"
  fi

  local ros_env aloha_env
  ros_env="$(build_ros_env_prefix)"
  aloha_env="$(build_aloha_env_prefix)"

  # 终端1: 机器人驱动
  local robot_cmd="${ros_env} cd '${ROS2_WS_ROOT}'; ros2 run welding_runtime robot_driver_bridge_node --ros-args -p robot_type:=duco; echo '[robot] 已退出'; read"

  # 终端2: 3D 相机
  local scan_cmd="${ros_env} cd '${ROS2_WS_ROOT}'; sleep 3; ros2 launch welding_scan3d_camera_driver scan3d_camera.launch.py; echo '[scan] 已退出'; read"

  # 终端3: 熔池相机
  local pool_cmd="${ros_env} cd '${ROS2_WS_ROOT}'; sleep 3; ros2 launch welding_pool_camera_driver pool_camera.launch.py; echo '[pool] 已退出'; read"

  # 终端4: Observation Bridge（相机列表由 INFER_CAMERA_NAMES 指定）
  local record_flags=""
  if [[ "${INFER_RECORD_IMAGES}" == "1" ]]; then
    record_flags="--record-images --record-root '${INFER_RECORD_ROOT}'"
  fi
  local bridge_cmd="${ros_env} export INFER_TASK_ID='${INFER_TASK_ID}'; export BRUSH_CKPT_DIR='${BRUSH_CKPT_DIR}'; export BRUSH_TASK_NAME='${BRUSH_TASK_NAME}'; cd '${ROS2_WS_ROOT}'; sleep 8; \"\${PYTHON_BIN:-/usr/bin/python3}\" scripts/robot_observation_bridge.py --bind tcp://127.0.0.1:5554 --max-observation-age-sec 30.0 --capture-scan --capture-2d-service /capture_2d --capture-service-timeout-sec 5.0 --capture-timeout-sec 10.0 --camera-names ${INFER_CAMERA_NAMES} --pool1-topic '${POOL1_CAMERA_TOPIC}' --paper-camera-device ${PAPER_CAMERA_DEVICE} --paper-camera-hz ${PAPER_CAMERA_HZ} --paper-camera-zoom ${PAPER_CAMERA_ZOOM} ${record_flags}; echo '[bridge] 已退出'; read"

  # 终端5: 安全盒（dry-run 或正式）
  local safety_cmd="${ros_env} cd '${ROS2_WS_ROOT}'; sleep 10; \"\${PYTHON_BIN:-/usr/bin/python3}\" scripts/workspace_safety.py serve-zmq-filter --workspace scripts/workspace_limits.json --pose-topic /tool_pos --real-service /mov_jog --zmq-bind tcp://127.0.0.1:5555 --max-step-m ${SAFETY_MAX_STEP_M} --max-target-age-sec 2.0 ${dry_run_flag}; echo '[safety] 已退出'; read"

  # 终端6: ACT infer（默认 Web 控制页，填完参数后再启动；INFER_AUTO_START=1 恢复自动启动）
  local qpos_flag="--no_qpos"
  if [[ "${USE_QPOS}" == "1" ]]; then
    qpos_flag="--use_qpos"
  fi
  local infer_auto_cmd="sleep 12; bash scripts/run_infer_brush_robot.sh --io_backend zmq --observation_zmq tcp://127.0.0.1:5554 --target_zmq tcp://127.0.0.1:5555 --max_timesteps ${INFER_MAX_TIMESTEPS} --device ${INFER_DEVICE} --capture_scan --observation_timeout_sec ${INFER_OBSERVATION_TIMEOUT_SEC} --debug_chunk --target_delta_gain ${INFER_TARGET_DELTA_GAIN} --max_amplified_step_m ${INFER_MAX_AMPLIFIED_STEP} --sleep_sec ${INFER_SLEEP_SEC} --task_id ${INFER_TASK_ID} --discrete_decode ${INFER_DISCRETE_DECODE} --discrete_temperature ${INFER_DISCRETE_TEMPERATURE} --record_targets_csv ${INFER_RECORD_CSV} ${qpos_flag}; echo '[infer] 已结束'; read"
  local infer_panel_cmd="export ACT_ROOT='${ACT_ROOT}'; export INFER_MODE='${MODE}'; export INFER_OBSERVATION_ZMQ='tcp://127.0.0.1:5554'; export INFER_TARGET_ZMQ='tcp://127.0.0.1:5555'; export INFER_PANEL_PORT='${INFER_PANEL_PORT}'; export INFER_OBSERVATION_TIMEOUT_SEC='${INFER_OBSERVATION_TIMEOUT_SEC}'; export INFER_DEVICE='${INFER_DEVICE}'; export INFER_MAX_TIMESTEPS='${INFER_MAX_TIMESTEPS}'; export INFER_TARGET_DELTA_GAIN='${INFER_TARGET_DELTA_GAIN}'; export INFER_CHUNK_SIZE='${INFER_CHUNK_SIZE}'; export INFER_SLEEP_SEC='${INFER_SLEEP_SEC}'; export INFER_DISCRETE_DECODE='${INFER_DISCRETE_DECODE}'; export INFER_DISCRETE_TEMPERATURE='${INFER_DISCRETE_TEMPERATURE}'; export USE_QPOS='${USE_QPOS}'; export BRUSH_CKPT_DIR='${BRUSH_CKPT_DIR}'; export BRUSH_TASK_NAME='${BRUSH_TASK_NAME}'; cd '${ROS2_WS_ROOT}'; python3 scripts/infer_control_panel.py; echo '[infer] 控制页已退出'; read"
  local infer_cmd="${aloha_env} export INFER_CHUNK_SIZE='${INFER_CHUNK_SIZE}'; export USE_QPOS='${USE_QPOS}'; cd '${ACT_ROOT}'; if [[ '${INFER_AUTO_START}' == '1' ]]; then ${infer_auto_cmd}; else ${infer_panel_cmd}; fi"

  # 终端7: 说明与监控
  local panel_hint="http://127.0.0.1:${INFER_PANEL_PORT}（infer 窗口，填完参数后点启动）"
  local start_hint="infer 不会自动开始。请在控制页填写参数后手动启动：${panel_hint}"
  if [[ "${INFER_AUTO_START}" == "1" ]]; then
    start_hint="INFER_AUTO_START=1：约 12 秒后 infer 自动开始。"
  fi
  local record_hint="INFER_RECORD_IMAGES=${INFER_RECORD_IMAGES} root=${INFER_RECORD_ROOT}"
  if [[ "${INFER_RECORD_IMAGES}" == "1" ]]; then
    record_hint="${record_hint}（bridge 会写 YYYY-MM-DD/HH-MM-SS_infer/）"
  fi
  local monitor_cmd="echo '多相机推理一键启动 — ${mode_label}'; echo; echo '窗口: robot | scan | pool | bridge | safety | infer | info'; echo; echo '${start_hint}'; echo '查看链路: bridge 窗口 observed 应包含 pose 和全部相机'; echo '  CAMERAS=${INFER_CAMERA_NAMES}'; echo 'safety 窗口: 每步打印 Δxyz(mm) / |Δ| / Δrpy；dry-run=\"dry-run 安全通过\"，正式=\"转发安全目标\"'; echo '录制: ${record_hint}'; echo; echo '退出但不停止: Ctrl+B 然后 D'; echo '停止全部: ${SCRIPT_DIR}/run_infer_3cam.sh kill'; echo; echo '默认参数:'; echo '  TASK_NAME=${BRUSH_TASK_NAME}'; echo '  CKPT_DIR=${BRUSH_CKPT_DIR}'; echo '  TIMESTEPS=${INFER_MAX_TIMESTEPS} GAIN=${INFER_TARGET_DELTA_GAIN} CHUNK=${INFER_CHUNK_SIZE} SLEEP=${INFER_SLEEP_SEC}s TASK_ID=${INFER_TASK_ID}'; echo '  DECODE=${INFER_DISCRETE_DECODE} TEMPERATURE=${INFER_DISCRETE_TEMPERATURE} USE_QPOS=${USE_QPOS}'; echo '  ROS_DOMAIN_ID=${ROS_DOMAIN_ID} PAPER_DEV=${PAPER_CAMERA_DEVICE}'; exec bash"

  tmux new-session -d -s "${SESSION}" -n robot   "bash -lc $(printf '%q' "${robot_cmd}")"
  tmux new-window  -t "${SESSION}" -n scan       "bash -lc $(printf '%q' "${scan_cmd}")"
  tmux new-window  -t "${SESSION}" -n pool       "bash -lc $(printf '%q' "${pool_cmd}")"
  tmux new-window  -t "${SESSION}" -n bridge     "bash -lc $(printf '%q' "${bridge_cmd}")"
  tmux new-window  -t "${SESSION}" -n safety     "bash -lc $(printf '%q' "${safety_cmd}")"
  tmux new-window  -t "${SESSION}" -n infer      "bash -lc $(printf '%q' "${infer_cmd}")"
  tmux new-window  -t "${SESSION}" -n info       "bash -lc $(printf '%q' "${monitor_cmd}")"

  echo "已创建 tmux 会话: ${SESSION}（${mode_label}）"
  echo "窗口: robot | scan | pool | bridge | safety | infer | info"
  echo
  if [[ "${INFER_AUTO_START}" == "1" ]]; then
    echo "INFER_AUTO_START=1：约 12 秒后 infer 自动开始。"
  else
    echo "infer 不会自动开始。请在浏览器打开控制页填写参数后启动："
    echo "  http://127.0.0.1:${INFER_PANEL_PORT}"
  fi
  echo "首次建议先用 dry-run 验证链路："
  echo "  ${SCRIPT_DIR}/run_infer_3cam.sh dry-run"
  echo
  echo "退出但不停止: Ctrl+B 然后 D"
  echo "停止全部: ${SCRIPT_DIR}/run_infer_3cam.sh kill"
  if [[ -t 1 ]]; then
    cmd_attach
  else
    echo "当前无交互终端，跳过 attach。进入会话: ${SCRIPT_DIR}/run_infer_3cam.sh attach"
  fi
}

case "${MODE}" in
  start|run|"")
    cmd_start
    ;;
  dry-run|dryrun)
    MODE="dry-run"
    cmd_start
    ;;
  attach)
    cmd_attach
    ;;
  kill|stop)
    cmd_kill
    ;;
  *)
    echo "用法: $0 [run|dry-run|attach|kill]"
    echo "  run       正式执行（默认，机械臂会动）"
    echo "  dry-run   只验证链路，机械臂不动"
    echo "  attach    重新进入会话"
    echo "  kill      停止全部"
    exit 1
    ;;
esac
