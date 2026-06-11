#!/usr/bin/env bash
# 训练数据采集便捷脚本（仅调用 scripts/ 下 Python 工具，不修改其他目录）
#
# 用法:
#   ./scripts/training_collect.sh              # 交互菜单
#   ./scripts/training_collect.sh home         # 回初始位（带随机偏移）
#   ./scripts/training_collect.sh episode      # 单轮: 回初始位 → 采集 → Enter 停止
#   ./scripts/training_collect.sh episode -n 5 # 连续 5 轮
#   ./scripts/training_collect.sh start|stop|status
#
# 环境变量:
#   TRAINING_CONFIG  自定义 JSON 配置路径
#   AUTO_WELDING_SETUP  auto_welding install/setup.bash（默认 ~/Documents/auto_welding/install/setup.bash）
#   ROS2_WS_SETUP      ros2_ws install/setup.bash（默认 ~/ros2_ws/install/setup.bash）

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AUTO_WELDING_SETUP="${AUTO_WELDING_SETUP:-${HOME}/Documents/auto_welding/install/setup.bash}"
ROS2_WS_SETUP="${ROS2_WS_SETUP:-${WS_ROOT}/install/setup.bash}"
CONFIG="${TRAINING_CONFIG:-${SCRIPT_DIR}/training_session_config.json}"

# colcon setup.bash 会读取未定义的 COLCON_TRACE 等变量，source 期间关闭 nounset
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

CMD="${1:-interactive}"
shift || true

exec python3 "${SCRIPT_DIR}/training_session.py" -c "${CONFIG}" "${CMD}" "$@"
