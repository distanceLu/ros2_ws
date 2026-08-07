#!/usr/bin/env bash
# 示教器「命令目标 vs 实际 TCP」对照实验记录器
#
# 目的：验证 Duco SDK 的 get_tcp_pose_command / getRobotStatus
# 在示教器步进时，是否能保留“想去的地方”（例如命令 5mm、实际只走 1mm）。
#
# 用法：
#   ./probe_duco_expected_vs_actual.sh              # 默认 50Hz，录到 data_collect/probe_...
#   ./probe_duco_expected_vs_actual.sh --hz 100
#   ./probe_duco_expected_vs_actual.sh --ip 192.168.1.10
#   ./probe_duco_expected_vs_actual.sh --connect-only   # 只测连接
#   ./probe_duco_expected_vs_actual.sh --rebuild        # 强制重编译
#
# 重要：
#   1) 开始前请先停掉占用 RPC 的 bridge，例如：
#        ros2_ws/scripts/collect_data.sh kill
#      或手动停 robot_driver_bridge_node
#   2) 本脚本只读状态，不发运动指令，也不 power_on/enable
#   3) 录制中用示教器做两组动作：
#        A. 空载步进约 5mm
#        B. 安全软阻挡/受阻步进约 5mm（只实际走出约 1mm）
#   4) Ctrl+C 结束；结束后看 csv 里 lin_err_mm / status_lin_err_mm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTO_WELDING_ROOT="${AUTO_WELDING_ROOT:-${HOME}/Documents/auto_welding}"
DUCO_INCLUDE="${DUCO_INCLUDE:-${AUTO_WELDING_ROOT}/third_party/duco/include}"
DUCO_LIB="${DUCO_LIB:-${AUTO_WELDING_ROOT}/third_party/duco/lib}"
BUILD_DIR="${BUILD_DIR:-${WS_ROOT}/scripts/.build_probe}"
BIN="${BUILD_DIR}/probe_duco_expected_vs_actual"
SRC="${SCRIPT_DIR}/probe_duco_expected_vs_actual.cpp"

ROBOT_IP="${ROBOT_IP:-192.168.1.10}"
ROBOT_PORT="${ROBOT_PORT:-7003}"
HZ="${HZ:-50}"
REBUILD=0
CONNECT_ONLY=0
NO_STATUS=0
FORCE_YES="${FORCE:-0}"
OUT_CSV=""
EXTRA_ARGS=()

print_usage() {
  cat <<EOF
用法: $0 [选项]

  --ip IP            机械臂 IP（默认 ${ROBOT_IP}）
  --port PORT        RPC 端口（默认 ${ROBOT_PORT}）
  --hz HZ            采样频率（默认 ${HZ}）
  --out PATH.csv     指定输出 csv；默认自动放到 data_collect/probe_...
  --no-status        不调用 getRobotStatus，只采 get_tcp_pose(_command)
  --connect-only     只测 open/close
  --rebuild          强制重新编译
  -y|--yes           bridge 仍在跑时不询问，直接继续
  -h|--help          帮助

环境变量: AUTO_WELDING_ROOT / DUCO_INCLUDE / DUCO_LIB / ROBOT_IP / ROBOT_PORT / HZ / FORCE=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ip)
      ROBOT_IP="${2:?}"; shift 2 ;;
    --port)
      ROBOT_PORT="${2:?}"; shift 2 ;;
    --hz)
      HZ="${2:?}"; shift 2 ;;
    --out)
      OUT_CSV="${2:?}"; shift 2 ;;
    --no-status)
      NO_STATUS=1; shift ;;
    --connect-only)
      CONNECT_ONLY=1; shift ;;
    --rebuild)
      REBUILD=1; shift ;;
    -y|--yes)
      FORCE_YES=1; shift ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "${SRC}" ]]; then
  echo "错误: 找不到源码 ${SRC}" >&2
  exit 1
fi
if [[ ! -f "${DUCO_LIB}/libDucoCobotAPI.so" ]]; then
  echo "错误: 找不到 ${DUCO_LIB}/libDucoCobotAPI.so" >&2
  exit 1
fi
if [[ ! -f "${DUCO_INCLUDE}/robot_control/DucoCobot.h" ]]; then
  echo "错误: 找不到 ${DUCO_INCLUDE}/robot_control/DucoCobot.h" >&2
  exit 1
fi

need_build=0
if [[ "${REBUILD}" -eq 1 || ! -x "${BIN}" ]]; then
  need_build=1
elif [[ "${SRC}" -nt "${BIN}" ]]; then
  need_build=1
fi

if [[ "${need_build}" -eq 1 ]]; then
  echo "[build] compiling probe ..."
  mkdir -p "${BUILD_DIR}"
  g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    -I"${DUCO_INCLUDE}" \
    "${SRC}" \
    -L"${DUCO_LIB}" -Wl,-rpath,"${DUCO_LIB}" \
    -lDucoCobotAPI -lpthread \
    -o "${BIN}"
  echo "[build] ok: ${BIN}"
fi

# 只匹配真正的 bridge 可执行进程；tmux 中退出后停在 read 的旧 bash
# 命令行仍含 robot_driver_bridge_node，不能据此误报。
if pgrep -x 'robot_driver_br' >/dev/null 2>&1; then
  echo "警告: 检测到 robot_driver_bridge_node 正在运行。" >&2
  echo "      Duco RPC 通常只允许一个客户端；建议先停掉 bridge 再测：" >&2
  echo "        ${SCRIPT_DIR}/collect_data.sh kill" >&2
  echo "      或: pkill -f robot_driver_bridge_node" >&2
  if [[ "${FORCE_YES}" == "1" ]]; then
    echo "      已指定 --yes/FORCE=1，继续尝试连接。" >&2
  elif [[ ! -t 0 ]]; then
    echo "错误: 非交互终端且未指定 --yes；请先停 bridge，或加 --yes。" >&2
    exit 1
  else
    read -r -p "仍要继续尝试连接？[y/N] " ans
    case "${ans}" in
      y|Y|yes|YES) ;;
      *) echo "已取消"; exit 1 ;;
    esac
  fi
fi

stamp="$(date +%Y-%m-%d/%H-%M-%S)"
if [[ -z "${OUT_CSV}" ]]; then
  OUT_DIR="${WS_ROOT}/data_collect/probe_expected_vs_actual/${stamp}"
  mkdir -p "${OUT_DIR}"
  OUT_CSV="${OUT_DIR}/expected_vs_actual.csv"
else
  OUT_DIR="$(dirname "${OUT_CSV}")"
  mkdir -p "${OUT_DIR}"
fi

META="${OUT_DIR}/probe_meta.json"
cat > "${META}" <<EOF
{
  "created_at": "$(date -Iseconds)",
  "robot_ip": "${ROBOT_IP}",
  "robot_port": ${ROBOT_PORT},
  "hz": ${HZ},
  "out_csv": "${OUT_CSV}",
  "duco_lib": "${DUCO_LIB}/libDucoCobotAPI.so",
  "use_status": $([ "${NO_STATUS}" -eq 1 ] && echo false || echo true),
  "note": "Compare command vs actual while using teach pendant. Look at lin_err_mm / status_lin_err_mm."
}
EOF

echo "=============================================="
echo "  Duco expected vs actual probe"
echo "=============================================="
echo "IP/Port : ${ROBOT_IP}:${ROBOT_PORT}"
echo "Hz      : ${HZ}"
echo "CSV     : ${OUT_CSV}"
echo "Meta    : ${META}"
echo
echo "建议测试流程："
echo "  1) 示教器空载，沿某一轴步进约 5mm"
echo "  2) 安全软阻挡/受阻，再步进约 5mm（实际尽量只走约 1mm）"
echo "  3) Ctrl+C 结束"
echo "  4) 用下面命令看最大跟踪误差："
echo "       python3 - <<'PY'"
echo "       import csv,statistics"
echo "       p='${OUT_CSV}'"
echo "       rows=list(csv.DictReader(open(p)))"
echo "       errs=[float(r['lin_err_mm']) for r in rows if r['tcp_valid']=='1']"
echo "       serrs=[float(r['status_lin_err_mm']) for r in rows if r['status_valid']=='1']"
echo "       print('rows',len(rows))"
echo "       print('lin_err_mm max/median', (max(errs),statistics.median(errs)) if errs else 'no valid rows')"
echo "       print('status_lin_err_mm max/median', (max(serrs),statistics.median(serrs)) if serrs else 'no valid rows')"
echo "       PY"
echo "=============================================="

cmd=(
  "${BIN}"
  --ip "${ROBOT_IP}"
  --port "${ROBOT_PORT}"
  --hz "${HZ}"
  --out "${OUT_CSV}"
)
if [[ "${NO_STATUS}" -eq 1 ]]; then
  cmd+=(--no-status)
fi
if [[ "${CONNECT_ONLY}" -eq 1 ]]; then
  cmd+=(--connect-only)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

export LD_LIBRARY_PATH="${DUCO_LIB}:${LD_LIBRARY_PATH:-}"
exec "${cmd[@]}"
