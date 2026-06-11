#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shugen/ros2_ws/scripts/camera_3d_capture2d"
DRIVER_SRC="/home/shugen/Documents/auto_welding/src/nodes/welding_scan3d_camera_driver/src/scan_camera_node.cpp"
SCRIPT_DST="/home/shugen/ros2_ws/scripts/camera_capture_node.py"
AUTO_WELDING_WS="/home/shugen/Documents/auto_welding"

echo "[1/3] Apply modified driver and script from ${ROOT}/modified"
cp "${ROOT}/modified/scan_camera_node.cpp" "${DRIVER_SRC}"
cp "${ROOT}/modified/camera_capture_node.py" "${SCRIPT_DST}"

echo "[2/3] Build welding_scan3d_camera_driver"
cd "${AUTO_WELDING_WS}"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select welding_scan3d_camera_driver

echo "[3/3] Done. Source workspace before use:"
echo "  source ${AUTO_WELDING_WS}/install/setup.bash"
echo "  source /home/shugen/ros2_ws/install/setup.bash"
