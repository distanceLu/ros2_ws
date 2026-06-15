1.第一个终端开启熔池与3D相机

source ~/Documents/auto_welding/install/setup.bash
source ~/ros2_ws/install/setup.bash

python3 ~/ros2_ws/scripts/camera_capture_node.py --ros-args \
  -p auto_start_camera_keys:=3d,pool \
  -p keep_launched_drivers_on_exit:=true


2.第二个终端开启机械臂 + 遥操

source ~/Documents/auto_welding/install/setup.bash
ros2 run welding_runtime robot_driver_bridge_node --ros-args -p robot_type:=duco
# 另开终端启用 Inverse3 / teleop_enable


3.开启第三个终端 运行采集节点

source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/scripts/training_data_collect.py


4.开启第四个终端 运行交互式采集工具

~/ros2_ws/scripts/training_collect.sh

交互工具具体用法：
- `h` — 回初始位（带随机偏移）
- `s` / `x` — 开始 / 停止采集
- `e` — **单轮**：回初始位 → 开始采集 → 遥操 → **Enter 停止**
- `r` — 连续多轮 episode
- `p` — 查看当前位姿与初始位距离

一般的流程为： h自动恢复到原点，s开始轨迹采集，使用示教机通过调节运动到目标位置，x停止采集，h自动恢复到原点



