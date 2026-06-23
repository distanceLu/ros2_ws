注意由于轨迹文件较大并且本地磁盘太小请将轨迹上传
ossutil sync /home/shugen/yanjie/ros2_ws/data_collect oss://gpu-ai/lcx/brush_pen/data_collect/


# 数据采集与拍照流程



## 功能一：一键拍照（熔池 + 3D 纯 2D）

用于快速拍一组图，不启动机械臂，不采集轨迹。

### 直接拍照

```bash
/home/shugen/yanjie/ros2_ws/scripts/snap_pool_3d2d.sh
```

脚本会自动完成：

1. 若 `camera_capture_node` 未运行，则后台启动相机节点，只启动 `pool,3d`。
2. 调用 `/camera_capture/capture_once_3d_2d`，保存 3D 相机纯 2D 图。
3. 调用 `/camera_capture/capture_once_pool`，保存熔池图。
4. 若两路都返回 `success=True`，脚本打印完成；任一路返回 `success=False` 会按失败处理。
5. 成功且由脚本启动相机时，默认保留 `camera_capture_node` 后台运行，便于连续拍照。

### 相机节点已运行时

```bash
/home/shugen/yanjie/ros2_ws/scripts/snap_pool_3d2d.sh --no-start
```

### 只启动相机，稍后再拍

```bash
/home/shugen/yanjie/ros2_ws/scripts/snap_pool_3d2d.sh --daemon
/home/shugen/yanjie/ros2_ws/scripts/snap_pool_3d2d.sh --no-start
```

### 拍照输出

当前 `camera_capture_node.py` 默认保存到：

```text
/home/shugen/yanjie/ros2_ws/camera_images/3d_2d/YYYY-MM-DD/HH-MM-SS/
/home/shugen/yanjie/ros2_ws/camera_images/pool/YYYY-MM-DD/HH-MM-SS/
```

`ros2 service call` 成功返回的 `message` 里也会包含完整图片路径。

### 停止后台相机节点

如果由 `snap_pool_3d2d.sh` 自动拉起相机节点，脚本结束时会打印 pid：

```bash
kill <pid>
```

自动启动日志：

```text
/tmp/snap_pool_3d2d_camera.log
```

### 3D 纯 2D 拍照失败时

典型错误：

```text
/capture_2d succeeded but no new image on /scan/image_raw within 10.0s
```

含义：`/capture_2d` 服务调用成功了，但 3D 驱动没有在 `/scan/image_raw` 发布新图。训练采集优先使用 `/capture_2d_image` 直接取图，可避开这个发布链路问题。熔池相机能拍成功时，说明熔池链路没问题，问题集中在 3D 驱动或当前运行环境。

训练采集节点 `training_data_collect.py` 现已优先使用 `/capture_2d_image`：该服务调用无投影 `Capture2D` 并直接返回图像，因此不依赖 `/scan/image_raw` 是否发布，也不会调用 `/scan_3d` 打开扫描光。停止采集时会输出 `3d_2d_fail` 计数。修改采集逻辑后需重启采集会话，避免使用修复前启动的节点：

```bash
/home/shugen/yanjie/ros2_ws/scripts/collect_data.sh kill
/home/shugen/yanjie/ros2_ws/scripts/collect_data.sh
```

确认相关节点是否还残留：

```bash
ros2 node list
```

如果还看到这些节点，说明旧进程仍在，需要关闭对应终端或手动 kill：

```text
/unified_camera_capture_node
/scan_camera_node
/pool_camera_node
/training_data_collect_node
```

只启动拍照相机并复测：

```bash
/home/shugen/yanjie/ros2_ws/scripts/snap_pool_3d2d.sh --daemon
/home/shugen/yanjie/ros2_ws/scripts/snap_pool_3d2d.sh --no-start
```

检查 3D 图像 topic 是否真的有消息：

```bash
source /opt/ros/jazzy/setup.bash
source /home/shugen/Documents/auto_welding/install/local_setup.bash
source /home/shugen/yanjie/ros2_ws/install/local_setup.bash

ros2 topic info /scan/image_raw -v
ros2 service call /capture_2d std_srvs/srv/Trigger {}
ros2 topic echo --once --qos-reliability best_effort /scan/image_raw
```

如果 `/capture_2d` 返回 `success=True`，但 `ros2 topic echo` 仍然超时，说明 `scan_camera_node` 没有实际发布图像。此时查看 camera 窗口或日志，重点找 `Capture2D`、`GetImage`、`publish_image`、`RVC` 相关输出。

## 功能二：一键采集训练轨迹

用于采集小模型训练数据。该流程会同时启动相机、机器人桥接、数据落盘节点和交互控制界面。

### 一键启动

```bash
/home/shugen/yanjie/ros2_ws/scripts/collect_data.sh
```

脚本会创建 tmux 会话 `welding_collect`，包含窗口：

| 窗口 | 作用 |
| --- | --- |
| `camera` | 启动 `camera_capture_node.py`，只起 `3d,pool` |
| `robot` | 启动 `welding_runtime robot_driver_bridge_node`，提供 `/mov_jog` 和 `/tool_pos` |
| `collect` | 启动 `training_data_collect.py`，负责保存熔池图、3D 2D 图、纸面相机图、TCP 位姿、遥操速度 |
| `session` | 启动 `training_collect.sh`，用于回初始位和控制 episode |
| `monitor` | 打印相机监控和服务检查命令 |

### tmux 操作

如果已经在 tmux 里面，再执行 `collect_data.sh` 会提示：

```text
sessions should be nested with care
```

这不是采集失败，而是不要在 tmux 里面再 attach tmux。正确做法是在当前 tmux 中切窗口：

```text
Ctrl+B，然后按 W
```

选择 `session` 窗口。只有看到 `>>>` 后，`h/r/e/s/x` 才是采集命令。若在普通 shell 里输入 `h`，会得到 `h: command not found`。

退出 tmux 但不停止采集：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
/home/shugen/yanjie/ros2_ws/scripts/collect_data.sh attach
```

停止全部采集相关窗口：

```bash
/home/shugen/yanjie/ros2_ws/scripts/collect_data.sh kill
```

### 采集 10 条轨迹

进入 tmux 的 `session` 窗口，看到 `>>>` 后输入：

```text
r
连续几轮? 10
```

每一轮流程：

1. 自动调用 `/mov_jog` 回到配置的初始位姿附近。
2. 自动调用 `/training_data_collect_activate` 开始保存数据。
3. 使用遥操完成一条轨迹。
4. 在提示 `3/3 采集中 — 遥操完成后按 Enter 停止...` 时按 Enter。
5. 自动调用 `/training_data_collect_deactivate` 停止本轮保存。
6. 按提示继续下一轮。

### 数据输出

```text
/home/shugen/yanjie/ros2_ws/data_collect/YYYY-MM-DD/HH-MM-SS/
  camera_pool/          熔池图
  camera_3d_2d/         3D 相机无激光 2D 图
  camera_paper_aruco/   纸面工控机 USB 相机图
  paper_state/
    paper_aruco_pose.csv  每张纸面图的时间索引；默认不做实时 ArUco 定位
  robot_state/
    tool_pose.csv       TCP 位姿，来自 /tool_pos
    control_speed.csv   遥操速度，来自 /spacenav/twist
  session_meta.json
  episode_home_pose.json
```

### 图像文件名格式

熔池、3D 2D、纸面相机三路图像使用相同命名规则：

```text
当天微秒时间戳.计数.jpg
```

示例：

```text
71366530114.000001.jpg
71366861003.000001.jpg
```

训练转换时，`convert_brush_data_to_act_hdf5.py` 会读取文件名前半段作为 timestamp，并按 pool 时间轴对齐 scan_2d 与 paper_aruco。

### 纸面相机说明

- 设备默认：`/dev/video0`
- 默认采集频率：`15 Hz`（与熔池相机保存频率接近；可通过环境变量 `PAPER_CAMERA_HZ` 调整）
- 默认分辨率：`3840 x 2160`
- 采集由 `training_data_collect.py` 在 `/training_data_collect_activate` 后自动启动
- 不需要单独再开 `paper_aruco_localize.py`
- 默认只保存原始 JPG，不在采集时实时做 ArUco 定位，避免拖慢采集频率

如果本次采集不需要纸面相机，可在启动采集节点时关闭：

```bash
python3 scripts/training_data_collect.py --ros-args -p enable_paper_camera:=false
```

常用参数：

```bash
python3 scripts/training_data_collect.py --ros-args \
  -p enable_paper_camera:=true \
  -p paper_camera_device:=/dev/video0 \
  -p paper_camera_hz:=15.0 \
  -p paper_camera_width:=3840 \
  -p paper_camera_height:=2160
```

一键采集时也可在启动前设置：

```bash
PAPER_CAMERA_HZ=15.0 /home/shugen/yanjie/ros2_ws/scripts/collect_data.sh
```

如果确实需要在采集时同步写入 ArUco 检测结果，可显式打开实时定位：

```bash
python3 scripts/training_data_collect.py --ros-args \
  -p paper_camera_localize:=true
```

注意：实时定位会对每张 4K 图做 ArUco 检测，可能显著降低 `camera_paper_aruco/` 的保存速度。训练数据采集一般保持默认 `false`，只保留原始照片即可。

### 转换为 ACT HDF5

采集完成后，如果 session 中存在 `camera_paper_aruco/`，转换脚本会自动写入 HDF5 的 `observations/images/paper_aruco`：

```bash
cd /home/shugen/yanjie/act
conda activate aloha

python3 scripts/convert_brush_data_to_act_hdf5.py \
  --raw_dir /home/shugen/yanjie/ros2_ws/data_collect/YYYY-MM-DD \
  --out_dir /home/shugen/yanjie/act/data/brush_hdf5/YYYY-MM-DD \
  --overwrite
```

注意：当前已训练好的 ACT 模型仍只使用 `pool` 和 `scan_2d`。要把 `paper_aruco` 真正用于训练，还需要后续更新 `constants.py` 里的 `camera_names` 并重新训练。

### 相机画面监控

在 `monitor` 窗口或任意已 source 环境的终端运行：

```bash
ros2 run image_view image_view --ros-args -r image:=/pool_camera/image_raw
ros2 run image_view image_view --ros-args -r image:=/scan/image_raw
```

### 服务与节点检查

如果 `session` 窗口提示服务未就绪，可以检查：

```bash
ros2 node list
ros2 service list | grep -E "mov_jog|training_data|capture_2d"
ros2 node info /robot_driver_bridge
ros2 topic info /pool_camera/image_raw -v
ros2 topic info /scan/image_raw -v
```

正常情况下：

```text
/mov_jog                 common_interface/srv/Move
/tool_pos                common_interface/msg/TcpPos
/capture_2d              std_srvs/srv/Trigger
/training_data_collect_activate
/training_data_collect_deactivate
```

## 功能三：机械臂 TCP 安全区

`workspace_safety.py` 用于给模型输出的 TCP 位姿加一层软件安全过滤。当前策略是限制 TCP 的 `x/y/z` 工作空间，不直接用 `rx/ry/rz` 欧拉角做硬限制，因为欧拉角存在正负绕回和多解问题。姿态如需限制，后续建议改为工具轴夹角或四元数角距离。

安全区由示教器取点生成：先记录 `P0/Px/Py` 建立任务坐标系，再记录工作区边界点。运行时会把模型目标点和路径采样点转换到该任务坐标系内，检查是否落在允许盒子中。当前配置建议 `x/y` 保留少量安全距离，`z` 安全距离可设为 `0`，避免末端碰不到目标物体。

### 示教安全区

```bash
cd /home/shugen/yanjie/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash

python3 scripts/workspace_safety.py teach \
  --out scripts/workspace_limits.json \
  --margin-mm 0 \
  --tool-clearance-mm 5
```

示教完成后确认 `scripts/workspace_limits.json` 中策略为：

```json
"position_clearance_m": {
  "x": 0.005,
  "y": 0.005,
  "z": 0.0
},
"check_orientation": false
```

### 检查与自测

```bash
python3 scripts/workspace_safety.py describe --workspace scripts/workspace_limits.json
python3 scripts/workspace_safety.py self-test --workspace scripts/workspace_limits.json
```

`describe` 用于查看原始范围、有效范围和安全距离；`self-test` 会自动测试内部点、边界点和越界点，确认非法点能被拦截。

### 生成并验证巡检点

```bash
python3 scripts/workspace_safety.py export-inspection-csv \
  --workspace scripts/workspace_limits.json \
  --out scripts/workspace_inspection_waypoints.csv \
  --inset-mm 30

python3 scripts/workspace_safety.py run-inspection \
  --workspace scripts/workspace_limits.json \
  --waypoints scripts/workspace_inspection_waypoints.csv \
  --skip-current-check
```

### 实机绕安全区运行

确认 `/tool_pos` 和 `/mov_jog` 正常后执行：

```bash
python3 scripts/workspace_safety.py run-inspection \
  --workspace scripts/workspace_limits.json \
  --waypoints scripts/workspace_inspection_waypoints.csv \
  --service /mov_jog \
  --allow-entry-from-raw \
  --entry-tolerance-mm 1 \
  --execute
```

第一次实机测试不要加 `--yes`。脚本会先要求输入 `RUN`，并在每个巡检点移动前等待回车确认。测试时使用低速，保持急停可用，若运动方向或高度异常应立即停止。
