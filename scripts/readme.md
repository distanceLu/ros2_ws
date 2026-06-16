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

含义：`/capture_2d` 服务调用成功了，但 3D 驱动没有在 `/scan/image_raw` 发布新图。熔池相机能拍成功时，说明熔池链路没问题，问题集中在 3D 驱动或当前运行环境。

先重启旧会话，避免使用修复前启动的节点：

```bash
/home/shugen/yanjie/ros2_ws/scripts/collect_data.sh kill
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
| `collect` | 启动 `training_data_collect.py`，负责保存图像、TCP 位姿、遥操速度 |
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
  robot_state/
    tool_pose.csv       TCP 位姿，来自 /tool_pos
    control_speed.csv   遥操速度，来自 /spacenav/twist
  session_meta.json
  episode_home_pose.json
```

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
