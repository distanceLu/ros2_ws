# 3D 相机无激光 2D 拍照

复现 RVCManager 中「2D相机使用光机 = 否」的行为。

## 目录说明

| 路径 | 说明 |
|---|---|
| `originals/` | 修改前的原始文件备份 |
| `modified/` | 已修改的完整文件副本 |
| `apply_and_build.sh` | 将 modified 应用到源码并编译驱动 |

## 改动文件

1. `Documents/auto_welding/.../scan_camera_node.cpp`
   - 新增参数 `use_projector_capturing_2d_image`（默认 `false`）
   - 新增 `Capture2D()` 路径
   - 新增 ROS 服务 `/capture_2d`

2. `ros2_ws/scripts/camera_capture_node.py`
   - 新增服务 `/camera_capture/capture_once_3d_2d`
   - 拍照前自动 `/stop_capture`，再调 `/capture_2d`
   - 图片保存到 `camera_images/3d_2d/`

## 一次性部署

```bash
bash /home/shugen/ros2_ws/scripts/camera_3d_capture2d/apply_and_build.sh
source ~/Documents/auto_welding/install/setup.bash
source ~/ros2_ws/install/setup.bash
```

## 使用方式

### 方式 A：统一拍照脚本（推荐）

终端 1：

```bash
source ~/Documents/auto_welding/install/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/scripts/camera_capture_node.py
```

终端 2：

```bash
source ~/Documents/auto_welding/install/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 service call /camera_capture/capture_once_3d_2d std_srvs/srv/Trigger {}
```

成功返回示例：

```text
success=True, message='[3D相机纯2D] Saved 3D camera 2D image via /capture_2d (no projector): ...'
```

### 方式 B：直接调驱动服务

需先确保 `scan_camera_node` 在运行，且已重新编译：

```bash
ros2 service call /stop_capture std_srvs/srv/Trigger {}
ros2 service call /capture_2d std_srvs/srv/Trigger {}
```

图像会发布到 `/scan/image_raw`。

## 与线扫拍照的区别

| 服务 | 激光 | SDK API | 保存目录 |
|---|---|---|---|
| `/camera_capture/capture_once_3d` | 有（线扫） | `Capture()` | `camera_images/3d/` |
| `/camera_capture/capture_once_3d_2d` | 无 | `Capture2D()` | `camera_images/3d_2d/` |

## 重启 3D 驱动

修改驱动后必须重启 `scan_camera_node`：

```bash
pkill -f scan_camera_node
ros2 launch welding_scan3d_camera_driver scan3d_camera.launch.py auto_capture:=false
```
