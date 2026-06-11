#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_data_collect.py

面向小模型训练的同步数据采集：
  - 熔池相机：/pool_camera/image_raw（连续保存）
  - 3D 相机 2D 图：定时调用 /capture_2d（无激光，等同 RVCManager）
  - 机械臂：/tool_pos、/joint_pos
  - 遥操动作：/spacenav/twist

与旧版 data_collect.py 的区别：
  - 不再依赖 /image_topic0（MindVision 2D）
  - 使用 3D 相机纯 2D + 熔池相机组成双视角图像集
  - 修复 joint_pos 消息类型（JointPos）
  - 记录遥操 twist 作为 action

使用方式：
  终端1：启动相机与机械臂（不要 run.launch.py）
    python3 scripts/camera_capture_node.py --ros-args \\
      -p auto_start_camera_keys:=3d,pool \\
      -p keep_launched_drivers_on_exit:=true
    ros2 run welding_runtime robot_driver_bridge_node   # 或 ros2 run robot_control robot_control_node

  终端2：启动本采集节点
    source ~/Documents/auto_welding/install/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 scripts/training_data_collect.py

  终端3：开始/停止采集
    ros2 service call /training_data_collect_activate std_srvs/srv/Trigger {}
    ros2 service call /training_data_collect_deactivate std_srvs/srv/Trigger {}

数据目录：
  ~/ros2_ws/data_collect/YYYY-MM-DD/HH-MM-SS/
    camera_pool/          熔池图
    camera_3d_2d/         3D 相机无激光 2D 图
    robot_state/          tool_pose.csv, joint_state.csv, control_speed.csv
    session_meta.json     采集参数摘要
"""

from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from robot_control.msg import JointPos, JogPos
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


def imgmsg_to_bgr8(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if encoding == "rgb8":
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if encoding in ("mono8", "8uc1"):
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if encoding == "bgra8":
        bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def elapsed_microseconds(save_date: str) -> int:
    current = datetime.now()
    target = datetime.strptime(save_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    return int((current - target).total_seconds() * 1_000_000)


class TrainingDataCollectNode(Node):
    def __init__(self):
        super().__init__("training_data_collect_node")
        self.callback_group = ReentrantCallbackGroup()

        self.save_dir_root = Path(
            self.declare_parameter("save_dir_root", "/home/shugen/ros2_ws/data_collect").value
        )
        self.pool_topic = self.declare_parameter("pool_camera_topic", "/pool_camera/image_raw").value
        self.scan_topic = self.declare_parameter("scan_image_topic", "/scan/image_raw").value
        self.capture_2d_service = self.declare_parameter("capture_2d_service", "/capture_2d").value
        self.capture_3d_2d_hz = float(self.declare_parameter("capture_3d_2d_hz", 5.0).value)
        self.save_every_n_pool = max(1, int(self.declare_parameter("save_every_n_pool", 1).value))

        self.run_mode = False
        self.save_date: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.pool_dir: Optional[Path] = None
        self.scan_dir: Optional[Path] = None
        self.robot_dir: Optional[Path] = None

        self._pool_count = 0
        self._scan_count = 0
        self._tool_count = 0
        self._joint_count = 0
        self._twist_count = 0
        self._scan_recv_count = 0
        self._scan_lock = threading.Lock()

        self.capture_2d_client = self.create_client(
            Trigger, self.capture_2d_service, callback_group=self.callback_group
        )

        self.create_subscription(
            Image,
            self.pool_topic,
            self._cb_pool_image,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Image,
            self.scan_topic,
            self._cb_scan_image,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            JogPos,
            "/tool_pos",
            self._cb_tool_pose,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            JointPos,
            "/joint_pos",
            self._cb_joint_state,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Twist,
            "/spacenav/twist",
            self._cb_twist,
            50,
            callback_group=self.callback_group,
        )

        self.create_service(
            Trigger,
            "/training_data_collect_activate",
            self._activate_callback,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/training_data_collect_deactivate",
            self._deactivate_callback,
            callback_group=self.callback_group,
        )

        self._scan_timer = self.create_timer(
            1.0 / max(0.5, self.capture_3d_2d_hz),
            self._trigger_capture_2d,
            callback_group=self.callback_group,
        )
        self._scan_timer.cancel()

        self.get_logger().info("Training data collect node started.")
        self.get_logger().info(f"Save root: {self.save_dir_root}")
        self.get_logger().info(f"Pool topic: {self.pool_topic}")
        self.get_logger().info(f"3D 2D trigger: {self.capture_2d_service} @ {self.capture_3d_2d_hz} Hz")

    def _activate_callback(self, _request, response):
        if self.run_mode:
            response.success = False
            response.message = "Already active"
            return response

        self.save_date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H-%M-%S")
        self.session_dir = self.save_dir_root / self.save_date / timestamp
        self.pool_dir = self.session_dir / "camera_pool"
        self.scan_dir = self.session_dir / "camera_3d_2d"
        self.robot_dir = self.session_dir / "robot_state"
        for folder in (self.pool_dir, self.scan_dir, self.robot_dir):
            folder.mkdir(parents=True, exist_ok=True)

        meta = {
            "created_at": datetime.now().isoformat(),
            "pool_topic": self.pool_topic,
            "scan_topic": self.scan_topic,
            "capture_2d_service": self.capture_2d_service,
            "capture_3d_2d_hz": self.capture_3d_2d_hz,
            "save_every_n_pool": self.save_every_n_pool,
        }
        (self.session_dir / "session_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self._pool_count = 0
        self._scan_count = 0
        self._tool_count = 0
        self._joint_count = 0
        self._twist_count = 0
        self.run_mode = True
        self._scan_timer.reset()

        self.get_logger().info(f"Collection activated: {self.session_dir}")
        response.success = True
        response.message = f"Started saving to {self.session_dir}"
        return response

    def _deactivate_callback(self, _request, response):
        self.run_mode = False
        self._scan_timer.cancel()
        summary = (
            f"pool={self._pool_count}, 3d_2d={self._scan_count}, "
            f"tool_pose={self._tool_count}, joint={self._joint_count}, twist={self._twist_count}"
        )
        self.get_logger().info(f"Collection deactivated. {summary}")
        response.success = True
        response.message = summary
        return response

    def _save_image(self, folder: Path, msg: Image, counter: int) -> None:
        ts = elapsed_microseconds(self.save_date or datetime.now().strftime("%Y-%m-%d"))
        image = imgmsg_to_bgr8(msg)
        path = folder / f"{ts}.{counter:06d}.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"cv2.imwrite failed: {path}")

    def _cb_pool_image(self, msg: Image) -> None:
        if not self.run_mode or self.pool_dir is None:
            return
        self._pool_count += 1
        if self._pool_count % self.save_every_n_pool != 0:
            return
        try:
            self._save_image(self.pool_dir, msg, self._pool_count)
            if self._pool_count % 30 == 0:
                self.get_logger().info(f"Saved pool images: {self._pool_count}")
        except Exception as exc:
            self.get_logger().error(f"Save pool image failed: {exc}")

    def _cb_scan_image(self, msg: Image) -> None:
        with self._scan_lock:
            self._scan_recv_count += 1
        if not self.run_mode or self.scan_dir is None:
            return
        # 只保存由 capture_2d 触发后到达的新帧（_scan_pending 标记）
        if not getattr(self, "_scan_pending", False):
            return
        self._scan_pending = False
        self._scan_count += 1
        try:
            self._save_image(self.scan_dir, msg, self._scan_count)
            if self._scan_count % 10 == 0:
                self.get_logger().info(f"Saved 3d_2d images: {self._scan_count}")
        except Exception as exc:
            self.get_logger().error(f"Save 3d_2d image failed: {exc}")

    def _trigger_capture_2d(self) -> None:
        if not self.run_mode:
            return
        if not self.capture_2d_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warning("/capture_2d not available", throttle_duration_sec=5.0)
            return
        self._scan_pending = True
        try:
            future = self.capture_2d_client.call_async(Trigger.Request())
            future.add_done_callback(self._on_capture_2d_done)
        except Exception as exc:
            self._scan_pending = False
            self.get_logger().warning(f"/capture_2d call failed: {exc}", throttle_duration_sec=3.0)

    def _on_capture_2d_done(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._scan_pending = False
            self.get_logger().warning(f"/capture_2d failed: {exc}", throttle_duration_sec=3.0)
            return
        if result is None or not result.success:
            self._scan_pending = False
            detail = result.message if result is not None else "no response"
            self.get_logger().warning(f"/capture_2d rejected: {detail}", throttle_duration_sec=3.0)

    def _append_csv(self, path: Path, header: list[str], row: list) -> None:
        file_exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def _cb_tool_pose(self, msg: JogPos) -> None:
        if not self.run_mode or self.robot_dir is None:
            return
        ts = elapsed_microseconds(self.save_date or datetime.now().strftime("%Y-%m-%d"))
        self._append_csv(
            self.robot_dir / "tool_pose.csv",
            ["timestamp", "x", "y", "z", "rx", "ry", "rz"],
            [ts, msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz],
        )
        self._tool_count += 1

    def _cb_joint_state(self, msg: JointPos) -> None:
        if not self.run_mode or self.robot_dir is None:
            return
        ts = elapsed_microseconds(self.save_date or datetime.now().strftime("%Y-%m-%d"))
        self._append_csv(
            self.robot_dir / "joint_state.csv",
            ["timestamp", "j1", "j2", "j3", "j4", "j5", "j6"],
            [ts, msg.j1, msg.j2, msg.j3, msg.j4, msg.j5, msg.j6],
        )
        self._joint_count += 1

    def _cb_twist(self, msg: Twist) -> None:
        if not self.run_mode or self.robot_dir is None:
            return
        ts = elapsed_microseconds(self.save_date or datetime.now().strftime("%Y-%m-%d"))
        self._append_csv(
            self.robot_dir / "control_speed.csv",
            ["timestamp", "vx", "vy", "vz", "wx", "wy", "wz"],
            [
                ts,
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z,
            ],
        )
        self._twist_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = TrainingDataCollectNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down.")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
