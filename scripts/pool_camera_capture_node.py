#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pool_camera_capture_node.py

ROS2 熔池相机采集节点：
1. 订阅 /pool_camera/image_raw
2. 支持保存单张图片
3. 支持连续保存图片
4. 支持通过参数修改图像话题、保存目录、图片格式、抽帧间隔

前提：需先启动熔池相机驱动 pool_camera_node，它才会持续发布图像：
  ros2 launch welding_pool_camera_driver pool_camera.launch.py

服务：
  /pool_camera/capture_once   保存当前最新一帧
  /pool_camera/start_capture   开始连续保存
  /pool_camera/stop_capture    停止连续保存
"""

import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


def imgmsg_to_bgr8(msg: Image) -> np.ndarray:
    """不依赖 cv_bridge，避免 venv NumPy 2.x 与 ROS cv_bridge 不兼容。"""
    encoding = msg.encoding.lower()
    if encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        ).copy()
    if encoding == "rgb8":
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if encoding in ("mono8", "8uc1"):
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if encoding == "bgra8":
        bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 4
        )
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if encoding in ("16uc1", "16sc1"):
        dtype = np.uint16 if encoding == "16uc1" else np.int16
        raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        normalized = cv2.normalize(raw.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        gray = normalized.astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class PoolCameraCaptureNode(Node):
    def __init__(self):
        super().__init__("pool_camera_capture_node")

        # 可通过 --ros-args -p xxx:=yyy 修改
        self.image_topic = self.declare_parameter(
            "image_topic", "/pool_camera/image_raw"
        ).value

        self.save_dir = self.declare_parameter(
            "save_dir", "/home/shugen/ros2_ws/pool_camera_images"
        ).value

        self.image_ext = self.declare_parameter(
            "image_ext", "jpg"
        ).value.lower().lstrip(".")

        self.save_every_n = int(self.declare_parameter(
            "save_every_n", 1
        ).value)

        if self.save_every_n < 1:
            self.save_every_n = 1

        self.image_wait_s = float(
            self.declare_parameter("image_wait_s", 8.0).value
        )

        self.latest_msg = None
        self.running = False
        self.current_session_dir = None
        self.recv_count = 0
        self.saved_count = 0

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # 订阅熔池相机图像（与 pool_camera_node 的 SensorDataQoS 一致）
        self.sub_image = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        # 三个服务：保存一张、开始连续保存、停止连续保存
        self.srv_capture_once = self.create_service(
            Trigger,
            "/pool_camera/capture_once",
            self.capture_once_callback
        )

        self.srv_start_capture = self.create_service(
            Trigger,
            "/pool_camera/start_capture",
            self.start_capture_callback
        )

        self.srv_stop_capture = self.create_service(
            Trigger,
            "/pool_camera/stop_capture",
            self.stop_capture_callback
        )

        self.get_logger().info("Pool camera capture node started.")
        self.get_logger().info(f"Subscribed image topic: {self.image_topic}")
        self.get_logger().info(f"Save root dir: {self.save_dir}")
        self.get_logger().info(f"Image extension: {self.image_ext}")
        self.get_logger().info(f"Save every N frames: {self.save_every_n}")
        self.get_logger().info(f"Image wait timeout: {self.image_wait_s}s")
        self.get_logger().info(
            "Prerequisite: ros2 launch welding_pool_camera_driver pool_camera.launch.py"
        )
        self.get_logger().info("Services:")
        self.get_logger().info("  /pool_camera/capture_once")
        self.get_logger().info("  /pool_camera/start_capture")
        self.get_logger().info("  /pool_camera/stop_capture")

    def make_new_session_dir(self) -> Path:
        """
        每次开始连续采集，或首次保存单张图片时，创建一个新目录：
        save_dir/YYYY-MM-DD/HH-MM-SS/
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        session_dir = Path(self.save_dir) / date_str / time_str
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def make_filename(self, msg: Image) -> str:
        """
        优先使用 ROS 消息时间戳；如果时间戳为空，则使用系统当前时间。
        """
        sec = int(msg.header.stamp.sec)
        nanosec = int(msg.header.stamp.nanosec)

        if sec != 0 or nanosec != 0:
            timestamp_us = sec * 1_000_000 + nanosec // 1_000
            return f"{timestamp_us}.{self.image_ext}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{timestamp}.{self.image_ext}"

    def ros_image_to_cv2(self, msg: Image):
        return imgmsg_to_bgr8(msg)

    def save_image_msg(self, msg: Image, target_dir: Path) -> Path:
        """
        将一帧 ROS Image 消息保存到 target_dir。
        """
        cv_img = self.ros_image_to_cv2(msg)
        filename = self.make_filename(msg)
        image_path = target_dir / filename

        ok = cv2.imwrite(str(image_path), cv_img)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {image_path}")

        self.saved_count += 1
        return image_path

    def image_callback(self, msg: Image):
        """
        每收到一帧图像都会进入这里。
        如果 running=False，只缓存最新帧，不保存。
        如果 running=True，则按 save_every_n 抽帧保存。
        """
        self.latest_msg = msg
        self.recv_count += 1

        if not self.running:
            return

        if self.recv_count % self.save_every_n != 0:
            return

        try:
            image_path = self.save_image_msg(msg, self.current_session_dir)
            if self.saved_count % 30 == 0:
                self.get_logger().info(
                    f"Saved {self.saved_count} images. Latest: {image_path}"
                )
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")

    def wait_for_image(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest_msg is not None:
                return True
            time.sleep(0.05)
        return False

    def capture_once_callback(self, request, response):
        """
        保存当前缓存的最新一帧；若尚未收到图像则等待一段时间。
        """
        if self.latest_msg is None and not self.wait_for_image(self.image_wait_s):
            response.success = False
            response.message = (
                f"No image received from {self.image_topic} within {self.image_wait_s}s. "
                "pool_camera_node is probably not running. Start driver first:\n"
                "  ros2 launch welding_pool_camera_driver pool_camera.launch.py\n"
                f"Then verify: ros2 topic hz {self.image_topic}"
            )
            return response

        if self.current_session_dir is None:
            self.current_session_dir = self.make_new_session_dir()

        try:
            image_path = self.save_image_msg(self.latest_msg, self.current_session_dir)
            response.success = True
            response.message = f"Saved one image: {image_path}"
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Failed to save image: {e}"
            self.get_logger().error(response.message)

        return response

    def start_capture_callback(self, request, response):
        """
        开始连续保存。
        """
        if self.running:
            response.success = False
            response.message = f"Already capturing. Current dir: {self.current_session_dir}"
            return response

        self.current_session_dir = self.make_new_session_dir()
        self.running = True
        self.saved_count = 0

        if self.latest_msg is None and not self.wait_for_image(self.image_wait_s):
            self.running = False
            response.success = False
            response.message = (
                f"Started continuous capture aborted: no image from {self.image_topic}. "
                "Start pool_camera_node first:\n"
                "  ros2 launch welding_pool_camera_driver pool_camera.launch.py"
            )
            return response

        response.success = True
        response.message = f"Started continuous capture. Saving to: {self.current_session_dir}"
        self.get_logger().info(response.message)
        return response

    def stop_capture_callback(self, request, response):
        """
        停止连续保存。
        """
        if not self.running:
            response.success = False
            response.message = "Not capturing."
            return response

        self.running = False
        response.success = True
        response.message = (
            f"Stopped capture. Saved {self.saved_count} images. "
            f"Last session dir: {self.current_session_dir}"
        )
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PoolCameraCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
