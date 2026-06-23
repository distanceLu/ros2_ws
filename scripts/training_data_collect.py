#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_data_collect.py

面向小模型训练的同步数据采集：
  - 熔池相机：/pool_camera/image_raw（连续保存）
  - 3D 相机 2D 图：独立线程同步调用 /capture_2d_image（无投影），直接保存服务返回的图像
  - 纸面相机：工控机 USB 相机（连续保存，文件名格式与上面两路一致）
  - 机械臂：/tool_pos
  - 遥操动作：/spacenav/twist

与旧版 data_collect.py 的区别：
  - 不再依赖 /image_topic0（MindVision 2D）
  - 使用 3D 相机纯 2D + 熔池相机 + 纸面相机组成多视角图像集
  - 使用 welding_runtime 的 common_interface/TcpPos 记录 TCP 位姿
  - 记录遥操 twist 作为 action

使用方式：
  终端1：启动相机与机械臂（不要 run.launch.py）
    python3 scripts/camera_capture_node.py --ros-args \\
      -p auto_start_camera_keys:=3d,pool \\
      -p keep_launched_drivers_on_exit:=true
    ros2 run welding_runtime robot_driver_bridge_node --ros-args -p robot_type:=duco

  终端2：启动本采集节点
    source ~/Documents/auto_welding/install/setup.bash
    source /home/shugen/yanjie/ros2_ws/install/setup.bash
    python3 scripts/training_data_collect.py

  终端3：开始/停止采集
    ros2 service call /training_data_collect_activate std_srvs/srv/Trigger {}
    ros2 service call /training_data_collect_deactivate std_srvs/srv/Trigger {}

数据目录：
  /home/shugen/yanjie/ros2_ws/data_collect/YYYY-MM-DD/HH-MM-SS/
    camera_pool/          熔池图
    camera_3d_2d/         3D 相机无激光 2D 图
    camera_paper_aruco/   纸面工控机 USB 相机图
    paper_state/          paper_aruco_pose.csv
    robot_state/          tool_pose.csv, control_speed.csv
    session_meta.json     采集参数摘要
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_aruco_localize import append_collect_pose, localize_paper, write_collect_header

import cv2
import numpy as np
import rclpy
from common_interface.msg import TcpPos
from common_interface.srv import Scan3D
from geometry_msgs.msg import Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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
            self.declare_parameter("save_dir_root", "/home/shugen/yanjie/ros2_ws/data_collect").value
        )
        self.pool_topic = self.declare_parameter("pool_camera_topic", "/pool_camera/image_raw").value
        self.scan_topic = self.declare_parameter("scan_image_topic", "/scan/image_raw").value
        self.capture_2d_service = self.declare_parameter("capture_2d_service", "/capture_2d").value
        self.capture_2d_image_service = self.declare_parameter(
            "capture_2d_image_service", "/capture_2d_image"
        ).value
        self.capture_3d_2d_hz = float(self.declare_parameter("capture_3d_2d_hz", 5.0).value)
        self.capture_service_timeout_sec = float(
            self.declare_parameter("capture_service_timeout_sec", 5.0).value
        )
        self.capture_timeout_sec = float(self.declare_parameter("capture_timeout_sec", 10.0).value)
        self.scan_image_wait_sec = float(self.declare_parameter("scan_image_wait_sec", 2.0).value)
        self.restart_3d_on_timeout = bool(self.declare_parameter("restart_3d_on_timeout", True).value)
        self.save_every_n_pool = max(1, int(self.declare_parameter("save_every_n_pool", 1).value))
        self.enable_paper_camera = bool(self.declare_parameter("enable_paper_camera", True).value)
        self.paper_camera_device = self.declare_parameter("paper_camera_device", "/dev/video0").value
        self.paper_camera_hz = float(self.declare_parameter("paper_camera_hz", 15.0).value)
        self.paper_camera_width = int(self.declare_parameter("paper_camera_width", 3840).value)
        self.paper_camera_height = int(self.declare_parameter("paper_camera_height", 2160).value)
        self.paper_camera_warmup_frames = max(
            0, int(self.declare_parameter("paper_camera_warmup_frames", 10).value)
        )
        self.paper_camera_rotate_180 = bool(
            self.declare_parameter("paper_camera_rotate_180", True).value
        )
        self.paper_camera_localize = bool(
            self.declare_parameter("paper_camera_localize", False).value
        )
        self.paper_jpeg_quality = max(
            1, min(100, int(self.declare_parameter("paper_jpeg_quality", 95).value))
        )

        self.run_mode = False
        self.save_date: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.pool_dir: Optional[Path] = None
        self.scan_dir: Optional[Path] = None
        self.paper_dir: Optional[Path] = None
        self.paper_state_dir: Optional[Path] = None
        self.paper_pose_csv: Optional[Path] = None
        self.robot_dir: Optional[Path] = None

        self._pool_count = 0
        self._scan_count = 0
        self._paper_count = 0
        self._tool_count = 0
        self._twist_count = 0
        self._scan_recv_count = 0
        self._scan_fail_count = 0
        self._scan_fallback_count = 0
        self._scan_restart_count = 0
        self._last_scan_restart_monotonic = 0.0
        self._scan_lock = threading.Lock()
        self._scan_condition = threading.Condition(self._scan_lock)
        self._scan_capturing = False
        self._scan_capture_since = 0
        self._scan_saved_this_capture = False
        self._scan_running = False
        self._scan_thread: Optional[threading.Thread] = None
        self._paper_lock = threading.Lock()
        self._paper_running = False
        self._paper_thread: Optional[threading.Thread] = None
        self._paper_capture = None

        self.capture_2d_client = self.create_client(
            Trigger, self.capture_2d_service, callback_group=self.callback_group
        )
        self.capture_2d_image_client = self.create_client(
            Scan3D, self.capture_2d_image_service, callback_group=self.callback_group
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
            TcpPos,
            "/tool_pos",
            self._cb_tool_pose,
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

        self.get_logger().info("Training data collect node started.")
        self.get_logger().info(f"Save root: {self.save_dir_root}")
        self.get_logger().info(f"Pool topic: {self.pool_topic}")
        self.get_logger().info(
            f"3D 2D trigger: {self.capture_2d_image_service} @ {self.capture_3d_2d_hz} Hz "
            f"(no projector; legacy topic wait={self.scan_image_wait_sec}s)"
        )
        if self.enable_paper_camera:
            self.get_logger().info(
                f"Paper camera: {self.paper_camera_device} @ {self.paper_camera_hz} Hz "
                f"({'localize' if self.paper_camera_localize else 'raw-only'})"
            )
        else:
            self.get_logger().info("Paper camera: disabled")

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
        self.paper_dir = self.session_dir / "camera_paper_aruco"
        self.paper_state_dir = self.session_dir / "paper_state"
        self.paper_pose_csv = self.paper_state_dir / "paper_aruco_pose.csv"
        self.robot_dir = self.session_dir / "robot_state"
        folders = [self.pool_dir, self.scan_dir, self.robot_dir]
        if self.enable_paper_camera:
            folders.extend([self.paper_dir, self.paper_state_dir])
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
        if self.enable_paper_camera and self.paper_pose_csv is not None:
            write_collect_header(self.paper_pose_csv)

        meta = {
            "created_at": datetime.now().isoformat(),
            "pool_topic": self.pool_topic,
            "scan_topic": self.scan_topic,
            "capture_2d_service": self.capture_2d_service,
            "capture_2d_image_service": self.capture_2d_image_service,
            "capture_3d_2d_hz": self.capture_3d_2d_hz,
            "capture_service_timeout_sec": self.capture_service_timeout_sec,
            "capture_timeout_sec": self.capture_timeout_sec,
            "scan_image_wait_sec": self.scan_image_wait_sec,
            "restart_3d_on_timeout": self.restart_3d_on_timeout,
            "save_every_n_pool": self.save_every_n_pool,
            "enable_paper_camera": self.enable_paper_camera,
            "paper_camera_device": self.paper_camera_device,
            "paper_camera_hz": self.paper_camera_hz,
            "paper_camera_width": self.paper_camera_width,
            "paper_camera_height": self.paper_camera_height,
            "paper_camera_rotate_180": self.paper_camera_rotate_180,
            "paper_camera_localize": self.paper_camera_localize,
        }
        (self.session_dir / "session_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self._pool_count = 0
        self._scan_count = 0
        self._scan_fail_count = 0
        self._scan_fallback_count = 0
        self._scan_restart_count = 0
        self._paper_count = 0
        self._tool_count = 0
        self._twist_count = 0
        self.run_mode = True
        self._start_scan_capture_thread()
        if self.enable_paper_camera:
            self._start_paper_camera_thread()

        self.get_logger().info(f"Collection activated: {self.session_dir}")
        response.success = True
        response.message = f"Started saving to {self.session_dir}"
        return response

    def _deactivate_callback(self, _request, response):
        self.run_mode = False
        self._stop_scan_capture_thread()
        self._stop_paper_camera_thread()
        summary = (
            f"pool={self._pool_count}, 3d_2d={self._scan_count}, "
            f"3d_2d_fail={self._scan_fail_count}, "
            f"3d_2d_fallback={self._scan_fallback_count}, "
            f"3d_2d_restart={self._scan_restart_count}, "
            f"paper_aruco={self._paper_count}, "
            f"tool_pose={self._tool_count}, twist={self._twist_count}"
        )
        self.get_logger().info(f"Collection deactivated. {summary}")
        response.success = True
        response.message = summary
        return response

    def _save_image(self, folder: Path, msg: Image, counter: int) -> str:
        image = imgmsg_to_bgr8(msg)
        return self._save_bgr_image(folder, image, counter)

    def _save_scan_msg(self, msg: Image) -> bool:
        if self.scan_dir is None:
            return False
        with self._scan_condition:
            self._scan_count += 1
            counter = self._scan_count
            folder = self.scan_dir
        try:
            self._save_image(folder, msg, counter)
            if counter % 10 == 0:
                self.get_logger().info(f"Saved 3d_2d images: {counter}")
            return True
        except Exception as exc:
            self.get_logger().error(f"Save 3d_2d image failed: {exc}")
            return False

    def _save_bgr_image(self, folder: Path, image: np.ndarray, counter: int, jpeg_quality: int = 95) -> str:
        ts = elapsed_microseconds(self.save_date or datetime.now().strftime("%Y-%m-%d"))
        image_name = f"{ts}.{counter:06d}.jpg"
        path = folder / image_name
        if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]):
            raise RuntimeError(f"cv2.imwrite failed: {path}")
        return image_name

    def _start_scan_capture_thread(self) -> None:
        if self._scan_thread is not None and self._scan_thread.is_alive():
            return
        self._scan_running = True
        self._scan_thread = threading.Thread(target=self._scan_collect_loop, daemon=True)
        self._scan_thread.start()

    def _stop_scan_capture_thread(self) -> None:
        self._scan_running = False
        thread = self._scan_thread
        if thread is not None:
            thread.join(timeout=max(self.capture_timeout_sec, self.scan_image_wait_sec) + 2.0)
        self._scan_thread = None
        with self._scan_condition:
            self._scan_capturing = False
            self._scan_condition.notify_all()

    def _scan_collect_loop(self) -> None:
        period = 1.0 / max(0.5, self.capture_3d_2d_hz)
        self.get_logger().info(
            f"3D 2D collect started: service={self.capture_2d_service}, target_hz={self.capture_3d_2d_hz}"
        )
        try:
            while self._scan_running and self.run_mode:
                loop_start = time.monotonic()
                self._capture_scan_once()
                sleep_s = period - (time.monotonic() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self.get_logger().info(
                f"3D 2D collect stopped: saved={self._scan_count}, failed={self._scan_fail_count}"
            )

    def _call_capture_2d_sync(self) -> tuple[bool, str]:
        if not self.capture_2d_client.wait_for_service(timeout_sec=self.capture_service_timeout_sec):
            return False, f"{self.capture_2d_service} not available"

        future = self.capture_2d_client.call_async(Trigger.Request())
        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        if not done.wait(timeout=self.capture_timeout_sec):
            return False, f"{self.capture_2d_service} call timeout after {self.capture_timeout_sec}s"

        try:
            result = future.result()
        except Exception as exc:
            return False, f"{self.capture_2d_service} call failed: {exc}"

        if result is None:
            return False, f"{self.capture_2d_service} returned no response"
        if not result.success:
            detail = result.message or "unknown error"
            return False, f"{self.capture_2d_service} rejected: {detail}"
        return True, result.message or "ok"

    def _call_capture_2d_image_sync(self) -> tuple[bool, str, Optional[Image]]:
        if not self.capture_2d_image_client.wait_for_service(timeout_sec=0.1):
            return False, f"{self.capture_2d_image_service} not available", None

        future = self.capture_2d_image_client.call_async(Scan3D.Request())
        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        if not done.wait(timeout=self.capture_timeout_sec):
            return False, f"{self.capture_2d_image_service} call timeout", None

        try:
            result = future.result()
        except Exception as exc:
            return False, f"{self.capture_2d_image_service} call failed: {exc}", None

        if result is None:
            return False, f"{self.capture_2d_image_service} returned no response", None
        if not result.success:
            detail = result.message or "unknown error"
            return False, f"{self.capture_2d_image_service} rejected: {detail}", None
        if result.image.width == 0 or result.image.height == 0 or not result.image.data:
            return False, f"{self.capture_2d_image_service} returned empty image", None
        return True, result.message or "ok", result.image

    def _restart_3d_driver(self, reason: str) -> None:
        if not self.restart_3d_on_timeout:
            return
        now = time.monotonic()
        if now - self._last_scan_restart_monotonic < 8.0:
            return
        self._last_scan_restart_monotonic = now
        self._scan_restart_count += 1
        self.get_logger().warning(f"Restarting 3D camera driver after {reason}")

        try:
            subprocess.run(
                ["pkill", "-f", "/scan_camera_node"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
            subprocess.run(
                ["pkill", "-f", "ros2 launch welding_scan3d_camera_driver scan3d_camera.launch.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
            time.sleep(1.0)
            command = (
                "source /opt/ros/jazzy/setup.bash && "
                "source /home/shugen/Documents/auto_welding/install/local_setup.bash && "
                "source /home/shugen/yanjie/ros2_ws/install/setup.bash && "
                "exec ros2 launch welding_scan3d_camera_driver scan3d_camera.launch.py"
            )
            subprocess.Popen(
                ["bash", "-lc", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to restart 3D camera driver: {exc}")

    def _capture_scan_once(self) -> None:
        if not self.run_mode or self.scan_dir is None:
            return

        image_ok, image_detail, image_msg = self._call_capture_2d_image_sync()
        if image_ok and image_msg is not None and self._save_scan_msg(image_msg):
            return
        if not image_detail.endswith("not available"):
            self._scan_fail_count += 1
            self.get_logger().warning(image_detail, throttle_duration_sec=3.0)
            if "timeout" in image_detail:
                self._restart_3d_driver(image_detail)
            return

        with self._scan_condition:
            self._scan_capturing = True
            self._scan_capture_since = self._scan_recv_count
            self._scan_saved_this_capture = False

        ok, detail = self._call_capture_2d_sync()
        if not ok:
            with self._scan_condition:
                self._scan_capturing = False
                self._scan_condition.notify_all()
            self._scan_fail_count += 1
            self.get_logger().warning(detail, throttle_duration_sec=3.0)
            return

        with self._scan_condition:
            deadline = time.monotonic() + self.scan_image_wait_sec
            while not self._scan_saved_this_capture and time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._scan_condition.wait(timeout=remaining)
            saved = self._scan_saved_this_capture
            self._scan_capturing = False
            self._scan_condition.notify_all()

        if not saved:
            self._scan_fail_count += 1
            self.get_logger().warning(
                f"{self.capture_2d_service} succeeded but no new image on {self.scan_topic} "
                f"within {self.scan_image_wait_sec}s",
                throttle_duration_sec=3.0,
            )

    def _start_paper_camera_thread(self) -> None:
        if self._paper_thread is not None and self._paper_thread.is_alive():
            return
        self._paper_running = True
        self._paper_thread = threading.Thread(target=self._paper_collect_loop, daemon=True)
        self._paper_thread.start()

    def _stop_paper_camera_thread(self) -> None:
        self._paper_running = False
        thread = self._paper_thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._paper_thread = None
        capture = self._paper_capture
        self._paper_capture = None
        if capture is not None:
            capture.release()

    def _paper_collect_loop(self) -> None:
        cap = cv2.VideoCapture(self.paper_camera_device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.get_logger().error(f"Paper camera unavailable: {self.paper_camera_device}")
            self._paper_running = False
            return

        self._paper_capture = cap
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.paper_camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.paper_camera_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(self.paper_camera_warmup_frames):
            cap.grab()

        period = 1.0 / max(0.1, self.paper_camera_hz)
        self.get_logger().info(
            f"Paper camera collect started: device={self.paper_camera_device}, hz={self.paper_camera_hz}"
        )
        try:
            while self._paper_running and self.run_mode:
                loop_start = time.monotonic()
                ok = cap.grab()
                frame = None
                if ok:
                    ok, frame = cap.retrieve()
                if not ok or frame is None or frame.size == 0:
                    self.get_logger().warning(
                        "Paper camera read failed",
                        throttle_duration_sec=3.0,
                    )
                    time.sleep(period)
                    continue
                if self.paper_camera_rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                with self._paper_lock:
                    if not self.run_mode or self.paper_dir is None or self.paper_pose_csv is None:
                        continue
                    self._paper_count += 1
                    counter = self._paper_count
                    paper_dir = self.paper_dir
                    paper_pose_csv = self.paper_pose_csv

                try:
                    image_name = self._save_bgr_image(
                        paper_dir, frame, counter, jpeg_quality=self.paper_jpeg_quality
                    )
                    ts = int(image_name.split(".", 1)[0])
                    if self.paper_camera_localize:
                        _, pose = localize_paper(frame)
                        append_collect_pose(paper_pose_csv, ts, image_name, pose)
                    else:
                        self._append_paper_raw_row(paper_pose_csv, ts, image_name)
                    if counter % 10 == 0:
                        self.get_logger().info(f"Saved paper_aruco images: {counter}")
                except Exception as exc:
                    self.get_logger().error(f"Save paper_aruco image failed: {exc}")

                sleep_s = period - (time.monotonic() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            if self._paper_capture is cap:
                cap.release()
                self._paper_capture = None
            self.get_logger().info(f"Paper camera collect stopped: saved={self._paper_count}")

    def _append_paper_raw_row(self, path: Path, timestamp: int, image_name: str) -> None:
        # Keep the CSV as a lightweight time index; ArUco fields stay empty in raw-only mode.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp},{image_name},,,,,,\n")

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
        save_target: Optional[tuple[Path, int]] = None
        with self._scan_condition:
            self._scan_recv_count += 1
            if (
                self.run_mode
                and self.scan_dir is not None
                and self._scan_capturing
                and not self._scan_saved_this_capture
                and self._scan_recv_count > self._scan_capture_since
            ):
                self._scan_saved_this_capture = True
                self._scan_count += 1
                save_target = (self.scan_dir, self._scan_count)
                self._scan_condition.notify_all()

        if save_target is None:
            return
        folder, counter = save_target
        try:
            self._save_image(folder, msg, counter)
            if counter % 10 == 0:
                self.get_logger().info(f"Saved 3d_2d images: {counter}")
        except Exception as exc:
            self.get_logger().error(f"Save 3d_2d image failed: {exc}")

    def _append_csv(self, path: Path, header: list[str], row: list) -> None:
        file_exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def _cb_tool_pose(self, msg: TcpPos) -> None:
        if not self.run_mode or self.robot_dir is None:
            return
        ts = elapsed_microseconds(self.save_date or datetime.now().strftime("%Y-%m-%d"))
        self._append_csv(
            self.robot_dir / "tool_pose.csv",
            ["timestamp", "x", "y", "z", "rx", "ry", "rz"],
            [ts, msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz],
        )
        self._tool_count += 1


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
