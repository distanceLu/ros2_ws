#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_capture_node.py

只依赖本脚本完成三种相机的统一拍照：
  1. 2D 相机：camera_node，默认话题 /image_topic0
  2. 3D 相机：scan_camera_node，默认话题 /scan/image_raw，优先调用 /scan_3d
  3. 3D 相机纯 2D：scan_camera_node，调用 /capture_2d（无激光，等同 RVCManager）
  4. 熔池相机：pool_camera_node，默认话题 /pool_camera/image_raw

使用方式：
  cd ~/ros2_ws
  source install/setup.bash
  python3 scripts/camera_capture_node.py

然后按需调用：
  ros2 service call /camera_capture/capture_once_2d std_srvs/srv/Trigger {}
  ros2 service call /camera_capture/capture_once_3d std_srvs/srv/Trigger {}
  ros2 service call /camera_capture/capture_once_3d_2d std_srvs/srv/Trigger {}
  ros2 service call /camera_capture/capture_once_pool std_srvs/srv/Trigger {}

本脚本只负责启动/等待相机驱动和保存图片，不修改任何驱动源码、launch 或配置文件。
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from common_interface.srv import Scan3D
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent


def reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        depth=depth,
    )


def imgmsg_to_bgr8(msg: Image) -> np.ndarray:
    """不依赖 cv_bridge，避免 venv NumPy 2.x 与 ROS cv_bridge 不兼容。"""
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
    if encoding in ("16uc1", "16sc1"):
        dtype = np.uint16 if encoding == "16uc1" else np.int16
        raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        normalized = cv2.normalize(raw.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        gray = normalized.astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


@dataclass
class CameraSpec:
    key: str
    label: str
    topic: str
    node_name: str
    launch_package: str
    launch_file: str
    qos: object
    save_subdir: str
    scan_service: str = ""
    start_service: str = ""
    stop_service: str = ""
    capture_2d_service: str = ""


@dataclass
class CameraState:
    spec: CameraSpec
    latest_msg: Optional[Image] = None
    running: bool = False
    current_session_dir: Optional[Path] = None
    recv_count: int = 0
    saved_count: int = 0
    launch_process: Optional[subprocess.Popen] = None


class UnifiedCameraCaptureNode(Node):
    def __init__(self):
        super().__init__("unified_camera_capture_node")
        self.callback_group = ReentrantCallbackGroup()

        self.save_dir_root = Path(
            self.declare_parameter("save_dir", str(WORKSPACE_ROOT / "camera_images")).value
        )
        self.save_dir_root.mkdir(parents=True, exist_ok=True)
        self.image_ext = self.declare_parameter("image_ext", "jpg").value.lower().lstrip(".")
        self.save_every_n = max(1, int(self.declare_parameter("save_every_n", 1).value))
        self.driver_wait_s = float(self.declare_parameter("driver_wait_s", 20.0).value)
        self.image_wait_s = float(self.declare_parameter("image_wait_s", 10.0).value)
        self.scan3d_timeout_s = float(self.declare_parameter("scan3d_timeout_s", 20.0).value)
        self.auto_start_drivers_on_startup = bool(
            self.declare_parameter("auto_start_drivers_on_startup", True).value
        )
        auto_start_keys = self.declare_parameter(
            "auto_start_camera_keys", "2d,3d,pool"
        ).value
        self.auto_start_camera_keys = {
            item.strip() for item in auto_start_keys.split(",") if item.strip()
        }
        self.keep_launched_drivers_on_exit = bool(
            self.declare_parameter("keep_launched_drivers_on_exit", False).value
        )

        self.camera_2d_topic = self.declare_parameter("camera_2d_topic", "/image_topic0").value
        self.camera_2d_use_standalone_node = bool(
            self.declare_parameter("camera_2d_use_standalone_node", True).value
        )
        default_2d_cfg = (
            WORKSPACE_ROOT
            / "install"
            / "camera_sdk"
            / "share"
            / "camera_sdk"
            / "config"
            / "config.yaml"
        )
        self.camera_2d_cfg_file = self.declare_parameter(
            "camera_2d_cfg_file",
            str(default_2d_cfg),
        ).value
        self.camera_2d_fre = self.declare_parameter("camera_2d_fre", 20).value
        self.camera_3d_topic = self.declare_parameter("camera_3d_topic", "/scan/image_raw").value
        self.pool_camera_topic = self.declare_parameter(
            "pool_camera_topic", "/pool_camera/image_raw"
        ).value
        self.camera_3d_capture_2d_service = self.declare_parameter(
            "camera_3d_capture_2d_service", "/capture_2d"
        ).value
        self.camera_3d_stop_service = self.declare_parameter(
            "camera_3d_stop_service", "/stop_capture"
        ).value

        self.cameras = {
            "2d": CameraState(
                CameraSpec(
                    key="2d",
                    label="2D相机",
                    topic=self.camera_2d_topic,
                    node_name="/camera_node",
                    launch_package="camera_sdk",
                    launch_file="run.launch.py",
                    qos=reliable_qos(),
                    save_subdir="2d",
                )
            ),
            "3d": CameraState(
                CameraSpec(
                    key="3d",
                    label="3D相机",
                    topic=self.camera_3d_topic,
                    node_name="/scan_camera_node",
                    launch_package="welding_scan3d_camera_driver",
                    launch_file="scan3d_camera.launch.py",
                    qos=qos_profile_sensor_data,
                    save_subdir="3d",
                    scan_service="/scan_3d",
                    start_service="/start_capture",
                    stop_service=self.camera_3d_stop_service,
                    capture_2d_service=self.camera_3d_capture_2d_service,
                )
            ),
            "3d_2d": CameraState(
                CameraSpec(
                    key="3d_2d",
                    label="3D相机纯2D",
                    topic=self.camera_3d_topic,
                    node_name="/scan_camera_node",
                    launch_package="welding_scan3d_camera_driver",
                    launch_file="scan3d_camera.launch.py",
                    qos=qos_profile_sensor_data,
                    save_subdir="3d_2d",
                    stop_service=self.camera_3d_stop_service,
                    capture_2d_service=self.camera_3d_capture_2d_service,
                )
            ),
            "pool": CameraState(
                CameraSpec(
                    key="pool",
                    label="熔池相机",
                    topic=self.pool_camera_topic,
                    node_name="/pool_camera_node",
                    launch_package="welding_pool_camera_driver",
                    launch_file="pool_camera.launch.py",
                    qos=qos_profile_sensor_data,
                    save_subdir="pool",
                )
            ),
        }

        self.scan3d_client = self.create_client(
            Scan3D, self.cameras["3d"].spec.scan_service, callback_group=self.callback_group
        )
        self.scan3d_start_client = self.create_client(
            Trigger, self.cameras["3d"].spec.start_service, callback_group=self.callback_group
        )
        self.capture_2d_client = self.create_client(
            Trigger, self.camera_3d_capture_2d_service, callback_group=self.callback_group
        )
        self.scan3d_stop_client = self.create_client(
            Trigger, self.camera_3d_stop_service, callback_group=self.callback_group
        )

        subscribed_topics: set[str] = set()
        for state in self.cameras.values():
            if state.spec.key == "3d_2d":
                continue
            if state.spec.topic in subscribed_topics:
                continue
            subscribed_topics.add(state.spec.topic)
            self.create_subscription(
                Image,
                state.spec.topic,
                lambda msg, camera_key=state.spec.key: self._image_callback(camera_key, msg),
                state.spec.qos,
                callback_group=self.callback_group,
            )

        self._startup_thread_started = False
        self._create_capture_services()
        self._log_startup_info()
        self.startup_timer = self.create_timer(
            0.5,
            self._start_configured_drivers_on_startup_async,
            callback_group=self.callback_group,
        )

    def _create_capture_services(self) -> None:
        service_specs = [
            ("/camera_capture/capture_once_2d", lambda req, res: self._capture_once("2d", res)),
            ("/camera_capture/capture_once_3d", lambda req, res: self._capture_once("3d", res)),
            (
                "/camera_capture/capture_once_3d_2d",
                lambda req, res: self._capture_once("3d_2d", res),
            ),
            ("/camera_capture/capture_once_pool", lambda req, res: self._capture_once("pool", res)),
            ("/camera_capture/start_capture_2d", lambda req, res: self._start_capture("2d", res)),
            ("/camera_capture/start_capture_3d", lambda req, res: self._start_capture("3d", res)),
            ("/camera_capture/start_capture_pool", lambda req, res: self._start_capture("pool", res)),
            ("/camera_capture/stop_capture_2d", lambda req, res: self._stop_capture("2d", res)),
            ("/camera_capture/stop_capture_3d", lambda req, res: self._stop_capture("3d", res)),
            ("/camera_capture/stop_capture_pool", lambda req, res: self._stop_capture("pool", res)),
        ]
        for service_name, callback in service_specs:
            self.create_service(Trigger, service_name, callback, callback_group=self.callback_group)

    def _log_startup_info(self) -> None:
        self.get_logger().info("Unified camera capture node started.")
        self.get_logger().info(f"Save root dir: {self.save_dir_root}")
        for state in self.cameras.values():
            spec = state.spec
            self.get_logger().info(
                f"{spec.label}: topic={spec.topic}, node={spec.node_name}, "
                f"launch=ros2 launch {spec.launch_package} {spec.launch_file}"
            )
        self.get_logger().info(
            f"Auto start drivers on startup: {self.auto_start_drivers_on_startup}, "
            f"keys={sorted(self.auto_start_camera_keys)}"
        )
        self.get_logger().info("Services:")
        for key in ("2d", "3d", "3d_2d", "pool"):
            self.get_logger().info(f"  /camera_capture/capture_once_{key}")
            self.get_logger().info(f"  /camera_capture/start_capture_{key}")
            self.get_logger().info(f"  /camera_capture/stop_capture_{key}")

    def _start_configured_drivers_on_startup_async(self) -> None:
        if self._startup_thread_started:
            return
        self._startup_thread_started = True
        self.startup_timer.cancel()
        if not self.auto_start_drivers_on_startup:
            return
        thread = threading.Thread(
            target=self._start_configured_drivers_on_startup,
            daemon=True,
        )
        thread.start()

    def _resolve_auto_start_driver_keys(self) -> set[str]:
        keys: set[str] = set()
        for key in self.auto_start_camera_keys:
            if key == "3d_2d":
                keys.add("3d")
            elif key in self.cameras:
                keys.add(key)
        return keys

    def _start_configured_drivers_on_startup(self) -> None:
        for key in sorted(self._resolve_auto_start_driver_keys()):
            state = self.cameras[key]
            ok, message = self._prepare_camera(state)
            if ok:
                self.get_logger().info(f"[{state.spec.label}] startup ready: {message}")
            else:
                self.get_logger().warn(f"[{state.spec.label}] startup failed: {message}")

    def _image_callback(self, camera_key: str, msg: Image) -> None:
        state = self.cameras[camera_key]
        state.latest_msg = msg
        state.recv_count += 1
        if camera_key == "3d":
            shared = self.cameras.get("3d_2d")
            if shared is not None:
                shared.latest_msg = msg
                shared.recv_count = state.recv_count

        if not state.running:
            return
        if state.recv_count % self.save_every_n != 0:
            return

        try:
            path = self._save_image_msg(state, msg)
            if state.saved_count % 30 == 0:
                self.get_logger().info(f"[{state.spec.label}] saved {state.saved_count}: {path}")
        except Exception as exc:
            self.get_logger().error(f"[{state.spec.label}] save failed: {exc}")

    def _ros_setup(self) -> str:
        parts: list[str] = []
        ros_setup = Path("/opt/ros/jazzy/setup.bash")
        auto_welding = Path("/home/shugen/Documents/auto_welding/install/local_setup.bash")
        ws_setup = WORKSPACE_ROOT / "install" / "setup.bash"
        if ros_setup.is_file():
            parts.append(f"source {ros_setup}")
        if auto_welding.is_file():
            parts.append(f"source {auto_welding}")
        if ws_setup.is_file():
            parts.append(f"source {ws_setup}")
        return " && ".join(parts)

    def _ros2_command(self, args: list[str], timeout_s: float = 3.0) -> subprocess.CompletedProcess:
        setup = self._ros_setup()
        command = " ".join(args)
        shell_command = f"{setup} && {command}" if setup else command
        return subprocess.run(
            ["bash", "-lc", shell_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )

    def _node_is_running(self, node_name: str) -> bool:
        try:
            result = self._ros2_command(["ros2", "node", "list"], timeout_s=3.0)
        except Exception:
            return False
        nodes = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        return node_name in nodes

    def _driver_start_command(self, state: CameraState) -> str:
        spec = state.spec
        setup = self._ros_setup()
        if spec.key == "2d" and self.camera_2d_use_standalone_node:
            command = (
                "ros2 run camera_sdk camera_node --ros-args "
                f"-p cfg_file:={self.camera_2d_cfg_file} "
                f"-p fre:={self.camera_2d_fre}"
            )
        else:
            command = f"ros2 launch {spec.launch_package} {spec.launch_file}"
        return f"{setup} && {command}" if setup else command

    def _start_driver_if_needed(self, state: CameraState) -> tuple[bool, str]:
        spec = state.spec
        if self._node_is_running(spec.node_name):
            return True, f"{spec.node_name} already running"

        if state.launch_process is not None and state.launch_process.poll() is None:
            return self._wait_for_node(spec.node_name), f"waiting existing launch process for {spec.node_name}"

        command = self._driver_start_command(state)
        self.get_logger().info(f"[{spec.label}] starting driver: {command}")
        state.launch_process = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if self._wait_for_node(spec.node_name):
            return True, f"started {spec.node_name}"

        if state.launch_process.poll() is not None:
            output = ""
            if state.launch_process.stdout is not None:
                try:
                    output = state.launch_process.stdout.read()[-800:]
                except Exception:
                    output = ""
            return False, f"launch exited before {spec.node_name} appeared. output: {output}"

        return False, f"timeout waiting for {spec.node_name} after launch"

    def _wait_for_node(self, node_name: str) -> bool:
        deadline = time.monotonic() + self.driver_wait_s
        while time.monotonic() < deadline:
            if self._node_is_running(node_name):
                return True
            time.sleep(0.5)
        return False

    def _wait_for_image(self, state: CameraState, since_recv_count: int = -1) -> bool:
        deadline = time.monotonic() + self.image_wait_s
        while time.monotonic() < deadline:
            if state.latest_msg is not None and state.recv_count > since_recv_count:
                return True
            time.sleep(0.05)
        return False

    def _stop_3d_capture_stream_if_available(self) -> tuple[bool, str]:
        client = self.scan3d_stop_client
        if not client.wait_for_service(timeout_sec=3.0):
            return False, "/stop_capture not available"
        try:
            result = client.call(Trigger.Request())
        except Exception as exc:
            return False, f"/stop_capture call failed: {exc}"
        if result is None:
            return False, "/stop_capture returned no response"
        if result.success:
            return True, result.message
        return False, result.message or "stop_capture failed"

    def _start_3d_capture_stream_if_needed(self) -> tuple[bool, str]:
        client = self.scan3d_start_client
        if not client.wait_for_service(timeout_sec=3.0):
            return False, "/start_capture not available"
        try:
            result = client.call(Trigger.Request())
        except Exception as exc:
            return False, f"/start_capture call failed: {exc}"
        if result is None:
            return False, "/start_capture returned no response"
        if result.success:
            return True, result.message
        message = result.message or ""
        if "already" in message.lower() or "capturing" in message.lower():
            return True, message
        return False, message

    def _capture_3d_2d_by_service(self, state: CameraState) -> tuple[bool, str]:
        stop_ok, stop_msg = self._stop_3d_capture_stream_if_available()
        if not stop_ok:
            self.get_logger().warn(f"[3D相机纯2D] stop capture skipped: {stop_msg}")

        if not self.capture_2d_client.wait_for_service(timeout_sec=3.0):
            return False, "/capture_2d not available (rebuild welding_scan3d_camera_driver)"

        topic_state = self.cameras["3d"]
        since_recv_count = topic_state.recv_count
        try:
            result = self.capture_2d_client.call(Trigger.Request())
        except Exception as exc:
            return False, f"/capture_2d call failed: {exc}"
        if result is None:
            return False, "/capture_2d returned no response"
        if not result.success:
            return False, f"/capture_2d failed: {result.message}"

        if not self._wait_for_image(topic_state, since_recv_count=since_recv_count):
            return False, (
                f"/capture_2d succeeded but no new image on {topic_state.spec.topic} "
                f"within {self.image_wait_s}s"
            )
        if topic_state.latest_msg is None:
            return False, "/capture_2d succeeded but latest image is empty"

        try:
            path = self._save_image_msg(state, topic_state.latest_msg)
            return True, f"Saved 3D camera 2D image via /capture_2d (no projector): {path}"
        except Exception as exc:
            return False, f"save /capture_2d image failed: {exc}"

    def _capture_3d_by_service(self, state: CameraState) -> tuple[bool, str]:
        if not self.scan3d_client.wait_for_service(timeout_sec=3.0):
            return False, "/scan_3d not available"
        try:
            result = self.scan3d_client.call(Scan3D.Request())
        except Exception as exc:
            return False, f"/scan_3d call failed: {exc}"
        if result is None:
            return False, "/scan_3d returned no response"
        if not result.success:
            return False, f"/scan_3d failed: {result.message}"
        if result.image.width == 0 or result.image.height == 0:
            return False, "/scan_3d returned empty image"
        try:
            path = self._save_image_msg(state, result.image)
            state.latest_msg = result.image
            return True, f"Saved 3D image via /scan_3d: {path}"
        except Exception as exc:
            return False, f"save /scan_3d image failed: {exc}"

    def _make_session_dir(self, state: CameraState) -> Path:
        if state.current_session_dir is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H-%M-%S")
            state.current_session_dir = self.save_dir_root / state.spec.save_subdir / date_str / time_str
            state.current_session_dir.mkdir(parents=True, exist_ok=True)
        return state.current_session_dir

    def _make_filename(self, msg: Image) -> str:
        sec = int(msg.header.stamp.sec)
        nanosec = int(msg.header.stamp.nanosec)
        if sec != 0 or nanosec != 0:
            return f"{sec * 1_000_000 + nanosec // 1_000}.{self.image_ext}"
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{self.image_ext}"

    def _save_image_msg(self, state: CameraState, msg: Image) -> Path:
        session_dir = self._make_session_dir(state)
        image = imgmsg_to_bgr8(msg)
        image_path = session_dir / self._make_filename(msg)
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"cv2.imwrite failed: {image_path}")
        state.saved_count += 1
        return image_path

    def _prepare_camera(self, state: CameraState, *, enable_3d_stream: bool = False) -> tuple[bool, str]:
        driver_ok, driver_msg = self._start_driver_if_needed(state)
        if not driver_ok:
            return False, driver_msg

        if state.spec.key == "3d" and enable_3d_stream:
            stream_ok, stream_msg = self._start_3d_capture_stream_if_needed()
            if not stream_ok:
                self.get_logger().warn(f"[3D相机] /start_capture failed: {stream_msg}")

        return True, driver_msg

    def _capture_once(self, camera_key: str, response: Trigger.Response) -> Trigger.Response:
        state = self.cameras[camera_key]
        enable_3d_stream = camera_key == "3d"
        prepared, prepare_msg = self._prepare_camera(state, enable_3d_stream=enable_3d_stream)
        if not prepared:
            response.success = False
            response.message = f"[{state.spec.label}] driver not ready: {prepare_msg}"
            return response

        if camera_key == "3d_2d":
            ok, message = self._capture_3d_2d_by_service(state)
            response.success = ok
            response.message = f"[{state.spec.label}] {message}"
            if ok:
                self.get_logger().info(response.message)
            else:
                self.get_logger().error(response.message)
            return response

        if camera_key == "3d":
            ok, message = self._capture_3d_by_service(state)
            if ok:
                response.success = True
                response.message = f"[3D相机] {message}"
                self.get_logger().info(response.message)
                return response
            self.get_logger().warn(f"[3D相机] /scan_3d path failed, fallback to topic: {message}")

        if state.latest_msg is None:
            self._wait_for_image(state, since_recv_count=-1)

        if state.latest_msg is None:
            response.success = False
            response.message = (
                f"[{state.spec.label}] No image from {state.spec.topic}. "
                f"Driver status: {prepare_msg}. Check: ros2 topic hz {state.spec.topic}"
            )
            return response

        try:
            path = self._save_image_msg(state, state.latest_msg)
            response.success = True
            response.message = f"[{state.spec.label}] Saved one image: {path}"
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = f"[{state.spec.label}] save failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def _start_capture(self, camera_key: str, response: Trigger.Response) -> Trigger.Response:
        state = self.cameras[camera_key]
        prepared, prepare_msg = self._prepare_camera(
            state, enable_3d_stream=camera_key == "3d"
        )
        if not prepared:
            response.success = False
            response.message = f"[{state.spec.label}] driver not ready: {prepare_msg}"
            return response

        state.current_session_dir = None
        self._make_session_dir(state)
        state.running = True
        state.saved_count = 0
        response.success = True
        response.message = (
            f"[{state.spec.label}] Started continuous capture. "
            f"Saving to: {state.current_session_dir}. Driver: {prepare_msg}"
        )
        self.get_logger().info(response.message)
        return response

    def _stop_capture(self, camera_key: str, response: Trigger.Response) -> Trigger.Response:
        state = self.cameras[camera_key]
        if not state.running:
            response.success = False
            response.message = f"[{state.spec.label}] Not capturing."
            return response
        state.running = False
        response.success = True
        response.message = (
            f"[{state.spec.label}] Stopped capture. Saved {state.saved_count} images. "
            f"Last session dir: {state.current_session_dir}"
        )
        self.get_logger().info(response.message)
        return response

    def destroy_node(self) -> bool:
        if self.keep_launched_drivers_on_exit:
            return super().destroy_node()
        for state in self.cameras.values():
            process = state.launch_process
            if process is not None and process.poll() is None:
                self.get_logger().info(f"Stopping launched process for {state.spec.label}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedCameraCaptureNode()
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
