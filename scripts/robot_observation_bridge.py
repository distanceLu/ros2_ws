#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serve latest robot observations to non-ROS inference processes over ZeroMQ.

This bridge runs in the ROS2/Jazzy Python environment. It subscribes to
``/tool_pos`` and camera image topics, then exposes a lightweight ZMQ REP
endpoint that returns the latest observation as a Python object.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent


def _prepend_env_path(var_name: str, value: str) -> None:
    current = os.environ.get(var_name, "")
    items = [item for item in current.split(os.pathsep) if item]
    if value in items:
        return
    os.environ[var_name] = value if not current else value + os.pathsep + current


def bootstrap_local_ros_paths() -> None:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    install_root = WORKSPACE_ROOT / "install"
    candidates: list[Path] = []
    lib_dirs: list[Path] = []
    if install_root.is_dir():
        for child in sorted(install_root.iterdir()):
            lib_dir = child / "lib"
            if lib_dir.is_dir():
                lib_dirs.append(lib_dir)
            site_packages = child / "lib" / pyver / "site-packages"
            if site_packages.is_dir():
                candidates.append(site_packages)

    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        _prepend_env_path("PYTHONPATH", resolved)
        _prepend_env_path("LD_LIBRARY_PATH", resolved)

    for lib_dir in lib_dirs:
        resolved = str(lib_dir.resolve())
        _prepend_env_path("LD_LIBRARY_PATH", resolved)
        _prepend_env_path("LIBRARY_PATH", resolved)
        for library_path in sorted(lib_dir.glob("lib*.so")):
            try:
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def image_to_payload(msg: Any, received_at: float) -> dict[str, Any]:
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return {
        "height": int(msg.height),
        "width": int(msg.width),
        "encoding": str(msg.encoding),
        "is_bigendian": int(msg.is_bigendian),
        "step": int(msg.step),
        "data": bytes(msg.data),
        "stamp": float(stamp),
        "received_at": float(received_at),
    }


def bgr_image_to_payload(image: Any, received_at: float) -> dict[str, Any]:
    height, width = image.shape[:2]
    return {
        "height": int(height),
        "width": int(width),
        "encoding": "bgr8",
        "is_bigendian": 0,
        "step": int(width * 3),
        "data": image.tobytes(),
        "stamp": float(received_at),
        "received_at": float(received_at),
    }


def pose_to_payload(msg: Any, received_at: float) -> dict[str, Any]:
    return {
        "pose": [
            float(msg.x),
            float(msg.y),
            float(msg.z),
            float(msg.rx),
            float(msg.ry),
            float(msg.rz),
        ],
        "extra": [float(msg.e1), float(msg.e2), float(msg.e3)],
        "received_at": float(received_at),
    }


def missing_observation_fields(
    latest_pose: Optional[dict[str, Any]],
    latest_images: dict[str, dict[str, Any]],
    camera_names: list[str],
    max_age_sec: float,
) -> list[str]:
    now = time.time()
    missing: list[str] = []
    if latest_pose is None:
        missing.append("pose")
    elif now - float(latest_pose["received_at"]) > max_age_sec:
        missing.append("pose(stale)")
    for name in camera_names:
        image = latest_images.get(name)
        if image is None:
            missing.append(name)
        elif now - float(image["received_at"]) > max_age_sec:
            missing.append(f"{name}(stale)")
    return missing


def cmd_serve(args: argparse.Namespace) -> int:
    bootstrap_local_ros_paths()
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("缺少 pyzmq，请先安装 python3-zmq 或 pyzmq") from exc

    import rclpy
    from common_interface.msg import TcpPos
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_srvs.srv import Trigger

    class ObservationBridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("robot_observation_bridge")
            self.callback_group = ReentrantCallbackGroup()
            self.latest_pose: Optional[dict[str, Any]] = None
            self.latest_images: dict[str, dict[str, Any]] = {}
            self.request_count = 0
            self.ok_count = 0
            self.error_count = 0
            self.paper_capture: Optional[Any] = None
            self.paper_thread: Optional[threading.Thread] = None
            self.paper_running = False

            self.create_subscription(
                TcpPos,
                args.pose_topic,
                self._on_pose,
                10,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                Image,
                args.pool_topic,
                lambda msg: self._on_image("pool", msg),
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                Image,
                args.scan_topic,
                lambda msg: self._on_image("scan_2d", msg),
                qos_profile_sensor_data,
                callback_group=self.callback_group,
            )
            self.capture_client = self.create_client(
                Trigger,
                args.capture_2d_service,
                callback_group=self.callback_group,
            )

            self.zmq_context = zmq.Context.instance()
            self.zmq_socket = self.zmq_context.socket(zmq.REP)
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.bind(args.bind)
            self.create_timer(args.poll_period_sec, self._poll_zmq)
            self.create_timer(args.status_period_sec, self._log_status)
            if "paper_aruco" in args.camera_names:
                self._start_paper_camera_thread()

        def _on_pose(self, msg: Any) -> None:
            self.latest_pose = pose_to_payload(msg, time.time())

        def _on_image(self, name: str, msg: Any) -> None:
            self.latest_images[name] = image_to_payload(msg, time.time())

        def _start_paper_camera_thread(self) -> None:
            self.paper_running = True
            self.paper_thread = threading.Thread(
                target=self._paper_camera_loop,
                name="paper_aruco_camera",
                daemon=True,
            )
            self.paper_thread.start()

        def _paper_camera_loop(self) -> None:
            cap = cv2.VideoCapture(args.paper_camera_device, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.get_logger().error(f"Paper camera unavailable: {args.paper_camera_device}")
                self.paper_running = False
                return

            self.paper_capture = cap
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.paper_camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.paper_camera_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(max(0, args.paper_camera_warmup_frames)):
                cap.grab()

            period = 1.0 / max(0.1, args.paper_camera_hz)
            output_size = (args.paper_output_width, args.paper_output_height)
            self.get_logger().info(
                "paper_aruco camera started: "
                f"device={args.paper_camera_device}, hz={args.paper_camera_hz}, "
                f"capture={args.paper_camera_width}x{args.paper_camera_height}, "
                f"payload={args.paper_output_width}x{args.paper_output_height}"
            )
            try:
                while self.paper_running:
                    loop_start = time.monotonic()
                    ok = cap.grab()
                    frame = None
                    if ok:
                        ok, frame = cap.retrieve()
                    if ok and frame is not None and frame.size:
                        if args.paper_camera_rotate_180:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        if frame.shape[1] != output_size[0] or frame.shape[0] != output_size[1]:
                            frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
                        received_at = time.time()
                        self.latest_images["paper_aruco"] = bgr_image_to_payload(frame, received_at)
                    else:
                        self.get_logger().warning("Paper camera read failed", throttle_duration_sec=3.0)

                    sleep_s = period - (time.monotonic() - loop_start)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            finally:
                if self.paper_capture is cap:
                    self.paper_capture = None
                cap.release()
                self.get_logger().info("paper_aruco camera stopped")

        def _trigger_scan_capture(self) -> tuple[bool, str]:
            if not self.capture_client.service_is_ready():
                if not self.capture_client.wait_for_service(timeout_sec=args.capture_service_timeout_sec):
                    return False, f"capture service unavailable: {args.capture_2d_service}"
            future = self.capture_client.call_async(Trigger.Request())
            done = threading.Event()
            future.add_done_callback(lambda _: done.set())
            if not done.wait(timeout=args.capture_timeout_sec):
                return False, f"capture service timeout: {args.capture_2d_service}"
            result = future.result()
            if result is None:
                return False, f"capture service failed: {future.exception()}"
            if not result.success:
                return False, f"capture service returned failure: {result.message}"
            return True, str(result.message)

        def _build_response(self) -> dict[str, Any]:
            missing = missing_observation_fields(
                self.latest_pose,
                self.latest_images,
                args.camera_names,
                args.max_observation_age_sec,
            )
            if missing:
                self.error_count += 1
                return {
                    "ok": False,
                    "code": "MISSING_OBSERVATION",
                    "message": ",".join(missing),
                    "timestamp": time.time(),
                }
            self.ok_count += 1
            return {
                "ok": True,
                "code": "OK",
                "timestamp": time.time(),
                "pose": self.latest_pose,
                "images": {name: self.latest_images[name] for name in args.camera_names},
            }

        def _poll_zmq(self) -> None:
            try:
                request = self.zmq_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            except ValueError as exc:
                self.error_count += 1
                self.zmq_socket.send_pyobj(
                    {
                        "ok": False,
                        "code": "BAD_REQUEST",
                        "message": str(exc),
                        "timestamp": time.time(),
                    }
                )
                return

            self.request_count += 1
            capture_scan = bool(request.get("capture_scan", args.capture_scan))
            if capture_scan:
                ok, message = self._trigger_scan_capture()
                if not ok:
                    self.error_count += 1
                    self.zmq_socket.send_pyobj(
                        {
                            "ok": False,
                            "code": "CAPTURE_FAILED",
                            "message": message,
                            "timestamp": time.time(),
                        }
                    )
                    return

            self.zmq_socket.send_pyobj(self._build_response())

        def _log_status(self) -> None:
            observed = []
            if self.latest_pose is not None:
                observed.append("pose")
            observed.extend(sorted(self.latest_images))
            self.get_logger().info(
                "observation bridge status: "
                f"requests={self.request_count}, ok={self.ok_count}, errors={self.error_count}, "
                f"observed={observed}"
            )

        def destroy_node(self) -> bool:
            self.paper_running = False
            if self.paper_thread is not None:
                self.paper_thread.join(timeout=5.0)
                self.paper_thread = None
            self.zmq_socket.close(linger=0)
            return super().destroy_node()

    if not rclpy.ok():
        rclpy.init()
    node = ObservationBridgeNode()
    try:
        node.get_logger().info(f"observation bridge listening: {args.bind}")
        node.get_logger().info(
            f"topics: pose={args.pose_topic}, pool={args.pool_topic}, scan={args.scan_topic}; "
            f"cameras={args.camera_names}"
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.remove_node(node)
            executor.shutdown()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve latest ROS robot observations over ZMQ")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5554")
    parser.add_argument("--pose-topic", default="/tool_pos")
    parser.add_argument("--pool-topic", default="/pool_camera/image_raw")
    parser.add_argument("--scan-topic", default="/scan/image_raw")
    parser.add_argument("--capture-2d-service", default="/capture_2d")
    parser.add_argument("--capture-scan", action="store_true")
    parser.add_argument("--capture-timeout-sec", type=float, default=2.0)
    parser.add_argument("--capture-service-timeout-sec", type=float, default=1.0)
    parser.add_argument("--max-observation-age-sec", type=float, default=2.0)
    parser.add_argument("--poll-period-sec", type=float, default=0.02)
    parser.add_argument("--status-period-sec", type=float, default=5.0)
    parser.add_argument("--camera-names", nargs="+", default=["pool", "scan_2d", "paper_aruco"])
    parser.add_argument("--paper-camera-device", default="/dev/video0")
    parser.add_argument("--paper-camera-width", type=int, default=3840)
    parser.add_argument("--paper-camera-height", type=int, default=2160)
    parser.add_argument("--paper-camera-hz", type=float, default=15.0)
    parser.add_argument("--paper-camera-warmup-frames", type=int, default=10)
    parser.add_argument("--paper-camera-rotate-180", action="store_true", default=True)
    parser.add_argument("--no-paper-camera-rotate-180", action="store_false", dest="paper_camera_rotate_180")
    parser.add_argument("--paper-output-width", type=int, default=320)
    parser.add_argument("--paper-output-height", type=int, default=256)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    return int(cmd_serve(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
