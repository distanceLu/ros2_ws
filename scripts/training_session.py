#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_session.py — 交互式训练数据采集会话

功能：
  1. 将机械臂移动到配置的初始位姿附近（可加随机偏移）
  2. 调用 training_data_collect 的开始/停止服务
  3. 支持单轮/多轮 episode 流程
  4. 确认采集已停止后，缓慢沿基座 +Z 抬笔，再做 XY 小幅随机偏移

前提（需在其他终端已启动）：
  - robot_driver_bridge_node（提供 /mov_jog、/tool_pos）
  - training_data_collect.py
  - 相机驱动（camera_capture_node 等）

用法：
  # 交互菜单（默认）；task_id 可由 --task-id / 环境变量 TASK_ID 指定
  python3 scripts/training_session.py
  python3 scripts/training_session.py --task-id 3
  python3 scripts/training_session.py task 3        # 仅写入采集节点 task_id

  # 命令行
  python3 scripts/training_session.py home          # 回初始位（随机偏移）
  python3 scripts/training_session.py home --exact  # 精确回初始位
  python3 scripts/training_session.py start         # 开始采集（会先写入 task_id）
  python3 scripts/training_session.py stop          # 停止采集
  python3 scripts/training_session.py episode       # 回初始位 → 开始 → Enter 停止
  python3 scripts/training_session.py episode -n 5  # 连续 5 轮
  python3 scripts/training_session.py status        # 查看当前位姿

  # 自定义配置
  python3 scripts/training_session.py -c scripts/my_pose.json home
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "training_session_config.json"
FRAMES_TO_VIDEO_SCRIPT = SCRIPT_DIR / "frames_to_video.py"
DEFAULT_PREVIEW_VIDEO = {
    "enabled": True,
    # 兼容旧字段 camera；优先使用 cameras 列表（纸面 + 熔池）
    "cameras": ["camera_paper_aruco", "camera_pool", "camera_pool1"],
    "camera": "camera_paper_aruco",
    "width": 400,
    "height": 320,
}
DEFAULT_POST_STOP_RETRACT = {
    "enabled": True,
    "lift_z_m": 0.10,
    "lift_speed_mps": 0.03,
    "xy_jitter_m": 0.01,
    "collect_settle_sec": 0.5,
    "speedl_service": "/speedl_s",
    "speed_stop_service": "/speed_stop",
    "control_period_ms": 100,
    "position_tolerance_m": 0.001,
    "max_runtime_sec": 20.0,
    "max_command_speed_mps": 0.05,
    "max_z_drop_m": 0.003,
    "max_xy_drift_m": 0.020,
}


def _prepend_env_path(var_name: str, value: str) -> None:
    current = os.environ.get(var_name, "")
    items = [item for item in current.split(os.pathsep) if item]
    if value in items:
        return
    os.environ[var_name] = value if not current else value + os.pathsep + current


def _bootstrap_local_python_paths() -> None:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        WORKSPACE_ROOT / "venv" / "lib" / pyver / "site-packages",
        WORKSPACE_ROOT / "install" / "robot_control" / "lib" / pyver / "site-packages",
    ]
    install_root = WORKSPACE_ROOT / "install"
    lib_dirs: list[Path] = []
    if install_root.is_dir():
        for child in sorted(install_root.iterdir()):
            lib_dir = child / "lib"
            if lib_dir.is_dir():
                lib_dirs.append(lib_dir)
            candidate = child / "lib" / pyver / "site-packages"
            if candidate.is_dir():
                candidates.append(candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_str = str(resolved)
        if resolved_str not in sys.path:
            sys.path.insert(0, resolved_str)
        _prepend_env_path("PYTHONPATH", resolved_str)
        _prepend_env_path("LD_LIBRARY_PATH", resolved_str)

    seen_libs: set[Path] = set()
    for lib_dir in lib_dirs:
        resolved = lib_dir.resolve()
        if resolved in seen_libs:
            continue
        seen_libs.add(resolved)
        _prepend_env_path("LD_LIBRARY_PATH", str(resolved))
        _prepend_env_path("LIBRARY_PATH", str(resolved))


def _preload_local_rosidl_libraries() -> None:
    install_root = WORKSPACE_ROOT / "install"
    if not install_root.is_dir():
        return
    for child in sorted(install_root.iterdir()):
        lib_dir = child / "lib"
        if not lib_dir.is_dir():
            continue
        for library_path in sorted(lib_dir.glob("lib*.so")):
            try:
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


_bootstrap_local_python_paths()
_preload_local_rosidl_libraries()

import rclpy
from common_interface.msg import TcpPos
from common_interface.srv import Move, SpecialSpeedl
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def apply_random_offset(base: dict[str, float], offset_cfg: dict[str, float]) -> dict[str, float]:
    def mm(key: str) -> float:
        return random.uniform(-offset_cfg[key], offset_cfg[key]) / 1000.0

    def deg(key: str) -> float:
        return math.radians(random.uniform(-offset_cfg[key], offset_cfg[key]))

    return {
        "x": base["x"] + mm("x_mm"),
        "y": base["y"] + mm("y_mm"),
        "z": base["z"] + mm("z_mm"),
        "rx": base["rx"] + deg("rx_deg"),
        "ry": base["ry"] + deg("ry_deg"),
        "rz": base["rz"] + deg("rz_deg"),
    }


def pose_to_list(pose: dict[str, float]) -> list[float]:
    return [pose["x"], pose["y"], pose["z"], pose["rx"], pose["ry"], pose["rz"]]


def format_pose(pose: dict[str, float]) -> str:
    return (
        f"x={pose['x']:.6f}, y={pose['y']:.6f}, z={pose['z']:.6f}, "
        f"rx={pose['rx']:.6f}, ry={pose['ry']:.6f}, rz={pose['rz']:.6f}"
    )


def parse_session_dir(activate_message: str) -> Optional[Path]:
    match = re.search(r"Started saving to (.+)", activate_message.strip())
    if not match:
        return None
    return Path(match.group(1).strip())


def preview_video_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULT_PREVIEW_VIDEO)
    cfg.update(config.get("preview_video") or {})
    return cfg


def post_stop_retract_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULT_POST_STOP_RETRACT)
    cfg.update(config.get("post_stop_retract") or {})
    return cfg


def preview_video_cameras(cfg: dict[str, Any]) -> list[str]:
    """解析预览相机列表：优先 cameras，兼容旧配置的单个 camera。"""
    cameras = cfg.get("cameras")
    if isinstance(cameras, str) and cameras.strip():
        return [cameras.strip()]
    if isinstance(cameras, (list, tuple)) and cameras:
        return [str(c).strip() for c in cameras if str(c).strip()]
    camera = str(cfg.get("camera", "camera_paper_aruco")).strip()
    return [camera] if camera else ["camera_paper_aruco"]


def export_preview_video_for_camera(
    session_dir: Path,
    camera: str,
    width: int,
    height: int,
) -> tuple[bool, str]:
    cam_dir = session_dir / camera
    if not cam_dir.is_dir():
        return False, f"{camera} 目录不存在，跳过: {cam_dir}"
    if not FRAMES_TO_VIDEO_SCRIPT.is_file():
        return False, f"找不到脚本: {FRAMES_TO_VIDEO_SCRIPT}"

    cmd = [
        sys.executable,
        str(FRAMES_TO_VIDEO_SCRIPT),
        str(cam_dir),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    print(f"生成预览视频({camera}): {' '.join(cmd)}")
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return False, f"启动 frames_to_video 失败({camera}): {exc}"

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if stdout:
        print(stdout)
    if completed.returncode != 0:
        detail = stderr or stdout or f"exit={completed.returncode}"
        return False, f"{camera} 预览视频生成失败: {detail}"

    out_path = cam_dir / f"{camera}_{width}x{height}.mp4"
    if out_path.is_file():
        return True, f"{camera} 预览视频已生成: {out_path}"
    return True, stdout or f"{camera} 预览视频已生成（目录: {cam_dir}）"


def export_paper_preview_video(
    session_dir: Path,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """轨迹结束后，把配置中的相机帧导出为小尺寸预览 mp4（默认纸面+熔池）。"""
    cfg = preview_video_config(config)
    if not cfg.get("enabled", True):
        return True, "预览视频已关闭（preview_video.enabled=false）"

    width = int(cfg.get("width", 400))
    height = int(cfg.get("height", 320))
    cameras = preview_video_cameras(cfg)
    messages: list[str] = []
    any_ok = False
    for camera in cameras:
        ok, message = export_preview_video_for_camera(session_dir, camera, width, height)
        messages.append(message)
        any_ok = any_ok or ok

    # 至少一个成功即视为整体可用；全部失败才返回 False
    return (any_ok if cameras else False), " | ".join(messages) if messages else "未配置预览相机"


class TrainingSessionNode(Node):
    def __init__(self, config: dict[str, Any], task_id: int = 0):
        super().__init__("training_session_node")
        self.config = config
        self.task_id = int(task_id)
        self.last_session_dir: Optional[Path] = None
        move_cfg = config["move"]
        collect_cfg = config["collect"]
        self.collect_node_name = str(
            collect_cfg.get("collect_node_name", "training_data_collect_node")
        )

        self.move_client = self.create_client(Move, move_cfg["service"])
        self.activate_client = self.create_client(Trigger, collect_cfg["activate_service"])
        self.deactivate_client = self.create_client(Trigger, collect_cfg["deactivate_service"])
        self.set_params_client = self.create_client(
            SetParameters, f"/{self.collect_node_name}/set_parameters"
        )
        retract_cfg = post_stop_retract_config(config)
        self.speedl_client = self.create_client(SpecialSpeedl, str(retract_cfg["speedl_service"]))
        self.speed_stop_client = self.create_client(Empty, str(retract_cfg["speed_stop_service"]))

        self.collecting = False
        self._latest_pose: Optional[TcpPos] = None
        self._latest_pose_mono = 0.0
        self.create_subscription(TcpPos, "/tool_pos", self._on_tool_pos, 10)

    def _on_tool_pos(self, msg: TcpPos) -> None:
        self._latest_pose = msg
        self._latest_pose_mono = time.monotonic()

    def wait_for_services(
        self,
        timeout_sec: float = 10.0,
        require_collect: bool = True,
    ) -> bool:
        """等待 ROS 服务。home/reset 只需 /mov_jog；采集相关命令才需要 activate/deactivate。"""
        ok_move = self.move_client.wait_for_service(timeout_sec=timeout_sec)
        if not ok_move:
            self.get_logger().error(f"服务不可用: {self.config['move']['service']}")
            return False
        if not require_collect:
            return True

        ok_activate = self.activate_client.wait_for_service(timeout_sec=timeout_sec)
        ok_deactivate = self.deactivate_client.wait_for_service(timeout_sec=timeout_sec)
        if not ok_activate:
            self.get_logger().error(
                f"服务不可用: {self.config['collect']['activate_service']} "
                "(请先运行 training_data_collect.py)"
            )
        if not ok_deactivate:
            self.get_logger().error(
                f"服务不可用: {self.config['collect']['deactivate_service']}"
            )
        return ok_activate and ok_deactivate

    def get_current_pose(
        self,
        timeout_sec: float = 2.0,
        newer_than: Optional[float] = None,
    ) -> Optional[dict[str, float]]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_pose is None:
                continue
            if newer_than is not None and self._latest_pose_mono <= newer_than:
                continue
            msg = self._latest_pose
            return {
                "x": msg.x,
                "y": msg.y,
                "z": msg.z,
                "rx": msg.rx,
                "ry": msg.ry,
                "rz": msg.rz,
            }
        return None

    def move_absolute(self, pose: dict[str, float]) -> bool:
        if not self.move_client.service_is_ready():
            self.get_logger().error("mov_jog 服务未就绪")
            return False

        request = Move.Request()
        request.a = float(pose["x"])
        request.b = float(pose["y"])
        request.c = float(pose["z"])
        request.d = float(pose["rx"])
        request.e = float(pose["ry"])
        request.f = float(pose["rz"])
        request.block = True
        request.name = ""

        self.get_logger().info(f"绝对运动: {format_pose(pose)}")
        future = self.move_client.call_async(request)
        timeout = float(self.config["move"]["timeout_sec"])
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

        if future.result() is None:
            self.get_logger().error(f"运动失败: {future.exception()}")
            return False

        result = future.result()
        if hasattr(result, "success") and not bool(result.success):
            self.get_logger().error("运动失败: /mov_jog 返回 success=false")
            return False

        settle = float(self.config["move"].get("settle_sec", 0.5))
        if settle > 0:
            time.sleep(settle)
        return True

    def _call_trigger(self, client, label: str) -> tuple[bool, str]:
        if not client.service_is_ready():
            return False, f"{label} 服务未就绪"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            return False, f"{label} 调用失败: {future.exception()}"
        return bool(result.success), result.message

    def start_collect(self) -> tuple[bool, str]:
        ok_param, param_msg = self.apply_task_id()
        if not ok_param:
            return False, param_msg
        ok, message = self._call_trigger(self.activate_client, "开始采集")
        if ok:
            self.collecting = True
        return ok, message

    def stop_collect(self) -> tuple[bool, str]:
        ok, message = self._call_trigger(self.deactivate_client, "停止采集")
        if ok:
            self.collecting = False
        return ok, message

    def stop_speed_motion(self) -> tuple[bool, str]:
        if not self.speed_stop_client.wait_for_service(timeout_sec=1.0):
            return False, "/speed_stop 服务未就绪"
        future = self.speed_stop_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() is None:
            return False, f"/speed_stop 调用失败: {future.exception()}"
        return True, "已停止速度运动"

    def send_speedl_velocity(
        self,
        linear_velocity_tool: tuple[float, float, float],
        duration_ms: int,
    ) -> tuple[bool, str]:
        if not self.speedl_client.service_is_ready():
            if not self.speedl_client.wait_for_service(timeout_sec=1.0):
                return False, "/speedl_s 服务未就绪"
        request = SpecialSpeedl.Request()
        request.x = float(linear_velocity_tool[0])
        request.y = float(linear_velocity_tool[1])
        request.z = float(linear_velocity_tool[2])
        request.rx = 0.0
        request.ry = 0.0
        request.rz = 0.0
        request.e1 = 0.0
        request.e2 = 0.0
        request.e3 = 0.0
        request.time = int(duration_ms)
        request.quit_distance = 0.0
        future = self.speedl_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            return False, f"/speedl_s 调用失败: {future.exception()}"
        if hasattr(result, "success") and not bool(result.success):
            return False, "/speedl_s 返回 success=false"
        return True, "ok"

    def lift_base_z_slow(self, cfg: dict[str, Any]) -> tuple[bool, str]:
        distance_m = float(cfg["lift_z_m"])
        speed_mps = float(cfg["lift_speed_mps"])
        period_ms = int(cfg["control_period_ms"])
        tolerance_m = float(cfg["position_tolerance_m"])
        max_runtime_sec = float(cfg["max_runtime_sec"])
        max_command_speed = float(cfg["max_command_speed_mps"])
        max_z_drop_m = float(cfg.get("max_z_drop_m", 0.003))
        max_xy_drift_m = float(cfg.get("max_xy_drift_m", 0.020))
        period_s = max(0.02, period_ms / 1000.0)

        start_pose = self.get_current_pose(timeout_sec=3.0)
        if start_pose is None:
            return False, "无法获取 /tool_pos，取消抬笔"
        z_target = start_pose["z"] + distance_m
        started = time.monotonic()
        iterations = 0
        self.get_logger().info(
            f"缓慢抬笔: 基座 +Z {distance_m * 1000:.0f} mm, "
            f"z={start_pose['z']:.4f} -> {z_target:.4f}, v={speed_mps:.3f} m/s"
        )

        try:
            while True:
                pose = self.get_current_pose(timeout_sec=0.8)
                if pose is None:
                    return False, "抬笔过程中丢失 /tool_pos"
                dz = pose["z"] - start_pose["z"]
                xy_drift = math.hypot(pose["x"] - start_pose["x"], pose["y"] - start_pose["y"])
                if dz < -max_z_drop_m:
                    return False, (
                        f"抬笔方向异常：基座 Z 下降 {abs(dz) * 1000:.1f} mm，已停止"
                    )
                if xy_drift > max_xy_drift_m:
                    return False, (
                        f"抬笔时 XY 漂移 {xy_drift * 1000:.1f} mm 超过 "
                        f"{max_xy_drift_m * 1000:.0f} mm，已停止"
                    )
                error = z_target - pose["z"]
                if abs(error) <= tolerance_m:
                    return True, (
                        f"已沿基座 +Z 抬升 {dz * 1000:.1f} mm "
                        f"(目标 {distance_m * 1000:.0f} mm, XY漂移 {xy_drift * 1000:.1f} mm)"
                    )
                if time.monotonic() - started > max_runtime_sec:
                    return False, (
                        f"抬笔超时: 已抬 {dz * 1000:.1f} mm / "
                        f"目标 {distance_m * 1000:.0f} mm"
                    )

                commanded = math.copysign(
                    min(speed_mps, abs(error) / period_s, max_command_speed),
                    error,
                )
                # Duco /speedl_s 的 servo 把增量加在 get_tcp_pose（基座系）上，
                # 因此这里直接发基座 +Z 速度，不再变换到工具系。
                pose_stamp = self._latest_pose_mono
                ok, msg = self.send_speedl_velocity((0.0, 0.0, commanded), period_ms)
                if not ok:
                    return False, f"抬笔失败: {msg}"
                deadline = time.monotonic() + period_s
                while time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.05)
                self.get_current_pose(timeout_sec=0.5, newer_than=pose_stamp)
                iterations += 1
                if iterations == 1 or iterations % 10 == 0:
                    self.get_logger().info(
                        f"抬笔中 z={pose['z']:.4f} target={z_target:.4f} "
                        f"err={error * 1000:.1f} mm xy_drift={xy_drift * 1000:.1f} mm"
                    )
        finally:
            self.stop_speed_motion()
        return False, "抬笔循环异常结束"

    def jitter_xy(self, cfg: dict[str, Any]) -> tuple[bool, str]:
        limit_m = float(cfg["xy_jitter_m"])
        pose = self.get_current_pose(timeout_sec=3.0)
        if pose is None:
            return False, "无法获取 /tool_pos，取消 XY 随机偏移"
        dx = random.uniform(-limit_m, limit_m)
        dy = random.uniform(-limit_m, limit_m)
        target = dict(pose)
        target["x"] += dx
        target["y"] += dy
        self.get_logger().info(
            f"XY 随机偏移: dx={dx * 1000:.1f} mm, dy={dy * 1000:.1f} mm"
        )
        if not self.move_absolute(target):
            return False, "XY 随机偏移运动失败"
        return True, f"XY 随机偏移 dx={dx * 1000:.1f} mm, dy={dy * 1000:.1f} mm"

    def post_stop_retract(self, collection_was_active: bool) -> tuple[bool, str]:
        """仅在确认采集已关闭、且本次确实停掉了一条正在录的轨迹后才移动。"""
        cfg = post_stop_retract_config(self.config)
        if not cfg.get("enabled", True):
            return True, "停采后抬笔已关闭（post_stop_retract.enabled=false）"
        if not collection_was_active:
            return True, "采集本来就未在进行，跳过抬笔"
        if self.collecting:
            return False, "采集仍标记为进行中，拒绝移动"

        settle = float(cfg.get("collect_settle_sec", 0.3))
        if settle > 0:
            time.sleep(settle)

        self.stop_speed_motion()
        ok, message = self.lift_base_z_slow(cfg)
        if not ok:
            return False, message
        messages = [message]
        jok, jmsg = self.jitter_xy(cfg)
        messages.append(jmsg)
        return jok, "；".join(messages)

    def apply_task_id(self, task_id: Optional[int] = None) -> tuple[bool, str]:
        """在 activate 之前把 task_id 写入采集节点，供 session_meta.json 记录。"""
        if task_id is not None:
            self.task_id = int(task_id)
        if not self.set_params_client.wait_for_service(timeout_sec=3.0):
            msg = (
                f"无法设置 task_id：服务 /{self.collect_node_name}/set_parameters 未就绪 "
                "(请确认 training_data_collect.py 已启动)"
            )
            self.get_logger().error(msg)
            return False, msg

        request = SetParameters.Request()
        param = Parameter()
        param.name = "task_id"
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_INTEGER
        param.value.integer_value = int(self.task_id)
        request.parameters = [param]

        future = self.set_params_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            msg = f"设置 task_id={self.task_id} 失败: {future.exception()}"
            self.get_logger().error(msg)
            return False, msg
        if not result.results or not result.results[0].successful:
            reason = result.results[0].reason if result.results else "unknown"
            msg = f"设置 task_id={self.task_id} 被拒绝: {reason}"
            self.get_logger().error(msg)
            return False, msg

        msg = f"已写入采集节点 task_id={self.task_id}"
        self.get_logger().info(msg)
        return True, msg

    def go_home(self, exact: bool = False) -> tuple[bool, dict[str, float]]:
        base = self.config["home_pose"]
        target = dict(base) if exact else apply_random_offset(base, self.config["random_offset"])
        ok = self.move_absolute(target)
        return ok, target

    def save_episode_meta(self, activate_message: str, home_pose: dict[str, float]) -> None:
        session_dir = parse_session_dir(activate_message)
        if session_dir is None:
            return
        self.last_session_dir = session_dir
        meta_path = session_dir / "episode_home_pose.json"
        payload = {
            "task_id": int(self.task_id),
            "home_pose_used": home_pose,
            "configured_home_pose": self.config["home_pose"],
            "random_offset_config": self.config["random_offset"],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self.get_logger().info(f"已写入 episode 初始位姿: {meta_path}")
        except OSError as exc:
            self.get_logger().warning(f"写入 episode_home_pose.json 失败: {exc}")

    def remember_session_dir(self, activate_message: str) -> Optional[Path]:
        session_dir = parse_session_dir(activate_message)
        if session_dir is not None:
            self.last_session_dir = session_dir
        return session_dir

    def get_save_dir_root(self) -> Path:
        """返回采集节点实际使用的采集根目录。"""
        configured_root = self.config.get("collect", {}).get("save_dir_root")
        save_dir_root = configured_root or os.environ.get("SAVE_DIR_ROOT") or "/home/shugen/yanjie/ros2_ws/data_collect"
        return Path(save_dir_root).expanduser().resolve()

    def find_latest_session_dir(self) -> Optional[Path]:
        """扫描采集根目录下所有含 session_meta.json 的目录，返回最新轨迹。"""
        root = self.get_save_dir_root()
        if not root.is_dir():
            return None

        candidates: list[Path] = []
        for meta_path in root.rglob("session_meta.json"):
            candidates.append(meta_path.parent)

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def delete_session_dir(self, session_dir: Path) -> tuple[bool, str]:
        """删除指定 session 目录（整个轨迹文件夹），含二次确认。"""
        if not session_dir.is_dir():
            return False, f"目录不存在: {session_dir}"

        # 不允许删除采集根目录或其上级，防止误删大量数据
        save_dir_root = self.get_save_dir_root()
        try:
            session_dir.resolve().relative_to(save_dir_root)
        except ValueError:
            return False, f"目录不在采集根目录下，拒绝删除: {session_dir}"

        # 不允许删除正在采集中的目录
        if session_dir == self.last_session_dir and self.collecting:
            return False, "该轨迹正在采集中，请先 [x] 停止后再删除"

        try:
            size = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
            file_count = sum(1 for f in session_dir.rglob("*") if f.is_file())
        except OSError:
            size = 0
            file_count = 0

        confirm = input(
            f"即将删除最近轨迹: {session_dir}\n"
            f"  文件数: {file_count}, 总大小: {size / (1024 * 1024):.1f} MB\n"
            "确认删除？输入 y 确认，其他取消: "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            return False, "已取消删除"

        try:
            shutil.rmtree(session_dir)
        except OSError as exc:
            return False, f"删除失败: {exc}"

        # 如果删除的是 last_session_dir，清空引用
        if self.last_session_dir == session_dir:
            self.last_session_dir = None

        return True, f"已删除轨迹: {session_dir}"

    def export_last_paper_preview(self) -> tuple[bool, str]:
        if self.last_session_dir is None:
            return False, "没有可用的 session 目录，无法生成预览视频"
        return export_paper_preview_video(self.last_session_dir, self.config)


def print_banner(config_path: Path, task_id: int) -> None:
    print("\n=== 训练数据采集会话 ===")
    print(f"配置文件: {config_path}")
    print(f"当前 task_id: {task_id}（开始采集前会写入 session_meta.json）")
    print(
        "命令: [w]基座X+点动  [s]基座X-点动  [h]回初始位  [z]输入task_id并开始  "
        "[x]停止并抬笔  [p]删除最近轨迹  [e]单轮  [r]多轮  [t]设task_id  "
        "[v]预览视频  [status]位姿  [q]退出"
    )
    print("提示: 每条轨迹确认停采后会先缓慢抬笔 10cm 再做 XY 随机偏移，并生成纸面+熔池 400x320 预览 mp4\n")


def cmd_status(node: TrainingSessionNode) -> int:
    pose = node.get_current_pose(timeout_sec=3.0)
    if pose is None:
        print("无法获取 /tool_pos（驱动是否已启动？）")
        return 1
    print("当前位姿:", format_pose(pose))
    print(f"当前 task_id: {node.task_id}")
    if node.last_session_dir is not None:
        print(f"最近 session: {node.last_session_dir}")
    home = node.config["home_pose"]
    dist = math.sqrt(
        (pose["x"] - home["x"]) ** 2
        + (pose["y"] - home["y"]) ** 2
        + (pose["z"] - home["z"]) ** 2
    )
    print(f"与配置初始位直线距离: {dist * 1000:.2f} mm")
    return 0


def cmd_set_task_id(node: TrainingSessionNode, task_id: Optional[int] = None) -> int:
    if task_id is None:
        raw = input(f"新的 task_id？当前={node.task_id} [{node.task_id}]: ").strip()
        if raw == "":
            task_id = node.task_id
        else:
            try:
                task_id = int(raw)
            except ValueError:
                print("task_id 必须是整数")
                return 1
            if task_id < 0:
                print("task_id 必须是非负整数")
                return 1
    ok, message = node.apply_task_id(task_id)
    print(message)
    return 0 if ok else 1


def prompt_task_id_before_start(node: TrainingSessionNode) -> bool:
    """按 z 开始采集前确认本条轨迹的 task_id；返回 False 表示取消。"""
    while True:
        raw = input(
            f"请输入本条轨迹 task_id（非负整数，当前={node.task_id}，"
            "直接回车沿用，q取消）: "
        ).strip()
        if raw.lower() in ("q", "quit", "cancel"):
            print("已取消开始采集")
            return False
        if raw == "":
            task_id = node.task_id
        else:
            try:
                task_id = int(raw)
            except ValueError:
                print("task_id 必须是非负整数，请重新输入")
                continue
            if task_id < 0:
                print("task_id 必须是非负整数，请重新输入")
                continue

        # start_collect() 会在调用 activate 服务前将该值写入采集节点。
        node.task_id = task_id
        print(f"本条轨迹 task_id={task_id}")
        return True


def cmd_home(node: TrainingSessionNode, exact: bool) -> int:
    ok, target = node.go_home(exact=exact)
    if not ok:
        print("回初始位失败")
        return 1
    print("目标位姿:", format_pose(target))
    actual = node.get_current_pose(timeout_sec=3.0)
    if actual:
        print("当前位姿:", format_pose(actual))
    return 0


def cmd_jog_x(node: TrainingSessionNode, direction: int) -> int:
    """基于当前位姿沿基座 X 轴移动一个固定步长。"""
    jog_cfg = node.config.get("keyboard_jog", {})
    step_m = float(jog_cfg.get("x_step_m", 0.005))
    if not 0.0 < step_m <= 0.05:
        print(f"keyboard_jog.x_step_m 必须在 (0, 0.05] m 内，当前={step_m}")
        return 1

    current = node.get_current_pose(timeout_sec=3.0)
    if current is None:
        print("无法获取 /tool_pos，取消 X 轴移动")
        return 1

    target = dict(current)
    delta_x = step_m if direction > 0 else -step_m
    target["x"] += delta_x
    label = "+X" if direction > 0 else "-X"
    print(f"键盘点动 {label}: {abs(delta_x) * 1000:.1f} mm")
    if not node.move_absolute(target):
        print(f"沿 {label} 移动失败")
        return 1

    actual = node.get_current_pose(timeout_sec=2.0)
    if actual:
        print("当前位姿:", format_pose(actual))
    return 0


def cmd_infer_reset(
    node: TrainingSessionNode,
    lift_z_m: Optional[float] = None,
    lift_z_min_m: Optional[float] = None,
    lift_z_max_m: Optional[float] = None,
    xy_jitter_m: Optional[float] = None,
) -> int:
    """推理用 reset：相对当前位置抬笔 + 小范围 XY，禁止大跨度绝对 home。"""
    cfg = post_stop_retract_config(node.config)
    if lift_z_m is not None:
        cfg["lift_z_m"] = float(lift_z_m)
    else:
        z_min = 0.05 if lift_z_min_m is None else float(lift_z_min_m)
        z_max = 0.15 if lift_z_max_m is None else float(lift_z_max_m)
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        cfg["lift_z_m"] = random.uniform(z_min, z_max)
    if xy_jitter_m is not None:
        cfg["xy_jitter_m"] = float(xy_jitter_m)
    # 硬上限：XY 单轴随机幅度不超过 2cm，避免危险大位移
    cfg["xy_jitter_m"] = min(float(cfg["xy_jitter_m"]), 0.02)

    if not node.move_client.wait_for_service(timeout_sec=8.0):
        print(f"运动服务未就绪: {node.config['move']['service']}")
        return 1
    if not node.speedl_client.wait_for_service(timeout_sec=5.0):
        print("抬笔需要 /speedl_s，服务未就绪")
        return 1

    print(
        f"infer-reset: 抬笔 +Z {float(cfg['lift_z_m']) * 1000:.1f} mm（随机），"
        f"随后 XY 随机 ≤ ±{float(cfg['xy_jitter_m']) * 1000:.0f} mm（相对当前位置）"
    )
    node.stop_speed_motion()
    ok, message = node.lift_base_z_slow(cfg)
    print(message)
    if not ok:
        return 1
    jok, jmsg = node.jitter_xy(cfg)
    print(jmsg)
    actual = node.get_current_pose(timeout_sec=3.0)
    if actual:
        print("当前位姿:", format_pose(actual))
    return 0 if jok else 1


def cmd_start(node: TrainingSessionNode, home_pose: Optional[dict[str, float]] = None) -> int:
    ok, message = node.start_collect()
    print(message)
    if not ok:
        return 1
    if home_pose is not None:
        node.save_episode_meta(message, home_pose)
    else:
        node.remember_session_dir(message)
    return 0


def cmd_stop(node: TrainingSessionNode, export_preview: bool = True) -> int:
    was_collecting = node.collecting
    ok, message = node.stop_collect()
    print(message)
    if not ok:
        print("采集未确认关闭，取消抬笔，机械臂保持不动")
        return 1
    if was_collecting:
        print("采集已关闭，开始抬笔")
    rok, rmsg = node.post_stop_retract(collection_was_active=was_collecting)
    print(rmsg)
    if not rok:
        print("[warn] 停采后抬笔失败，轨迹数据已保存")
    if export_preview:
        vok, vmsg = node.export_last_paper_preview()
        print(vmsg)
        if not vok:
            return 1
    return 0


def cmd_preview_video(node: TrainingSessionNode) -> int:
    ok, message = node.export_last_paper_preview()
    print(message)
    return 0 if ok else 1


def cmd_delete_last_session(node: TrainingSessionNode) -> int:
    """一键删除最近一条采集轨迹（按修改时间排序取最新的 session 目录）。"""
    target = node.last_session_dir
    if target is None or not target.is_dir():
        target = node.find_latest_session_dir()

    if target is None:
        print("未找到任何采集轨迹，无法删除")
        return 1

    ok, message = node.delete_session_dir(target)
    print(message)
    return 0 if ok else 1


def cmd_episode(
    node: TrainingSessionNode,
    exact: bool,
    duration_sec: Optional[float],
    count: int,
) -> int:
    for idx in range(1, count + 1):
        if count > 1:
            print(f"\n--- Episode {idx}/{count} (task_id={node.task_id}) ---")

        print("1/4 回初始位...")
        ok, target = node.go_home(exact=exact)
        if not ok:
            print("回初始位失败，中止")
            return 1
        print("   ", format_pose(target))

        print(f"2/4 写入 task_id={node.task_id} 并开始采集...")
        ok, message = node.start_collect()
        print("   ", message)
        if not ok:
            return 1
        node.save_episode_meta(message, target)

        if duration_sec is not None and duration_sec > 0:
            print(f"3/4 采集中 {duration_sec:.0f}s（可遥操）...")
            time.sleep(duration_sec)
        else:
            print("3/4 采集中 — 遥操完成后按 Enter 停止...")
            try:
                input()
            except EOFError:
                print("(非交互终端，10s 后自动停止)")
                time.sleep(10.0)

        was_collecting = node.collecting
        ok, message = node.stop_collect()
        print("   ", message)
        if not ok:
            print("采集未确认关闭，取消抬笔，机械臂保持不动")
            return 1

        print("   采集已关闭，抬笔离纸...")
        rok, rmsg = node.post_stop_retract(collection_was_active=was_collecting)
        print("   ", rmsg)
        if not rok:
            print("   [warn] 停采后抬笔失败，轨迹数据已保存")

        print("4/4 生成预览视频（纸面+熔池）...")
        vok, vmsg = node.export_last_paper_preview()
        print("   ", vmsg)
        if not vok:
            print("   [warn] 预览视频失败，不影响已保存的轨迹数据")

        if idx < count:
            ans = input("继续下一轮? [Y/n] ").strip().lower()
            if ans in ("n", "no"):
                break

    print("\nEpisode 完成")
    return 0


def switch_to_teleop_window() -> None:
    session = os.environ.get("COLLECT_TMUX_SESSION", "").strip()
    if not session or not os.environ.get("TMUX"):
        return
    completed = subprocess.run(
        ["tmux", "select-window", "-t", f"{session}:teleop"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        print(f"[warn] 无法自动切换 teleop 窗口: {detail}")


def run_interactive(node: TrainingSessionNode) -> int:
    print_banner(Path(node.config.get("_config_path", DEFAULT_CONFIG)), node.task_id)
    # 启动时先同步一次，确保后续 episode 写入正确 task_id
    ok, message = node.apply_task_id()
    print(message if ok else f"[warn] {message}")
    while True:
        try:
            choice = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            return 0

        if choice in ("q", "quit", "exit"):
            return 0
        if choice in ("w", "x+"):
            cmd_jog_x(node, direction=1)
        elif choice in ("s", "x-"):
            cmd_jog_x(node, direction=-1)
        elif choice in ("h", "home"):
            cmd_home(node, exact=False)
        elif choice in ("he", "home-exact"):
            cmd_home(node, exact=True)
        elif choice in ("z", "start"):
            if prompt_task_id_before_start(node) and cmd_start(node) == 0:
                print("采集已开始，自动切换到 teleop；按 Esc 将停止、保存并返回本窗口。")
                switch_to_teleop_window()
        elif choice in ("x", "stop"):
            cmd_stop(node)
        elif choice in ("p", "delete"):
            cmd_delete_last_session(node)
        elif choice in ("e", "episode"):
            cmd_episode(node, exact=False, duration_sec=None, count=1)
        elif choice in ("r", "repeat"):
            raw = input("连续几轮? [3] ").strip()
            count = int(raw) if raw else 3
            cmd_episode(node, exact=False, duration_sec=None, count=count)
        elif choice in ("t", "task", "task_id"):
            cmd_set_task_id(node)
        elif choice in ("v", "video", "preview"):
            cmd_preview_video(node)
        elif choice in ("pose", "status"):
            cmd_status(node)
        elif choice == "?":
            print_banner(Path(node.config.get("_config_path", DEFAULT_CONFIG)), node.task_id)
        else:
            print("未知命令，输入 ? 查看帮助")


def resolve_initial_task_id(config: dict[str, Any], cli_task_id: Optional[int]) -> int:
    if cli_task_id is not None:
        return int(cli_task_id)
    env_raw = os.environ.get("TASK_ID", "").strip()
    if env_raw != "":
        return int(env_raw)
    collect_cfg = config.get("collect", {})
    if "task_id" in collect_cfg:
        return int(collect_cfg["task_id"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="交互式训练数据采集会话")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON 配置文件路径",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="轮廓 task_id（默认读环境变量 TASK_ID 或配置 collect.task_id，否则 0）",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("interactive", help="交互菜单（默认）")
    sub.add_parser("status", help="显示当前位姿")

    home_p = sub.add_parser("home", help="移动到初始位姿附近")
    home_p.add_argument("--exact", action="store_true", help="不加随机偏移")

    reset_p = sub.add_parser(
        "infer-reset",
        help="推理用 reset：相对当前位姿抬笔 + 小范围 XY（默认≤2cm，不做绝对 home）",
    )
    reset_p.add_argument(
        "--lift-z-m",
        type=float,
        default=None,
        help="固定抬升(m)；若设置则不用随机区间",
    )
    reset_p.add_argument(
        "--lift-z-min-m",
        type=float,
        default=0.05,
        help="随机抬升下限(m)，默认 0.05（5cm）",
    )
    reset_p.add_argument(
        "--lift-z-max-m",
        type=float,
        default=0.15,
        help="随机抬升上限(m)，默认 0.15（15cm）",
    )
    reset_p.add_argument(
        "--xy-jitter-m",
        type=float,
        default=None,
        help="XY 随机半幅(m)；硬上限 0.02",
    )

    sub.add_parser("start", help="开始数据采集")
    sub.add_parser("stop", help="停止数据采集")

    task_p = sub.add_parser("task", help="设置采集节点 task_id")
    task_p.add_argument("value", type=int, help="轮廓 task_id（非负整数）")

    ep_p = sub.add_parser("episode", help="回初始位 → 开始采集 → 停止")
    ep_p.add_argument("--exact", action="store_true", help="精确回初始位")
    ep_p.add_argument("-d", "--duration", type=float, default=None, help="采集时长(秒)，默认等待 Enter")
    ep_p.add_argument("-n", "--count", type=int, default=1, help="连续 episode 数量")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "interactive"

    if not args.config.is_file():
        print(f"配置文件不存在: {args.config}")
        return 1

    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    try:
        task_id = resolve_initial_task_id(config, args.task_id)
    except ValueError:
        print("task_id 必须是整数（来自 --task-id / TASK_ID / 配置）")
        return 1
    if task_id < 0:
        print("task_id 必须是非负整数")
        return 1

    rclpy.init()
    node = TrainingSessionNode(config, task_id=task_id)
    try:
        if command != "status":
            # home / infer-reset 不依赖采集节点；推理会话也能用
            require_collect = command not in ("home", "infer-reset")
            if command != "infer-reset" and not node.wait_for_services(
                timeout_sec=8.0, require_collect=require_collect
            ):
                if require_collect:
                    print("部分 ROS 服务未就绪，请检查 robot 驱动与 training_data_collect.py")
                else:
                    print("运动服务未就绪，请检查 robot 驱动（/mov_jog）")
                if command == "interactive":
                    print("仍可使用 [p] 查看位姿；运动/采集命令可能失败")
                elif command in ("home", "start", "stop", "episode", "task"):
                    return 1

        if command == "interactive":
            return run_interactive(node)
        if command == "status":
            return cmd_status(node)
        if command == "home":
            return cmd_home(node, exact=args.exact)
        if command == "infer-reset":
            return cmd_infer_reset(
                node,
                lift_z_m=args.lift_z_m,
                lift_z_min_m=args.lift_z_min_m,
                lift_z_max_m=args.lift_z_max_m,
                xy_jitter_m=args.xy_jitter_m,
            )
        if command == "start":
            return cmd_start(node)
        if command == "stop":
            return cmd_stop(node)
        if command == "task":
            return cmd_set_task_id(node, task_id=args.value)
        if command == "episode":
            return cmd_episode(node, exact=args.exact, duration_sec=args.duration, count=args.count)

        parser.print_help()
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
