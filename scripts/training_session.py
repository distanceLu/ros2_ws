#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_session.py — 交互式训练数据采集会话

功能：
  1. 将机械臂移动到配置的初始位姿附近（可加随机偏移）
  2. 调用 training_data_collect 的开始/停止服务
  3. 支持单轮/多轮 episode 流程

前提（需在其他终端已启动）：
  - robot_driver_bridge_node（提供 /mov_jog、/tool_pos）
  - training_data_collect.py
  - 相机驱动（camera_capture_node 等）

用法：
  # 交互菜单（默认）
  python3 scripts/training_session.py

  # 命令行
  python3 scripts/training_session.py home          # 回初始位（随机偏移）
  python3 scripts/training_session.py home --exact  # 精确回初始位
  python3 scripts/training_session.py start         # 开始采集
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
import sys
import time
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "training_session_config.json"


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
from common_interface.srv import Move
from rclpy.node import Node
from std_srvs.srv import Trigger


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


class TrainingSessionNode(Node):
    def __init__(self, config: dict[str, Any]):
        super().__init__("training_session_node")
        self.config = config
        move_cfg = config["move"]
        collect_cfg = config["collect"]

        self.move_client = self.create_client(Move, move_cfg["service"])
        self.activate_client = self.create_client(Trigger, collect_cfg["activate_service"])
        self.deactivate_client = self.create_client(Trigger, collect_cfg["deactivate_service"])

        self._latest_pose: Optional[TcpPos] = None
        self.create_subscription(TcpPos, "/tool_pos", self._on_tool_pos, 10)

    def _on_tool_pos(self, msg: TcpPos) -> None:
        self._latest_pose = msg

    def wait_for_services(self, timeout_sec: float = 10.0) -> bool:
        ok_move = self.move_client.wait_for_service(timeout_sec=timeout_sec)
        ok_activate = self.activate_client.wait_for_service(timeout_sec=timeout_sec)
        ok_deactivate = self.deactivate_client.wait_for_service(timeout_sec=timeout_sec)
        if not ok_move:
            self.get_logger().error(f"服务不可用: {self.config['move']['service']}")
        if not ok_activate:
            self.get_logger().error(
                f"服务不可用: {self.config['collect']['activate_service']} "
                "(请先运行 training_data_collect.py)"
            )
        if not ok_deactivate:
            self.get_logger().error(
                f"服务不可用: {self.config['collect']['deactivate_service']}"
            )
        return ok_move and ok_activate and ok_deactivate

    def get_current_pose(self, timeout_sec: float = 2.0) -> Optional[dict[str, float]]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_pose is not None:
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
        return self._call_trigger(self.activate_client, "开始采集")

    def stop_collect(self) -> tuple[bool, str]:
        return self._call_trigger(self.deactivate_client, "停止采集")

    def go_home(self, exact: bool = False) -> tuple[bool, dict[str, float]]:
        base = self.config["home_pose"]
        target = dict(base) if exact else apply_random_offset(base, self.config["random_offset"])
        ok = self.move_absolute(target)
        return ok, target

    def save_episode_meta(self, activate_message: str, home_pose: dict[str, float]) -> None:
        match = re.search(r"Started saving to (.+)", activate_message.strip())
        if not match:
            return
        session_dir = Path(match.group(1).strip())
        meta_path = session_dir / "episode_home_pose.json"
        payload = {
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


def print_banner(config_path: Path) -> None:
    print("\n=== 训练数据采集会话 ===")
    print(f"配置文件: {config_path}")
    print("命令: [h]回初始位  [s]开始  [x]停止  [e]单轮  [r]多轮  [p]位姿  [q]退出")
    print("提示: 需已启动 robot 驱动 + training_data_collect.py + 相机\n")


def cmd_status(node: TrainingSessionNode) -> int:
    pose = node.get_current_pose(timeout_sec=3.0)
    if pose is None:
        print("无法获取 /tool_pos（驱动是否已启动？）")
        return 1
    print("当前位姿:", format_pose(pose))
    home = node.config["home_pose"]
    dist = math.sqrt(
        (pose["x"] - home["x"]) ** 2
        + (pose["y"] - home["y"]) ** 2
        + (pose["z"] - home["z"]) ** 2
    )
    print(f"与配置初始位直线距离: {dist * 1000:.2f} mm")
    return 0


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


def cmd_start(node: TrainingSessionNode, home_pose: Optional[dict[str, float]] = None) -> int:
    ok, message = node.start_collect()
    print(message)
    if ok and home_pose is not None:
        node.save_episode_meta(message, home_pose)
    return 0 if ok else 1


def cmd_stop(node: TrainingSessionNode) -> int:
    ok, message = node.stop_collect()
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
            print(f"\n--- Episode {idx}/{count} ---")

        print("1/3 回初始位...")
        ok, target = node.go_home(exact=exact)
        if not ok:
            print("回初始位失败，中止")
            return 1
        print("   ", format_pose(target))

        print("2/3 开始采集...")
        ok, message = node.start_collect()
        print("   ", message)
        if not ok:
            return 1
        node.save_episode_meta(message, target)

        if duration_sec is not None and duration_sec > 0:
            print(f"3/3 采集中 {duration_sec:.0f}s（可遥操）...")
            time.sleep(duration_sec)
        else:
            print("3/3 采集中 — 遥操完成后按 Enter 停止...")
            try:
                input()
            except EOFError:
                print("(非交互终端，10s 后自动停止)")
                time.sleep(10.0)

        ok, message = node.stop_collect()
        print("   ", message)
        if not ok:
            return 1

        if idx < count:
            ans = input("继续下一轮? [Y/n] ").strip().lower()
            if ans in ("n", "no"):
                break

    print("\nEpisode 完成")
    return 0


def run_interactive(node: TrainingSessionNode) -> int:
    print_banner(Path(node.config.get("_config_path", DEFAULT_CONFIG)))
    while True:
        try:
            choice = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            return 0

        if choice in ("q", "quit", "exit"):
            return 0
        if choice in ("h", "home"):
            cmd_home(node, exact=False)
        elif choice in ("he", "home-exact"):
            cmd_home(node, exact=True)
        elif choice in ("s", "start"):
            cmd_start(node)
        elif choice in ("x", "stop"):
            cmd_stop(node)
        elif choice in ("e", "episode"):
            cmd_episode(node, exact=False, duration_sec=None, count=1)
        elif choice in ("r", "repeat"):
            raw = input("连续几轮? [3] ").strip()
            count = int(raw) if raw else 3
            cmd_episode(node, exact=False, duration_sec=None, count=count)
        elif choice in ("p", "pose", "status"):
            cmd_status(node)
        elif choice == "?":
            print_banner(Path(node.config.get("_config_path", DEFAULT_CONFIG)))
        else:
            print("未知命令，输入 ? 查看帮助")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="交互式训练数据采集会话")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON 配置文件路径",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("interactive", help="交互菜单（默认）")
    sub.add_parser("status", help="显示当前位姿")

    home_p = sub.add_parser("home", help="移动到初始位姿附近")
    home_p.add_argument("--exact", action="store_true", help="不加随机偏移")

    sub.add_parser("start", help="开始数据采集")
    sub.add_parser("stop", help="停止数据采集")

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

    rclpy.init()
    node = TrainingSessionNode(config)
    try:
        if command != "status":
            if not node.wait_for_services(timeout_sec=8.0):
                print("部分 ROS 服务未就绪，请检查 robot 驱动与 training_data_collect.py")
                if command == "interactive":
                    print("仍可使用 [p] 查看位姿；运动/采集命令可能失败")
                elif command in ("home", "start", "stop", "episode"):
                    return 1

        if command == "interactive":
            return run_interactive(node)
        if command == "status":
            return cmd_status(node)
        if command == "home":
            return cmd_home(node, exact=args.exact)
        if command == "start":
            return cmd_start(node)
        if command == "stop":
            return cmd_stop(node)
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
