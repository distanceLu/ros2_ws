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
    "camera": "camera_paper_aruco",
    "width": 400,
    "height": 320,
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
from common_interface.srv import Move
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
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


def parse_session_dir(activate_message: str) -> Optional[Path]:
    match = re.search(r"Started saving to (.+)", activate_message.strip())
    if not match:
        return None
    return Path(match.group(1).strip())


def preview_video_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULT_PREVIEW_VIDEO)
    cfg.update(config.get("preview_video") or {})
    return cfg


def export_paper_preview_video(
    session_dir: Path,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """轨迹结束后，把纸面相机帧导出为小尺寸预览 mp4，方便当场回看。"""
    cfg = preview_video_config(config)
    if not cfg.get("enabled", True):
        return True, "纸面预览视频已关闭（preview_video.enabled=false）"

    camera = str(cfg.get("camera", "camera_paper_aruco"))
    width = int(cfg.get("width", 400))
    height = int(cfg.get("height", 320))
    paper_dir = session_dir / camera
    if not paper_dir.is_dir():
        return False, f"纸面目录不存在，跳过预览视频: {paper_dir}"
    if not FRAMES_TO_VIDEO_SCRIPT.is_file():
        return False, f"找不到脚本: {FRAMES_TO_VIDEO_SCRIPT}"

    cmd = [
        sys.executable,
        str(FRAMES_TO_VIDEO_SCRIPT),
        str(paper_dir),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    print(f"生成纸面预览视频: {' '.join(cmd)}")
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return False, f"启动 frames_to_video 失败: {exc}"

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if stdout:
        print(stdout)
    if completed.returncode != 0:
        detail = stderr or stdout or f"exit={completed.returncode}"
        return False, f"预览视频生成失败: {detail}"

    out_path = paper_dir / f"{camera}_{width}x{height}.mp4"
    if out_path.is_file():
        return True, f"预览视频已生成: {out_path}"
    # 脚本可能改了命名；成功时仍返回 stdout 摘要
    return True, stdout or f"预览视频已生成（目录: {paper_dir}）"


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
        ok_param, param_msg = self.apply_task_id()
        if not ok_param:
            return False, param_msg
        return self._call_trigger(self.activate_client, "开始采集")

    def stop_collect(self) -> tuple[bool, str]:
        return self._call_trigger(self.deactivate_client, "停止采集")

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

    def export_last_paper_preview(self) -> tuple[bool, str]:
        if self.last_session_dir is None:
            return False, "没有可用的 session 目录，无法生成预览视频"
        return export_paper_preview_video(self.last_session_dir, self.config)


def print_banner(config_path: Path, task_id: int) -> None:
    print("\n=== 训练数据采集会话 ===")
    print(f"配置文件: {config_path}")
    print(f"当前 task_id: {task_id}（开始采集前会写入 session_meta.json）")
    print("命令: [h]回初始位  [s]输入task_id并开始  [x]停止  [e]单轮  [r]多轮  [t]设task_id  [v]预览视频  [p]位姿  [q]退出")
    print("提示: 每条轨迹停止后会自动生成纸面 400x320 预览 mp4\n")


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
    """按 s 开始采集前确认本条轨迹的 task_id；返回 False 表示取消。"""
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
    ok, message = node.stop_collect()
    print(message)
    if not ok:
        return 1
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

        ok, message = node.stop_collect()
        print("   ", message)
        if not ok:
            return 1

        print("4/4 生成纸面预览视频...")
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
        if choice in ("h", "home"):
            cmd_home(node, exact=False)
        elif choice in ("he", "home-exact"):
            cmd_home(node, exact=True)
        elif choice in ("s", "start"):
            if prompt_task_id_before_start(node):
                cmd_start(node)
        elif choice in ("x", "stop"):
            cmd_stop(node)
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
        elif choice in ("p", "pose", "status"):
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
            if not node.wait_for_services(timeout_sec=8.0):
                print("部分 ROS 服务未就绪，请检查 robot 驱动与 training_data_collect.py")
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
