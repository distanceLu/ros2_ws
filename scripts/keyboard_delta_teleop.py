#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused-window W/S teleoperation using small base-frame X pose deltas."""

from __future__ import annotations

import argparse
import os
import queue
import select
import subprocess
import sys
import termios
import threading
import time
import tkinter as tk
import tty
from tkinter import ttk
from typing import Optional

import rclpy
from common_interface.msg import TcpPos
from common_interface.srv import Move, SpecialSpeedl
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty


class DeltaTeleopNode(Node):
    def __init__(self, args: argparse.Namespace, events: queue.SimpleQueue[tuple[str, str]]):
        super().__init__("keyboard_delta_teleop")
        self.args = args
        self.events = events
        self._lock = threading.Lock()
        self._latest_pose: Optional[dict[str, float]] = None
        self._latest_pose_time = 0.0
        self._move_in_flight = False
        self._failed = False

        self.move_client = self.create_client(Move, args.move_service)
        self.speedl_client = self.create_client(SpecialSpeedl, args.speedl_service)
        self.speed_stop_client = self.create_client(Empty, args.speed_stop_service)
        self.create_subscription(TcpPos, args.pose_topic, self._on_pose, 10)

    def _on_pose(self, msg: TcpPos) -> None:
        pose = {
            "x": float(msg.x),
            "y": float(msg.y),
            "z": float(msg.z),
            "rx": float(msg.rx),
            "ry": float(msg.ry),
            "rz": float(msg.rz),
        }
        with self._lock:
            self._latest_pose = pose
            self._latest_pose_time = time.monotonic()

    def snapshot(self) -> tuple[Optional[dict[str, float]], float, bool, bool]:
        with self._lock:
            pose = dict(self._latest_pose) if self._latest_pose is not None else None
            age = time.monotonic() - self._latest_pose_time if pose is not None else float("inf")
            return pose, age, self._move_in_flight, self._failed

    def clear_failure(self) -> None:
        with self._lock:
            self._failed = False

    def send_speed(self, velocity: tuple[float, float, float]) -> tuple[bool, str]:
        if not self.speedl_client.service_is_ready():
            return False, f"速度服务未就绪: {self.args.speedl_service}"
        request = SpecialSpeedl.Request()
        request.x = float(velocity[0])
        request.y = float(velocity[1])
        request.z = float(velocity[2])
        request.rx = request.ry = request.rz = 0.0
        request.e1 = request.e2 = request.e3 = 0.0
        request.time = max(20, round(1000.0 / self.args.control_hz))
        request.quit_distance = 0.0
        future = self.speedl_client.call_async(request)
        deadline = time.monotonic() + 1.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.002)
        result = future.result() if future.done() else None
        if result is None:
            return False, f"{self.args.speedl_service} 调用失败: {future.exception()}"
        if hasattr(result, "success") and not bool(result.success):
            return False, f"{self.args.speedl_service} 返回 success=false"
        return True, (
            f"speed=({velocity[0] * 1000:.1f}, {velocity[1] * 1000:.1f}, "
            f"{velocity[2] * 1000:.1f})mm/s"
        )

    @staticmethod
    def _wait_future(future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.002)
        return future.done()

    def stop_speed(self) -> tuple[bool, str]:
        # Duco 驱动会保持最后一次 /speedl_s 速度。先写入零速度，即使
        # /speed_stop 暂时不可用，也能立即清除驱动中的持续速度目标。
        zero_ok, zero_message = self.send_speed((0.0, 0.0, 0.0))
        if not self.speed_stop_client.service_is_ready():
            return zero_ok, (
                "已发送零速度；/speed_stop 未就绪"
                if zero_ok
                else f"零速度失败: {zero_message}; /speed_stop 未就绪"
            )
        future = self.speed_stop_client.call_async(Empty.Request())
        stop_ok = self._wait_future(future, 1.0) and future.exception() is None
        if stop_ok:
            return True, "已发送零速度并调用 /speed_stop"
        return zero_ok, "已发送零速度，但 /speed_stop 调用超时或失败"

    def send_x_delta(self, delta_x: float) -> tuple[bool, str]:
        if abs(delta_x) > self.args.max_delta_m:
            return False, f"拒绝超限 Delta: {delta_x * 1000:.3f} mm"
        if not self.move_client.service_is_ready():
            return False, f"运动服务未就绪: {self.args.move_service}"

        with self._lock:
            if self._move_in_flight:
                return False, "busy"
            if self._latest_pose is None:
                return False, f"尚未收到 {self.args.pose_topic}"
            age = time.monotonic() - self._latest_pose_time
            if age > self.args.pose_timeout_sec:
                return False, f"位姿已过期: {age:.2f}s"
            target = dict(self._latest_pose)
            target["x"] += delta_x
            if self.args.x_min is not None and target["x"] < self.args.x_min:
                return False, f"到达 X 下限 {self.args.x_min:.4f}m"
            if self.args.x_max is not None and target["x"] > self.args.x_max:
                return False, f"到达 X 上限 {self.args.x_max:.4f}m"
            self._move_in_flight = True

        request = Move.Request()
        request.a = target["x"]
        request.b = target["y"]
        request.c = target["z"]
        request.d = target["rx"]
        request.e = target["ry"]
        request.f = target["rz"]
        request.block = True
        request.name = ""
        future = self.move_client.call_async(request)
        future.add_done_callback(self._on_move_done)
        return True, f"target x={target['x']:.6f}m"

    def _on_move_done(self, future) -> None:
        error = ""
        try:
            result = future.result()
            if result is None:
                error = "运动服务未返回结果"
            elif hasattr(result, "success") and not bool(result.success):
                error = "/mov_jog 返回 success=false"
        except Exception as exc:  # rclpy future propagates service exceptions here
            error = f"运动调用异常: {exc}"

        with self._lock:
            self._move_in_flight = False
            if error:
                self._failed = True
        if error:
            self.events.put(("error", error))


class TeleopWindow:
    def __init__(self, node: DeltaTeleopNode, args: argparse.Namespace):
        self.node = node
        self.args = args
        self.root = tk.Tk()
        self.root.title("机械臂 X 轴 Delta 遥操作")
        self.root.geometry("520x330")
        self.root.minsize(480, 300)

        self.keys = {"w": False, "s": False}
        self.enabled = True
        self.closing = False
        self.direction_text = tk.StringVar(value="停止")
        self.pose_text = tk.StringVar(value="等待 /tool_pos ...")
        self.service_text = tk.StringVar(value=f"等待 {args.move_service} ...")
        self.status_text = tk.StringVar(value="请单击窗口后按住 W 或 S")

        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="W / S 实时 Delta 位姿遥操作", font=("Sans", 18, "bold")).pack(pady=(0, 14))
        ttk.Label(frame, text="W：基座 +X    S：基座 -X    松开：停止", font=("Sans", 13)).pack()
        ttk.Label(frame, textvariable=self.direction_text, font=("Sans", 26, "bold")).pack(pady=14)
        ttk.Label(frame, textvariable=self.pose_text).pack()
        ttk.Label(frame, textvariable=self.service_text).pack(pady=(4, 0))
        ttk.Label(frame, textvariable=self.status_text, foreground="#a33", wraplength=460).pack(pady=12)
        ttk.Label(frame, text="Space：禁用运动    Enter：重新启用    Esc：退出\n窗口失焦会自动停止", justify=tk.CENTER).pack(side=tk.BOTTOM)

        for key in ("w", "W", "s", "S"):
            self.root.bind(f"<KeyPress-{key}>", self._key_press)
            self.root.bind(f"<KeyRelease-{key}>", self._key_release)
        self.root.bind("<space>", self._disable)
        self.root.bind("<Return>", self._enable)
        self.root.bind("<Escape>", self._close)
        self.root.bind("<FocusOut>", self._focus_out)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.period_ms = max(20, round(1000.0 / args.control_hz))
        self.delta_m = min(args.xyz_speed_mps / args.control_hz, args.max_delta_m)
        self.root.after(self.period_ms, self._control_tick)
        self.root.after(100, self._status_tick)
        self.root.after(150, self.root.focus_force)

    def _key_press(self, event) -> None:
        key = event.keysym.lower()
        if key in self.keys and self.enabled:
            self.keys[key] = True

    def _key_release(self, event) -> None:
        key = event.keysym.lower()
        if key in self.keys:
            self.keys[key] = False

    def _stop_keys(self) -> None:
        self.keys["w"] = False
        self.keys["s"] = False
        self.direction_text.set("停止")

    def _disable(self, _event=None) -> None:
        self.enabled = False
        self._stop_keys()
        self.status_text.set("运动已禁用；按 Enter 重新启用")

    def _enable(self, _event=None) -> None:
        self._stop_keys()
        self.node.clear_failure()
        self.enabled = True
        self.status_text.set("运动已启用")

    def _focus_out(self, _event=None) -> None:
        self._stop_keys()
        if self.enabled:
            self.status_text.set("窗口已失焦，运动停止；单击窗口后继续")

    def _direction(self) -> int:
        if not self.enabled:
            return 0
        if self.keys["w"] == self.keys["s"]:
            return 0
        return 1 if self.keys["w"] else -1

    def _control_tick(self) -> None:
        if self.closing:
            return
        direction = self._direction()
        if direction:
            ok, message = self.node.send_x_delta(direction * self.delta_m)
            if ok:
                self.direction_text.set("+X" if direction > 0 else "-X")
                self.status_text.set(message)
            elif message != "busy":
                self._stop_keys()
                self.status_text.set(message)
        else:
            self.direction_text.set("停止")
        self.root.after(self.period_ms, self._control_tick)

    def _status_tick(self) -> None:
        if self.closing:
            return
        pose, age, busy, failed = self.node.snapshot()
        if pose is None:
            self.pose_text.set(f"等待 {self.args.pose_topic} ...")
        else:
            self.pose_text.set(f"当前 X：{pose['x']:.6f} m    位姿年龄：{age:.2f} s")
        ready = self.node.move_client.service_is_ready()
        state = "执行 Delta 中" if busy else ("错误锁定" if failed else "就绪")
        self.service_text.set(
            f"{self.args.move_service}：{'已连接' if ready else '未连接'}    "
            f"状态：{state}    设定速度：{self.args.xyz_speed_mps * 1000:.1f} mm/s"
        )
        while True:
            try:
                kind, message = self.node.events.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                self._disable()
                self.status_text.set(f"{message}；按 Enter 确认并重新启用")
        self.root.after(100, self._status_tick)

    def _close(self, _event=None) -> None:
        if self.closing:
            return
        self.closing = True
        self.enabled = False
        self._stop_keys()
        self.root.quit()

    def run(self) -> None:
        self.root.mainloop()
        self.root.destroy()


class TerminalTeleop:
    """Raw-terminal fallback for headless sessions.

    Terminals do not expose key-release events. W/S key repeats refresh a short
    movement lease; when repeats stop, no further Delta commands are sent.
    """

    def __init__(self, node: DeltaTeleopNode, args: argparse.Namespace):
        self.node = node
        self.args = args
        self.period_s = max(0.02, 1.0 / args.control_hz)
        self.delta_m = min(args.xyz_speed_mps / args.control_hz, args.max_delta_m)
        self.velocity = (0.0, 0.0, 0.0)
        self.active_until = 0.0
        self.enabled = True
        self.last_status = ""
        self.finish_episode = False
        self.speed_active = False

    def _print_status(self, text: str) -> None:
        if text == self.last_status:
            return
        self.last_status = text
        print(f"\r\033[2K{text}", end="", flush=True)

    def _handle_key(self, key: str) -> bool:
        if key == "\x1b":
            self.velocity = (0.0, 0.0, 0.0)
            self.active_until = 0.0
            self.finish_episode = True
            return False
        if key == "q":
            self.velocity = (0.0, 0.0, 0.0)
            self.active_until = 0.0
            return False
        if key == " ":
            self.enabled = False
            self.velocity = (0.0, 0.0, 0.0)
            self._print_status("[禁用] 按 e 重新启用，q 退出")
        elif key == "e":
            self.node.clear_failure()
            self.enabled = True
            self.velocity = (0.0, 0.0, 0.0)
            self._print_status("[就绪] W/S= X，A/D= Y，上下= Z")
        elif key in ("w", "s", "a", "d", "up", "down") and self.enabled:
            self.velocity = self._velocity_for_key(key)
            self.active_until = time.monotonic() + self.args.key_release_timeout_sec
        elif key == "\x1b[":
            return True
        return True

    def _velocity_for_key(self, key: str) -> tuple[float, float, float]:
        speed = self.args.xyz_speed_mps
        return {
            "w": (speed, 0.0, 0.0),
            "s": (-speed, 0.0, 0.0),
            "a": (0.0, speed, 0.0),
            "d": (0.0, -speed, 0.0),
            "up": (0.0, 0.0, speed),
            "down": (0.0, 0.0, -speed),
        }[key]

    def run(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("终端遥操作需要交互式 TTY")
        old_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        print("\033[2J\033[H", end="")
        print("机械臂 XYZ 三轴速度遥操作（终端模式）")
        print("W：基座 +X；S：基座 -X；A：基座 +Y；D：基座 -Y")
        print("↑：基座 +Z；↓：基座 -Z；停止按键：自动停止")
        print("Space：禁用；e：启用；q：退出；Esc：结束采集并保存")
        print(
            f"XYZ统一速度={self.args.xyz_speed_mps * 1000:.1f}mm/s, "
            f"Delta={self.delta_m * 1000:.3f}mm, "
            f"按键停止超时={self.args.key_release_timeout_sec * 1000:.0f}ms\n"
        )
        try:
            running = True
            next_tick = time.monotonic()
            while running and rclpy.ok():
                readable, _, _ = select.select([sys.stdin], [], [], 0.01)
                if readable:
                    data = os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                    keys = []
                    index = 0
                    while index < len(data):
                        if data.startswith("\x1b[A", index):
                            keys.append("up")
                            index += 3
                        elif data.startswith("\x1b[B", index):
                            keys.append("down")
                            index += 3
                        else:
                            keys.append(data[index])
                            index += 1
                    for key in keys:
                        if not self._handle_key(key):
                            running = False
                            break

                now = time.monotonic()
                if now >= self.active_until:
                    self.velocity = (0.0, 0.0, 0.0)
                if now >= next_tick:
                    next_tick = now + self.period_s
                    pose, age, busy, failed = self.node.snapshot()
                    moving = any(abs(value) > 0.0 for value in self.velocity)
                    if moving and self.enabled and not failed:
                        ok, message = self.node.send_speed(self.velocity)
                        self.speed_active = ok
                        if ok:
                            labels = ("X", "Y", "Z")
                            axis_index = next(i for i, value in enumerate(self.velocity) if value != 0.0)
                            sign = "+" if self.velocity[axis_index] > 0 else "-"
                            pose_text = (
                                f" xyz=({pose['x']:.4f},{pose['y']:.4f},{pose['z']:.4f})m"
                                if pose else ""
                            )
                            self._print_status(f"[{sign}{labels[axis_index]}]{pose_text} {message}")
                        else:
                            self.velocity = (0.0, 0.0, 0.0)
                            self._print_status(f"[停止] {message}")
                    elif self.speed_active:
                        stopped, stop_message = self.node.stop_speed()
                        self.speed_active = False
                        self._print_status(
                            f"[停止] {stop_message}" if stopped else f"[停止失败] {stop_message}"
                        )
                    elif failed:
                        self.enabled = False
                        self.velocity = (0.0, 0.0, 0.0)
                        self._print_status("[错误锁定] 按 e 确认并重新启用")
                    elif pose is None:
                        self._print_status(f"[等待] {self.args.pose_topic}")
                    elif age > self.args.pose_timeout_sec:
                        self._print_status(f"[停止] 位姿已过期 {age:.2f}s")
                    elif not busy:
                        self._print_status(f"[停止] x={pose['x']:.6f}m")

                while True:
                    try:
                        kind, message = self.node.events.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "error":
                        self.enabled = False
                        self.velocity = (0.0, 0.0, 0.0)
                        self._print_status(f"[错误] {message}；按 e 重新启用")
        finally:
            self.velocity = (0.0, 0.0, 0.0)
            # 无论本地状态如何都发送零速度和 /speed_stop，覆盖异常退出、
            # Esc 与按键超时恰好并发等情况。
            stopped, stop_message = self.node.stop_speed()
            self.speed_active = False
            print(f"\n{'[已停止]' if stopped else '[停止失败]'} {stop_message}")
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
            print("\n遥操作已退出")
            if self.finish_episode:
                self._finish_and_return_to_session()

    def _finish_and_return_to_session(self) -> None:
        session = os.environ.get("COLLECT_TMUX_SESSION", "").strip()
        if not session or not os.environ.get("TMUX"):
            print("未处于 collect_data tmux 会话，无法自动停采；请在 session 窗口输入 x")
            return
        # session 正阻塞于 input('>>> ')，注入 x+Enter，复用原有停采、抬笔和保存流程。
        completed = subprocess.run(
            ["tmux", "send-keys", "-t", f"{session}:session", "x", "Enter"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print(f"自动发送停采失败: {(completed.stderr or '').strip()}")
            return
        subprocess.run(
            ["tmux", "select-window", "-t", f"{session}:session"],
            check=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W/S 实时控制机械臂基座 X 轴 Delta 位姿")
    parser.add_argument("--move-service", default="/mov_jog")
    parser.add_argument("--speedl-service", default="/speedl_s")
    parser.add_argument("--speed-stop-service", default="/speed_stop")
    parser.add_argument("--pose-topic", default="/tool_pos")
    parser.add_argument("--xyz-speed-mps", "--x-speed-mps", dest="xyz_speed_mps", type=float, default=0.01)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--max-delta-m", type=float, default=0.001)
    parser.add_argument("--pose-timeout-sec", type=float, default=0.5)
    parser.add_argument("--key-release-timeout-sec", type=float, default=0.16)
    parser.add_argument("--terminal", action="store_true", help="强制使用当前终端读取按键")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 < args.xyz_speed_mps <= 0.1:
        parser.error("--xyz-speed-mps 必须在 (0, 0.1] m/s")
    if not 1.0 <= args.control_hz <= 100.0:
        parser.error("--control-hz 必须在 [1, 100] Hz")
    if not 0.0 < args.max_delta_m <= 0.01:
        parser.error("--max-delta-m 必须在 (0, 0.01] m")
    if not 0.05 <= args.key_release_timeout_sec <= 1.0:
        parser.error("--key-release-timeout-sec 必须在 [0.05, 1.0] s")
    if args.x_min is not None and args.x_max is not None and args.x_min >= args.x_max:
        parser.error("--x-min 必须小于 --x-max")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
    rclpy.init()
    node = DeltaTeleopNode(args, events)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="ros-executor", daemon=True)
    spin_thread.start()

    try:
        if args.terminal or not os.environ.get("DISPLAY"):
            TerminalTeleop(node, args).run()
        else:
            TeleopWindow(node, args).run()
    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
