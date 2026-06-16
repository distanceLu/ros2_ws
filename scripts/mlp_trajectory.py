#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlp_trajectory.py - 用单条末端轨迹训练一个简易 MLP，并可推理/执行轨迹。

典型用法：
  python3 scripts/mlp_trajectory.py train \
    --data-dir data_collect/data_collect/2026-06-11/20-28-41 \
    --model-out models/brush_trajectory_mlp_index.npz

  python3 scripts/mlp_trajectory.py infer \
    --model models/brush_trajectory_mlp_index.npz \
    --csv-out models/brush_trajectory_pred_index.csv

  # 默认 dry-run，只打印将要发送的位姿；确认安全后才加 --execute
  python3 scripts/mlp_trajectory.py execute \
    --model models/brush_trajectory_mlp_index.npz --steps 120

  python3 scripts/mlp_trajectory.py execute \
    --model models/brush_trajectory_mlp_index.npz --steps 120 --execute
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
POSE_COLUMNS = ("x", "y", "z", "rx", "ry", "rz")


def _prepend_env_path(var_name: str, value: str) -> None:
    current = os.environ.get(var_name, "")
    items = [item for item in current.split(os.pathsep) if item]
    if value in items:
        return
    os.environ[var_name] = value if not current else value + os.pathsep + current


def bootstrap_local_ros_paths() -> None:
    """让直接 python3 运行脚本时也能找到本工作区 install 下的 ROS2 Python 包。"""
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    install_root = WORKSPACE_ROOT / "install"
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


def load_tool_pose_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, list[float]]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in ("timestamp", *POSE_COLUMNS) if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} 缺少列: {missing}")
        for row in reader:
            rows.append((float(row["timestamp"]), [float(row[col]) for col in POSE_COLUMNS]))

    if len(rows) < 8:
        raise ValueError(f"轨迹点太少，无法训练: {csv_path}")

    rows.sort(key=lambda item: item[0])
    timestamps = np.asarray([item[0] for item in rows], dtype=np.float64)
    poses = np.asarray([item[1] for item in rows], dtype=np.float64)
    return timestamps, poses


def cumulative_xyz_distance(poses: np.ndarray) -> np.ndarray:
    step = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def trim_static_edges(
    timestamps: np.ndarray,
    poses: np.ndarray,
    trim_eps_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    progress = cumulative_xyz_distance(poses)
    total = float(progress[-1])
    if trim_eps_m <= 0.0 or total <= 2.0 * trim_eps_m:
        return timestamps, poses, {"start_index": 0, "end_index": len(poses) - 1, "path_length_m": total}

    start_idx = int(np.searchsorted(progress, trim_eps_m, side="left"))
    end_idx = int(np.searchsorted(progress, total - trim_eps_m, side="right"))
    end_idx = max(start_idx + 2, min(end_idx, len(poses)))
    trimmed_t = timestamps[start_idx:end_idx]
    trimmed_p = poses[start_idx:end_idx]
    return trimmed_t, trimmed_p, {
        "start_index": start_idx,
        "end_index": end_idx - 1,
        "path_length_m": total,
        "trim_eps_m": trim_eps_m,
    }


def make_training_arrays(
    timestamps: np.ndarray,
    poses: np.ndarray,
    input_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start_pose = poses[0].copy()
    offsets = poses - start_pose
    scale = np.maximum(np.ptp(offsets, axis=0), 1e-6)
    y = offsets / scale

    if input_mode == "time":
        t = timestamps - timestamps[0]
        denom = float(t[-1]) if float(t[-1]) > 0 else 1.0
        x = t / denom
    elif input_mode == "index":
        x = np.linspace(0.0, 1.0, len(poses), dtype=np.float64)
    elif input_mode == "progress":
        progress = cumulative_xyz_distance(poses)
        denom = float(progress[-1]) if float(progress[-1]) > 0 else 1.0
        x = progress / denom
    else:
        raise ValueError(f"未知 input_mode: {input_mode}")

    meta = {
        "pose_columns": list(POSE_COLUMNS),
        "input_mode": input_mode,
        "start_pose": start_pose.tolist(),
        "output_scale": scale.tolist(),
        "end_delta": offsets[-1].tolist(),
        "num_samples": int(len(poses)),
    }
    return x.reshape(-1, 1), y, meta


def init_params(hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def weight(fan_in: int, fan_out: int) -> np.ndarray:
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)

    return {
        "w1": weight(1, hidden),
        "b1": np.zeros((hidden,), dtype=np.float64),
        "w2": weight(hidden, hidden),
        "b2": np.zeros((hidden,), dtype=np.float64),
        "w3": weight(hidden, 6),
        "b3": np.zeros((6,), dtype=np.float64),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ params["w1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["w2"] + params["b2"]
    a2 = np.tanh(z2)
    y = a2 @ params["w3"] + params["b3"]
    return y, {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    print_every: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    params = init_params(hidden, seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    history: list[dict[str, float]] = []
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n = float(len(x))
    for epoch in range(1, epochs + 1):
        pred, cache = forward(params, x)
        err = pred - y
        loss = float(np.mean(err * err))

        dy = (2.0 / n) * err / y.shape[1]
        grads: dict[str, np.ndarray] = {}
        grads["w3"] = cache["a2"].T @ dy + weight_decay * params["w3"]
        grads["b3"] = dy.sum(axis=0)
        da2 = dy @ params["w3"].T
        dz2 = da2 * (1.0 - np.tanh(cache["z2"]) ** 2)
        grads["w2"] = cache["a1"].T @ dz2 + weight_decay * params["w2"]
        grads["b2"] = dz2.sum(axis=0)
        da1 = dz2 @ params["w2"].T
        dz1 = da1 * (1.0 - np.tanh(cache["z1"]) ** 2)
        grads["w1"] = cache["x"].T @ dz1 + weight_decay * params["w1"]
        grads["b1"] = dz1.sum(axis=0)

        for key in params:
            m[key] = beta1 * m[key] + (1.0 - beta1) * grads[key]
            v[key] = beta2 * v[key] + (1.0 - beta2) * (grads[key] * grads[key])
            m_hat = m[key] / (1.0 - beta1**epoch)
            v_hat = v[key] / (1.0 - beta2**epoch)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch == 1 or epoch == epochs or epoch % print_every == 0:
            rmse_norm = math.sqrt(loss)
            history.append({"epoch": float(epoch), "loss": loss, "rmse_norm": rmse_norm})
            print(f"epoch={epoch:5d} loss={loss:.8f} rmse_norm={rmse_norm:.6f}")

    return params, history


def predict_offsets(model: dict[str, Any], steps: int) -> tuple[np.ndarray, np.ndarray]:
    params = {key: model[key] for key in ("w1", "b1", "w2", "b2", "w3", "b3")}
    s = np.linspace(0.0, 1.0, steps, dtype=np.float64).reshape(-1, 1)
    pred_norm, _ = forward(params, s)
    scale = np.asarray(json.loads(str(model["meta"].item()))["output_scale"], dtype=np.float64)
    offsets = pred_norm * scale
    # 强制首点为 0，终点为采集轨迹的终端偏移，避免网络边界处的微小拟合误差。
    meta = json.loads(str(model["meta"].item()))
    offsets[0] = 0.0
    offsets[-1] = np.asarray(meta["end_delta"], dtype=np.float64)
    return s.reshape(-1), offsets


def save_model(path: Path, params: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(params)
    payload["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez(path, **payload)


def load_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=False))


def write_prediction_csv(path: Path, s: np.ndarray, offsets: np.ndarray, recorded_start: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    abs_pose = recorded_start.reshape(1, 6) + offsets
    headers = (
        ["s"]
        + [f"delta_{col}" for col in POSE_COLUMNS]
        + [f"recorded_abs_{col}" for col in POSE_COLUMNS]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx in range(len(s)):
            writer.writerow([f"{s[idx]:.8f}", *[f"{v:.10f}" for v in offsets[idx]], *[f"{v:.10f}" for v in abs_pose[idx]]])


def pose_dict(values: np.ndarray) -> dict[str, float]:
    return {col: float(values[idx]) for idx, col in enumerate(POSE_COLUMNS)}


def format_pose(values: np.ndarray) -> str:
    pose = pose_dict(values)
    return ", ".join(f"{key}={value:.6f}" for key, value in pose.items())


def cmd_train(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv) if args.csv else Path(args.data_dir) / "robot_state" / "tool_pose.csv"
    timestamps, poses = load_tool_pose_csv(csv_path)
    timestamps, poses, trim_meta = trim_static_edges(timestamps, poses, args.trim_static_m)
    x, y, meta = make_training_arrays(timestamps, poses, args.input_mode)
    meta.update(
        {
            "source_csv": str(csv_path),
            "trim": trim_meta,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    )

    print(f"训练样本: {len(x)}")
    print(f"起点: {format_pose(poses[0])}")
    print(f"终点: {format_pose(poses[-1])}")
    print(f"终端相对偏移: {format_pose(poses[-1] - poses[0])}")

    params, history = train_mlp(
        x=x,
        y=y,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        print_every=args.print_every,
    )
    meta["history"] = history
    save_model(Path(args.model_out), params, meta)
    print(f"模型已保存: {args.model_out}")

    if args.csv_out:
        model = load_model(Path(args.model_out))
        s, offsets = predict_offsets(model, args.infer_steps)
        write_prediction_csv(Path(args.csv_out), s, offsets, poses[0])
        print(f"预测轨迹已保存: {args.csv_out}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    meta = json.loads(str(model["meta"].item()))
    s, offsets = predict_offsets(model, args.steps)
    recorded_start = np.asarray(meta["start_pose"], dtype=np.float64)
    write_prediction_csv(Path(args.csv_out), s, offsets, recorded_start)
    print(f"预测轨迹已保存: {args.csv_out}")
    print(f"终端相对偏移: {format_pose(offsets[-1])}")
    return 0


class TrajectoryExecutor:
    def __init__(self, service_name: str, timeout_sec: float):
        bootstrap_local_ros_paths()
        import rclpy
        from common_interface.msg import TcpPos
        from common_interface.srv import Move
        from rclpy.node import Node

        self.rclpy = rclpy
        self.Move = Move
        self.node = Node("mlp_trajectory_executor")
        self.move_client = self.node.create_client(Move, service_name)
        self.latest_pose: Optional[np.ndarray] = None

        def on_pose(msg: Any) -> None:
            self.latest_pose = np.asarray([msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz], dtype=np.float64)

        self.node.create_subscription(TcpPos, "/tool_pos", on_pose, 10)
        if not self.move_client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"服务不可用: {service_name}")

    def get_current_pose(self, timeout_sec: float) -> np.ndarray:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.latest_pose is not None:
                return self.latest_pose.copy()
        raise RuntimeError("无法获取 /tool_pos 当前位姿")

    def move_absolute(self, pose: np.ndarray, timeout_sec: float) -> bool:
        request = self.Move.Request()
        request.a = float(pose[0])
        request.b = float(pose[1])
        request.c = float(pose[2])
        request.d = float(pose[3])
        request.e = float(pose[4])
        request.f = float(pose[5])
        request.block = True
        request.name = ""
        future = self.move_client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if future.result() is None:
            self.node.get_logger().error(f"移动失败: {future.exception()}")
            return False
        return True

    def shutdown(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def validate_step_limits(poses: np.ndarray, max_step_m: float, max_step_rad: float, allow_large_steps: bool) -> None:
    d = np.diff(poses, axis=0)
    max_xyz = float(np.max(np.linalg.norm(d[:, :3], axis=1))) if len(d) else 0.0
    max_rot = float(np.max(np.linalg.norm(d[:, 3:], axis=1))) if len(d) else 0.0
    print(f"最大单步 xyz={max_xyz * 1000:.2f} mm, rot={max_rot:.5f} rad")
    if allow_large_steps:
        return
    if max_xyz > max_step_m or max_rot > max_step_rad:
        raise RuntimeError(
            "预测轨迹单步过大，已中止。可增加 --steps，或确认安全后加 --allow-large-steps。"
        )


def cmd_execute(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    meta = json.loads(str(model["meta"].item()))
    _, offsets = predict_offsets(model, args.steps)

    executor: Optional[TrajectoryExecutor] = None
    recorded_start = np.asarray(meta["start_pose"], dtype=np.float64)
    if args.recorded_absolute:
        start_pose = recorded_start
        print("执行模式: 使用采集时的绝对位姿")
    else:
        bootstrap_local_ros_paths()
        import rclpy

        if not rclpy.ok():
            rclpy.init()
        executor = TrajectoryExecutor(args.service, args.timeout_sec)
        start_pose = executor.get_current_pose(args.pose_timeout_sec)
        print("执行模式: 将学习到的相对轨迹平移到当前位姿")

    poses = start_pose.reshape(1, 6) + offsets
    validate_step_limits(poses, args.max_step_m, args.max_step_rad, args.allow_large_steps)
    print(f"起点: {format_pose(poses[0])}")
    print(f"终点: {format_pose(poses[-1])}")

    if not args.execute:
        if executor is not None:
            executor.shutdown()
        print("dry-run 完成：未发送运动命令。确认安全后添加 --execute。")
        return 0

    if executor is None:
        bootstrap_local_ros_paths()
        import rclpy

        if not rclpy.ok():
            rclpy.init()
        executor = TrajectoryExecutor(args.service, args.timeout_sec)
    try:
        for idx, pose in enumerate(poses[1:], start=1):
            print(f"[{idx}/{len(poses) - 1}] {format_pose(pose)}")
            if not executor.move_absolute(pose, args.timeout_sec):
                return 1
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
    finally:
        executor.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练/推理/执行单条末端轨迹 MLP")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="从 tool_pose.csv 训练 MLP")
    train.add_argument("--data-dir", default="data_collect/data_collect/2026-06-11/20-28-41")
    train.add_argument("--csv", default="")
    train.add_argument("--model-out", default="models/brush_trajectory_mlp_index.npz")
    train.add_argument("--csv-out", default="models/brush_trajectory_pred_index.csv")
    train.add_argument("--infer-steps", type=int, default=120)
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument("--epochs", type=int, default=8000)
    train.add_argument("--lr", type=float, default=0.003)
    train.add_argument("--weight-decay", type=float, default=1e-6)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--print-every", type=int, default=500)
    train.add_argument("--trim-static-m", type=float, default=0.001)
    train.add_argument("--input-mode", choices=("progress", "time", "index"), default="index")
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="导出模型预测轨迹 CSV")
    infer.add_argument("--model", default="models/brush_trajectory_mlp_index.npz")
    infer.add_argument("--csv-out", default="models/brush_trajectory_pred_index.csv")
    infer.add_argument("--steps", type=int, default=120)
    infer.set_defaults(func=cmd_infer)

    execute = sub.add_parser("execute", help="通过 ROS2 Move 服务执行预测轨迹")
    execute.add_argument("--model", default="models/brush_trajectory_mlp_index.npz")
    execute.add_argument("--steps", type=int, default=120)
    execute.add_argument("--service", default="/mov_jog")
    execute.add_argument("--execute", action="store_true", help="真正发送运动命令；不加时只 dry-run")
    execute.add_argument("--recorded-absolute", action="store_true", help="使用采集时绝对坐标，而不是当前位姿平移")
    execute.add_argument("--timeout-sec", type=float, default=10.0)
    execute.add_argument("--pose-timeout-sec", type=float, default=3.0)
    execute.add_argument("--sleep-sec", type=float, default=0.02)
    execute.add_argument("--max-step-m", type=float, default=0.01)
    execute.add_argument("--max-step-rad", type=float, default=0.05)
    execute.add_argument("--allow-large-steps", action="store_true")
    execute.set_defaults(func=cmd_execute)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
