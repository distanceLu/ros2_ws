#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Episode 级毛笔轨迹评分（奖励函数）。

设计目标
--------
基于机械臂 TCP 轨迹 (tool_pose.csv) 给出 episode reward。
作为图像 reward (ink_reward.py) 的稳健补充与交叉验证：
- 不受机械臂遮挡、光照、纸面背景干扰；
- 直接反映"画得多直、多长、多平滑"；
- 对现有轨迹数据立即可用，无需额外采集。

输入
----
- tool_pose_csv : 路径，列为 timestamp,x,y,z,rx,ry,rz
  或直接传入 np.ndarray shape (N, 3) 的 (x,y,z) 序列（单位 m）。

输出
----
dict (ScoreResult):
    valid, reason, n_frames,
    span_main_mm, span_lateral_mm, length_mm,
    straightness, length_score, smoothness, direction_score,
    score, reward

方法
----
1. 读取 tool_pose.csv，按 timestamp 排序。
2. 识别"画线阶段"：z 值较低的帧（arm 落到纸面附近）。
   阈值取 z 的分位数：z <= DRAW_Z_QUANTILE 分位数视为画线。
3. 主方向识别：画线阶段 (x,y) 跨度较大的轴为主方向，另一轴为侧向。
   - 竖线 episode：主方向跨度远大于侧向。
   - 横线 episode：同理自动适配。
4. 指标：
   - span_main_mm      主方向跨度 (mm)
   - span_lateral_mm   侧向跨度 (mm)
   - length_mm         轨迹累积长度 (mm)
   - straightness      = 1 - lateral_span / max(length, 1)  ∈ [0,1]，越直越接近 1
   - length_score      = min(1, span_main / TARGET_LENGTH_MM)
   - smoothness        = 1 / (1 + median_jerk / JERK_SCALE)  ∈ (0,1)
   - direction_score   = lateral_span / max(span_main, 1) 的反向 = 1 - min(1, lateral/main)
                        （与 straightness 类似但用 span 比率，更鲁棒）
5. 综合 score = 100 * (straightness * length_score * smoothness)
   reward = score / 100，可直接供 RL 使用。

参数 TARGET_LENGTH_MM 需要根据实际任务调整（默认 100 mm，对应竖线练习长度）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ----------------------------- 可调参数 -----------------------------

TARGET_LENGTH_MM = 100.0     # 目标笔画长度，length_score 满分基准
DRAW_Z_QUANTILE = 0.5        # z <= 该分位数视为画线阶段（0.5 = 中位数以下）
JERK_SCALE = 5000.0          # smoothness 的 jerk 归一化尺度（中位 jerk 越小越平滑）
MIN_DRAW_FRAMES = 5          # 画线阶段最少帧数，否则判定无效
MIN_SPAN_MM = 5.0            # 主方向最小跨度，否则判定无效（没画）

# 评分组合权重
W_STRAIGHTNESS = 0.5
W_LENGTH = 0.3
W_SMOOTHNESS = 0.2


# ----------------------------- 数据结构 -----------------------------


@dataclass
class ScoreResult:
    valid: bool
    reason: str = ""
    n_frames: int = 0
    n_draw_frames: int = 0
    span_main_mm: float = 0.0
    span_lateral_mm: float = 0.0
    length_mm: float = 0.0
    straightness: float = 0.0
    length_score: float = 0.0
    smoothness: float = 0.0
    direction_score: float = 0.0
    main_axis: str = ""       # "x" or "y"
    score: float = 0.0
    reward: float = 0.0
    draw_phase: Optional[dict] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "n_frames": self.n_frames,
            "n_draw_frames": self.n_draw_frames,
            "span_main_mm": round(self.span_main_mm, 2),
            "span_lateral_mm": round(self.span_lateral_mm, 2),
            "length_mm": round(self.length_mm, 2),
            "straightness": round(self.straightness, 4),
            "length_score": round(self.length_score, 4),
            "smoothness": round(self.smoothness, 4),
            "direction_score": round(self.direction_score, 4),
            "main_axis": self.main_axis,
            "score": round(self.score, 2),
            "reward": round(self.reward, 4),
            "draw_phase": self.draw_phase,
        }


# ----------------------------- 内部工具 -----------------------------


def _load_tool_pose(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 (ts_us, xyz)，shape (N,) 和 (N,3)。"""
    ts, xs, ys, zs = [], [], [], []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ts.append(int(row["timestamp"]))
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
                zs.append(float(row["z"]))
            except (KeyError, ValueError):
                continue
    if not ts:
        raise ValueError(f"空或格式错误的 tool_pose.csv: {csv_path}")
    ts_arr = np.array(ts, dtype=np.int64)
    xyz = np.stack([xs, ys, zs], axis=1).astype(np.float64)
    order = np.argsort(ts_arr)
    return ts_arr[order], xyz[order]


def _select_draw_phase(ts_us: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """返回布尔 mask，标记画线阶段帧。基于 z 分位数。"""
    z = xyz[:, 2]
    thr = np.quantile(z, DRAW_Z_QUANTILE)
    return z <= thr


def _compute_metrics(ts_us: np.ndarray, xyz_draw: np.ndarray) -> dict:
    x = xyz_draw[:, 0]
    y = xyz_draw[:, 1]
    t_sec = (ts_us - ts_us[0]).astype(np.float64) / 1e6

    span_x_mm = (x.max() - x.min()) * 1000.0
    span_y_mm = (y.max() - y.min()) * 1000.0

    if span_x_mm >= span_y_mm:
        main_axis = "x"
        span_main_mm = span_x_mm
        span_lateral_mm = span_y_mm
        main = x
        lateral = y
    else:
        main_axis = "y"
        span_main_mm = span_y_mm
        span_lateral_mm = span_x_mm
        main = y
        lateral = x

    dx = np.diff(xyz_draw[:, 0])
    dy = np.diff(xyz_draw[:, 1])
    dz = np.diff(xyz_draw[:, 2])
    length_mm = float(np.sum(np.sqrt(dx * dx + dy * dy + dz * dz)) * 1000.0)

    straightness = float(max(0.0, 1.0 - span_lateral_mm / max(length_mm, 1.0)))
    length_score = float(min(1.0, span_main_mm / TARGET_LENGTH_MM))
    direction_score = float(max(0.0, 1.0 - min(1.0, span_lateral_mm / max(span_main_mm, 1.0))))

    # smoothness via jerk magnitude (only xy, 投影到纸面)
    if len(t_sec) >= 5 and (t_sec[-1] - t_sec[0]) > 0:
        # 等距重采样避免采样不均导致 jerk 爆炸
        n = len(t_sec)
        u = np.linspace(0.0, 1.0, n)
        # 对时间归一化插值，得到等间距序列
        t_uniform = np.linspace(t_sec[0], t_sec[-1], max(n, 50))
        x_u = np.interp(t_uniform, t_sec, x)
        y_u = np.interp(t_uniform, t_sec, y)
        dt = np.diff(t_uniform)
        dt[dt == 0] = 1e-6
        vx = np.gradient(x_u, t_uniform)
        vy = np.gradient(y_u, t_uniform)
        ax = np.gradient(vx, t_uniform)
        ay = np.gradient(vy, t_uniform)
        jx = np.gradient(ax, t_uniform)
        jy = np.gradient(ay, t_uniform)
        jerk_mag = np.sqrt(jx * jx + jy * jy)
        median_jerk = float(np.median(jerk_mag))
    else:
        median_jerk = 0.0
    smoothness = float(1.0 / (1.0 + median_jerk / JERK_SCALE))

    return {
        "span_main_mm": span_main_mm,
        "span_lateral_mm": span_lateral_mm,
        "length_mm": length_mm,
        "straightness": straightness,
        "length_score": length_score,
        "smoothness": smoothness,
        "direction_score": direction_score,
        "main_axis": main_axis,
        "median_jerk": median_jerk,
    }


# ----------------------------- 主接口 -----------------------------


def score_trajectory_from_csv(csv_path: str | Path) -> ScoreResult:
    """从 tool_pose.csv 评分一个 episode。"""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return ScoreResult(valid=False, reason=f"tool_pose.csv 不存在: {csv_path}")
    try:
        ts, xyz = _load_tool_pose(csv_path)
    except Exception as exc:
        return ScoreResult(valid=False, reason=f"读取 tool_pose.csv 失败: {exc}")

    return score_trajectory(ts, xyz)


def score_trajectory(ts_us: np.ndarray, xyz: np.ndarray) -> ScoreResult:
    """从时间戳和 xyz 位置序列评分一个 episode。

    ts_us: (N,) 微秒时间戳
    xyz:   (N, 3) 米为单位
    """
    if ts_us is None or xyz is None or len(ts_us) < MIN_DRAW_FRAMES:
        return ScoreResult(valid=False, reason=f"轨迹点过少 (< {MIN_DRAW_FRAMES})")
    if xyz.shape[1] < 3:
        return ScoreResult(valid=False, reason="xyz 至少需要 3 列")

    draw_mask = _select_draw_phase(ts_us, xyz)
    n_draw = int(draw_mask.sum())
    if n_draw < MIN_DRAW_FRAMES:
        return ScoreResult(
            valid=False,
            reason=f"画线阶段帧过少 ({n_draw} < {MIN_DRAW_FRAMES})",
            n_frames=int(len(ts_us)),
            n_draw_frames=n_draw,
        )

    ts_draw = ts_us[draw_mask]
    xyz_draw = xyz[draw_mask]
    metrics = _compute_metrics(ts_draw, xyz_draw)

    if metrics["span_main_mm"] < MIN_SPAN_MM:
        return ScoreResult(
            valid=False,
            reason=f"主方向跨度过小 ({metrics['span_main_mm']:.2f} < {MIN_SPAN_MM} mm)，可能没画线",
            n_frames=int(len(ts_us)),
            n_draw_frames=n_draw,
            span_main_mm=metrics["span_main_mm"],
            span_lateral_mm=metrics["span_lateral_mm"],
            length_mm=metrics["length_mm"],
            main_axis=metrics["main_axis"],
        )

    score = 100.0 * (
        W_STRAIGHTNESS * metrics["straightness"]
        + W_LENGTH * metrics["length_score"]
        + W_SMOOTHNESS * metrics["smoothness"]
    )
    score = float(max(0.0, min(100.0, score)))
    reward = score / 100.0

    draw_phase = {
        "ts_start_us": int(ts_draw[0]),
        "ts_end_us": int(ts_draw[-1]),
        "duration_sec": float((ts_draw[-1] - ts_draw[0]) / 1e6),
    }

    return ScoreResult(
        valid=True,
        n_frames=int(len(ts_us)),
        n_draw_frames=n_draw,
        span_main_mm=metrics["span_main_mm"],
        span_lateral_mm=metrics["span_lateral_mm"],
        length_mm=metrics["length_mm"],
        straightness=metrics["straightness"],
        length_score=metrics["length_score"],
        smoothness=metrics["smoothness"],
        direction_score=metrics["direction_score"],
        main_axis=metrics["main_axis"],
        score=score,
        reward=reward,
        draw_phase=draw_phase,
    )


# ----------------------------- CLI 自检 -----------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(description="毛笔轨迹 episode 评分自检")
    parser.add_argument("--tool-pose", required=True, help="tool_pose.csv 路径")
    parser.add_argument("--out-dir", default="", help="输出目录（存 trajectory_reward.json）")
    args = parser.parse_args()

    result = score_trajectory_from_csv(args.tool_pose)
    print("=== trajectory_reward score ===")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trajectory_reward.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"saved: {out_dir / 'trajectory_reward.json'}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    sys.exit(_cli())
