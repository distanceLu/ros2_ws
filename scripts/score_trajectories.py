#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量评分现有轨迹数据，验证 reward 是否合理。

对一个 data_collect/YYYY-MM-DD/ 下的所有 session：
1. 轨迹 reward：读 robot_state/tool_pose.csv，调 trajectory_reward.score_trajectory_from_csv。
2. 图像 reward：自动选 before/after 纸面图，调 ink_reward.score_episode。
   - before = 该 session 纸面图按时间排序的第一张（episode 开始，arm 在初始位）
   - after  = 该 session 纸面图按时间排序的最后一张（episode 结束）
3. 写入 session/reward.json + session/reward_vis.png（图像可视化）。
4. 汇总输出 reward_summary.csv，列出每个 session 的两种 reward。

用法
----
    python3 scripts/score_trajectories.py --data-root data_collect/2026-06-25
    python3 scripts/score_trajectories.py --data-root data_collect/2026-06-25 --skip-image   # 只跑轨迹
    python3 scripts/score_trajectories.py --data-root data_collect/2026-06-25 --session 15-50-43

参数
----
--data-root    : data_collect 下的日期目录
--session      : 只评指定的 session（HH-MM-SS），可多次指定
--skip-image   : 跳过图像评分（图像评分较慢且依赖 before/after 选帧质量）
--max-sessions : 最多评多少 session（调试用）
--out-csv      : 汇总 CSV 输出路径，默认 <data-root>/reward_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ink_reward import score_episode as score_image_episode
from trajectory_reward import score_trajectory_from_csv


def _list_sessions(data_root: Path, allow: set[str] | None) -> list[Path]:
    if not data_root.is_dir():
        return []
    sessions = sorted(p for p in data_root.iterdir() if p.is_dir() and "-" in p.name)
    if allow:
        sessions = [p for p in sessions if p.name in allow]
    return sessions


def _paper_image_paths(session: Path) -> list[Path]:
    paper_dir = session / "camera_paper_aruco"
    if not paper_dir.is_dir():
        return []
    imgs = sorted(paper_dir.glob("*.jpg"), key=lambda p: p.name)
    return imgs


def _pick_before_after(imgs: list[Path]) -> tuple[Optional[Path], Optional[Path]]:
    if not imgs:
        return None, None
    return imgs[0], imgs[-1]


def _score_one_session(
    session: Path,
    do_image: bool,
) -> dict:
    record = {
        "session": session.name,
        "trajectory_valid": False,
        "trajectory_score": 0.0,
        "trajectory_reward": 0.0,
        "trajectory_reason": "",
        "image_valid": False,
        "image_mode": "",
        "image_score": 0.0,
        "image_reward": 0.0,
        "image_reason": "",
        "image_target_id": "",
    }

    # 轨迹评分
    tool_pose = session / "robot_state" / "tool_pose.csv"
    if tool_pose.is_file():
        try:
            t_res = score_trajectory_from_csv(tool_pose)
            record.update({
                "trajectory_valid": t_res.valid,
                "trajectory_score": round(t_res.score, 2),
                "trajectory_reward": round(t_res.reward, 4),
                "trajectory_reason": t_res.reason,
            })
            (session / "trajectory_reward.json").write_text(
                json.dumps(t_res.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            record["trajectory_reason"] = f"trajectory_score exception: {exc}"
    else:
        record["trajectory_reason"] = "tool_pose.csv missing"

    # 图像评分
    if do_image:
        imgs = _paper_image_paths(session)
        before_p, after_p = _pick_before_after(imgs)
        if before_p is None or after_p is None:
            record["image_reason"] = "no paper images"
        else:
            try:
                before = cv2.imread(str(before_p))
                after = cv2.imread(str(after_p))
                i_res = score_image_episode(before, after, generate_vis=True)
                record.update({
                    "image_valid": i_res.valid,
                    "image_mode": i_res.mode,
                    "image_score": round(i_res.score, 2),
                    "image_reward": round(i_res.reward, 4),
                    "image_reason": i_res.reason,
                    "image_target_id": i_res.target_id,
                })
                payload = i_res.to_dict()
                payload["before_image"] = str(before_p)
                payload["after_image"] = str(after_p)
                (session / "image_reward.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                if i_res.vis is not None:
                    cv2.imwrite(str(session / "reward_vis.png"), i_res.vis)
            except Exception as exc:
                record["image_reason"] = f"image_score exception: {exc}"

    return record


def _write_summary_csv(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="批量评分轨迹数据")
    parser.add_argument("--data-root", required=True, help="data_collect/YYYY-MM-DD 目录")
    parser.add_argument("--session", action="append", default=[], help="只评指定 session (HH-MM-SS)")
    parser.add_argument("--skip-image", action="store_true", help="跳过图像评分")
    parser.add_argument("--max-sessions", type=int, default=0, help="最多评多少 session（0=全部）")
    parser.add_argument("--out-csv", default="", help="汇总 CSV 路径")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"data-root 不存在: {data_root}")
        return 1

    allow = set(args.session) if args.session else None
    sessions = _list_sessions(data_root, allow)
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]

    if not sessions:
        print(f"未找到 session: {data_root}")
        return 1

    out_csv = Path(args.out_csv) if args.out_csv else data_root / "reward_summary.csv"

    print(f"开始评分 {len(sessions)} 个 session（图像评分={'关' if args.skip_image else '开'}）")
    rows: list[dict] = []
    t0 = time.monotonic()
    for i, session in enumerate(sessions, 1):
        rec = _score_one_session(session, do_image=not args.skip_image)
        rows.append(rec)
        status = (
            f"[{i}/{len(sessions)}] {session.name} "
            f"traj={'OK' if rec['trajectory_valid'] else 'INVALID'} "
            f"score={rec['trajectory_score']} "
        )
        if not args.skip_image:
            status += (
                f"img={'OK' if rec['image_valid'] else 'INVALID'} "
                f"score={rec['image_score']} "
            )
        print(status)

    _write_summary_csv(rows, out_csv)
    elapsed = time.monotonic() - t0
    print(f"\n完成，耗时 {elapsed:.1f}s，汇总: {out_csv}")

    # 简单统计
    valid_traj = [r for r in rows if r["trajectory_valid"]]
    valid_img = [r for r in rows if r["image_valid"]]
    if valid_traj:
        ts = [r["trajectory_score"] for r in valid_traj]
        print(
            f"轨迹 reward: valid {len(valid_traj)}/{len(rows)}, "
            f"score min={min(ts):.1f} max={max(ts):.1f} mean={sum(ts)/len(ts):.1f}"
        )
    if valid_img:
        iss = [r["image_score"] for r in valid_img]
        print(
            f"图像 reward: valid {len(valid_img)}/{len(rows)}, "
            f"score min={min(iss):.1f} max={max(iss):.1f} mean={sum(iss)/len(iss):.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
