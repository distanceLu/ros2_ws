#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 data_collect 轨迹帧目录导出为定尺寸视频。

默认把 camera_paper_aruco 下按时间戳排序的原始帧 resize 到 400x320，
写出 **H.264(avc1)** mp4（先经 OpenCV 写中间文件，再用 ffmpeg 转码），
以便 Cursor / VS Code 内置播放器可直接预览。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

FRAME_RE = re.compile(r"^(\d+)\.(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
DEFAULT_CAMERAS = ("camera_paper_aruco",)


def list_frames(image_dir: Path) -> List[Tuple[int, int, Path]]:
    """返回 (timestamp, index, path)，排除 *_400x320 等派生文件。"""
    frames: List[Tuple[int, int, Path]] = []
    for p in image_dir.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        frames.append((int(m.group(1)), int(m.group(2)), p))
    frames.sort(key=lambda x: (x[0], x[1]))
    return frames


def estimate_fps(frames: Sequence[Tuple[int, int, Path]], fallback: float = 20.0) -> float:
    """用相邻帧时间戳估计 fps；时间戳单位按微秒处理。"""
    if len(frames) < 2:
        return fallback
    dts = [frames[i + 1][0] - frames[i][0] for i in range(len(frames) - 1)]
    dts = [d for d in dts if d > 0]
    if not dts:
        return fallback
    dts_sorted = sorted(dts)
    med = dts_sorted[len(dts_sorted) // 2]
    # 常见采集时间戳为微秒；若间隔过小则按纳秒再试
    if med >= 1000:
        fps = 1e6 / med
    else:
        fps = 1e9 / max(med, 1)
    if not np.isfinite(fps) or fps < 1.0 or fps > 120.0:
        return fallback
    return float(fps)


def resize_frame(
    bgr: np.ndarray,
    width: int,
    height: int,
    keep_aspect: bool,
) -> np.ndarray:
    if not keep_aspect:
        return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

    h, w = bgr.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def reencode_h264(src: Path, dst: Path) -> None:
    """用 ffmpeg 转成 Cursor/浏览器可播的 H.264（avc1）。"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，无法转码为 H.264")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(
            f"ffmpeg H.264 转码失败 (code={proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-800:]}"
        )


def write_video(
    frames: Sequence[Tuple[int, int, Path]],
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    keep_aspect: bool,
    h264: bool = True,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV 直接写 H.264 在不少环境不稳定；先写 mp4v，再 ffmpeg→H.264。
    # Cursor / VS Code 内置播放器需要 H.264(avc1)，纯 mp4v(mpeg4) 会报加载失败。
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path: Optional[Path] = None
    write_target = out_path
    if h264:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{out_path.stem}_",
            suffix=".mp4v.mp4",
            dir=str(out_path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        write_target = tmp_path

    writer = cv2.VideoWriter(str(write_target), fourcc, fps, (width, height))
    if not writer.isOpened():
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"无法创建视频写入器: {write_target}")

    written = 0
    try:
        for _, _, path in frames:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[warn] 读图失败，跳过: {path}", file=sys.stderr)
                continue
            frame = resize_frame(img, width, height, keep_aspect)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    try:
        if h264:
            assert tmp_path is not None
            if written == 0:
                tmp_path.unlink(missing_ok=True)
                return 0
            reencode_h264(tmp_path, out_path)
        return written
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def resolve_image_dirs(input_path: Path, camera_names: Sequence[str]) -> List[Path]:
    """支持：相机目录 / session 目录 / 日期目录（含多个 session）。"""
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # 直接是相机帧目录
    if list_frames(input_path):
        return [input_path]

    # session：含指定相机子目录
    cams = [input_path / name for name in camera_names if (input_path / name).is_dir()]
    if cams:
        return cams

    # 日期根：遍历 HH-MM-SS session
    found: List[Path] = []
    for session in sorted(p for p in input_path.iterdir() if p.is_dir()):
        for name in camera_names:
            cam = session / name
            if cam.is_dir() and list_frames(cam):
                found.append(cam)
    if found:
        return found

    raise FileNotFoundError(
        f"在 {input_path} 下未找到可导出的帧目录 "
        f"(期望相机名: {', '.join(camera_names)})"
    )


def default_out_path(image_dir: Path, width: int, height: int) -> Path:
    # .../YYYY-MM-DD/HH-MM-SS/camera_xxx -> 写到 session 旁或相机目录旁
    return image_dir / f"{image_dir.name}_{width}x{height}.mp4"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="轨迹帧目录 -> 定尺寸 mp4 视频")
    p.add_argument(
        "input",
        type=Path,
        help="相机帧目录、session 目录，或日期目录（批量）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 mp4 路径；批量时忽略，按每个相机目录自动命名",
    )
    p.add_argument("--width", type=int, default=400)
    p.add_argument("--height", type=int, default=320)
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="输出帧率；默认按时间戳估计，估不准则用 20",
    )
    p.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        default=None,
        help="session/日期模式下要导出的相机子目录名，可重复；默认 camera_paper_aruco",
    )
    p.add_argument(
        "--keep-aspect",
        action="store_true",
        help="保持比例并 letterbox 到目标尺寸（默认直接拉伸）",
    )
    p.add_argument(
        "--no-h264",
        action="store_true",
        help="跳过 ffmpeg H.264 转码，直接保留 OpenCV mp4v（Cursor 可能无法预览）",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    cameras = tuple(args.cameras) if args.cameras else DEFAULT_CAMERAS
    image_dirs = resolve_image_dirs(args.input, cameras)

    if args.output is not None and len(image_dirs) > 1:
        print("[error] 批量导出时请勿指定单一 --output", file=sys.stderr)
        return 2

    total = 0
    for image_dir in image_dirs:
        frames = list_frames(image_dir)
        if not frames:
            print(f"[skip] 无有效帧: {image_dir}")
            continue

        fps = args.fps if args.fps is not None else estimate_fps(frames)
        out_path = args.output if args.output is not None else default_out_path(
            image_dir, args.width, args.height
        )

        n = write_video(
            frames,
            out_path,
            width=args.width,
            height=args.height,
            fps=fps,
            keep_aspect=args.keep_aspect,
            h264=not args.no_h264,
        )
        total += n
        codec = "h264" if not args.no_h264 else "mp4v"
        print(
            f"[ok] {image_dir} -> {out_path}  "
            f"frames={n}/{len(frames)}  size={args.width}x{args.height}  "
            f"fps={fps:.2f}  codec={codec}"
        )

    if total == 0:
        print("[error] 没有写出任何帧", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
