#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 data_collect 轨迹帧 resize 为定尺寸 jpg，不覆盖原图。

默认输出旁路文件：``{原名stem}_{W}x{H}.jpg``
例如 ``60487812489.000001_400x320.jpg``。

用法
----
    # 某天全部纸面相机
    python3 scripts/frames_to_resized_images.py \\
        /home/shugen/yanjie/ros2_ws/data_collect/2026-07-21

    # 单个 session / 相机目录
    python3 scripts/frames_to_resized_images.py \\
        /home/shugen/yanjie/ros2_ws/data_collect/2026-07-21/16-48-04/camera_paper_aruco
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

FRAME_RE = re.compile(r"^(\d+)\.(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
DEFAULT_CAMERAS = ("camera_paper_aruco",)


def list_frames(image_dir: Path) -> List[Tuple[int, int, Path]]:
    """返回 (timestamp, index, path)，排除 *_WxH 等派生文件。"""
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


def out_path_for(src: Path, width: int, height: int) -> Path:
    return src.with_name(f"{src.stem}_{width}x{height}.jpg")


def convert_dir(
    image_dir: Path,
    width: int,
    height: int,
    keep_aspect: bool,
    skip_existing: bool,
    jpeg_quality: int,
) -> Tuple[int, int, int]:
    """返回 (written, skipped_existing, failed)。"""
    frames = list_frames(image_dir)
    written = skipped = failed = 0
    for _, _, src in frames:
        dst = out_path_for(src, width, height)
        if skip_existing and dst.is_file():
            skipped += 1
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] 读图失败，跳过: {src}", file=sys.stderr)
            failed += 1
            continue
        frame = resize_frame(img, width, height, keep_aspect)
        ok = cv2.imwrite(
            str(dst),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            print(f"[warn] 写图失败: {dst}", file=sys.stderr)
            failed += 1
            continue
        written += 1
    return written, skipped, failed


def resolve_image_dirs(input_path: Path, camera_names: Sequence[str]) -> List[Path]:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if list_frames(input_path):
        return [input_path]

    cams = [input_path / name for name in camera_names if (input_path / name).is_dir()]
    if cams:
        return cams

    found: List[Path] = []
    for session in sorted(p for p in input_path.iterdir() if p.is_dir()):
        for name in camera_names:
            cam = session / name
            if cam.is_dir() and list_frames(cam):
                found.append(cam)
    if found:
        return found

    raise FileNotFoundError(
        f"在 {input_path} 下未找到可转换的帧目录 "
        f"(期望相机名: {', '.join(camera_names)})"
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="轨迹帧 -> 定尺寸 jpg（不覆盖原图）")
    p.add_argument(
        "input",
        type=Path,
        help="相机帧目录、session 目录，或日期目录（批量）",
    )
    p.add_argument("--width", type=int, default=400)
    p.add_argument("--height", type=int, default=320)
    p.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        default=None,
        help="session/日期模式下相机子目录名，可重复；默认 camera_paper_aruco",
    )
    p.add_argument(
        "--keep-aspect",
        action="store_true",
        help="保持比例并 letterbox（默认直接拉伸到目标尺寸）",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="即使目标已存在也重新写出",
    )
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    cameras = tuple(args.cameras) if args.cameras else DEFAULT_CAMERAS
    image_dirs = resolve_image_dirs(args.input, cameras)

    total_w = total_s = total_f = 0
    for image_dir in image_dirs:
        w, s, f = convert_dir(
            image_dir,
            width=args.width,
            height=args.height,
            keep_aspect=args.keep_aspect,
            skip_existing=not args.no_skip_existing,
            jpeg_quality=args.jpeg_quality,
        )
        total_w += w
        total_s += s
        total_f += f
        n_src = len(list_frames(image_dir))
        print(
            f"[ok] {image_dir}  src={n_src}  "
            f"wrote={w}  skipped={s}  failed={f}  "
            f"size={args.width}x{args.height}"
        )

    print(
        f"[done] wrote={total_w} skipped={total_s} failed={total_f} "
        f"dirs={len(image_dirs)}"
    )
    if total_w == 0 and total_s == 0:
        print("[error] 没有写出任何帧", file=sys.stderr)
        return 1
    if total_f > 0 and total_w == 0 and total_s == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
