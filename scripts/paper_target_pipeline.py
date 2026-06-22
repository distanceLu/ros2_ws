#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end: capture/localize paper, then detect vertical-stroke targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_aruco_localize import capture_usb_frame, localize_paper, save_outputs
from paper_target_detect import detect_targets_in_image, draw_detection, _gray_image, _preprocess, detect_frames
from paper_target_shapes import PaperTemplateSpec, default_demo_spec, save_template_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize paper and detect stroke targets.")
    parser.add_argument("--image", default="", help="Use existing image instead of camera capture")
    parser.add_argument("--template-dir", default=str(SCRIPT_DIR.parent / "paper_template"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR.parent / "paper_target_output"))
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--rotate-180", action="store_true", default=True)
    parser.add_argument("--no-rotate-180", action="store_false", dest="rotate_180")
    parser.add_argument("--generate-template", action="store_true", help="Regenerate template/masks before detect")
    parser.add_argument("--min-score", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template_dir = Path(args.template_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = template_dir / "template_config.json"
    if args.generate_template or not config_path.is_file():
        save_template_assets(default_demo_spec(), template_dir)
        print(f"template generated: {template_dir}")

    spec = PaperTemplateSpec.from_dict(json.loads(config_path.read_text(encoding="utf-8")))

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"Cannot read image: {args.image}")
    else:
        bgr = capture_usb_frame(args.device, args.width, args.height, args.warmup_frames, args.rotate_180)

    found, pose = localize_paper(bgr)
    aruco_paths = save_outputs(out_dir / "aruco", bgr, found, pose)
    rectified = pose.rectified_bgr
    if rectified is None:
        raise SystemExit("ArUco localization failed; cannot rectify paper")

    rectified_path = out_dir / "rectified.jpg"
    cv2.imwrite(str(rectified_path), rectified)

    detections = detect_targets_in_image(rectified, spec=spec, min_score=args.min_score)
    frames = detect_frames(_preprocess(_gray_image(rectified)))
    overlay = draw_detection(rectified, detections, frames)
    overlay_path = out_dir / "target_detection_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    result = {
        "aruco": aruco_paths,
        "rectified": str(rectified_path),
        "overlay": str(overlay_path),
        "detections": [det.to_dict() for det in detections],
    }
    result_path = out_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== Paper target pipeline ===")
    print(f"ArUco IDs: {pose.detected_ids}")
    print(f"frames detected: {len(frames)}")
    print(f"targets detected: {len(detections)}")
    for det in detections:
        print(f"  {det.target_id} score={det.score:.3f} center=({det.center_x:.1f},{det.center_y:.1f})")
    print(f"result: {result_path}")
    return 0 if detections else 2


if __name__ == "__main__":
    raise SystemExit(main())
