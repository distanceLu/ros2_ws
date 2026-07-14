#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture from USB camera and localize paper pose via four corner ArUco markers.

Layout (DICT_4X4_50):
  TL=0, TR=1, BR=2, BL=3

Usage:
  python3 scripts/paper_aruco_localize.py
  python3 scripts/paper_aruco_localize.py --image /path/to.jpg --no-capture
  python3 scripts/paper_aruco_localize.py --collect --session-dir /path/to/session
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "paper_aruco_output"
DEFAULT_DATA_COLLECT_ROOT = SCRIPT_DIR.parent / "data_collect"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
CORNER_IDS = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
EXPECTED_IDS = (0, 1, 2, 3)


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray  # shape (4, 2), image pixels
    center: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.center = self.corners.mean(axis=0)


@dataclass
class PaperPose:
    detected_ids: list[int]
    missing_ids: list[int]
    corner_points_image: dict[str, np.ndarray]
    homography_image_to_paper: Optional[np.ndarray]
    paper_size_px: tuple[int, int]
    angle_deg: float
    center_image: np.ndarray
    rectified_bgr: Optional[np.ndarray]


def make_detector_params() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters_create()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.001
    params.maxMarkerPerimeterRate = 10.0
    params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    params.maxErroneousBitsInBorderRate = 0.5
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def detect_markers_on_gray(gray: np.ndarray) -> dict[int, np.ndarray]:
    if gray.size == 0 or gray.shape[0] < 20 or gray.shape[1] < 20:
        return {}
    dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)
    params = make_detector_params()
    found: dict[int, np.ndarray] = {}

    variants: list[tuple[str, np.ndarray, float]] = [("raw", gray, 1.0)]
    for scale in (0.5, 1.5, 2.0):
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((f"scale_{scale}", resized, scale))
    for tag, g in [
        ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)),
        ("blur", cv2.GaussianBlur(gray, (5, 5), 0)),
    ]:
        variants.append((tag, g, 1.0))

    for _, g, scale in variants:
        corners, ids, _ = cv2.aruco.detectMarkers(g, dictionary, parameters=params)
        if ids is None:
            continue
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0] / scale
            area = cv2.contourArea(pts.astype(np.float32))
            key = int(marker_id)
            if key not in found:
                found[key] = pts
            else:
                old_area = cv2.contourArea(found[key].astype(np.float32))
                if area > old_area:
                    found[key] = pts
    return found


def find_marker_blobs_on_white_paper(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white[:, : max(0, int(bgr.shape[1] * 0.15))] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    paper = max(contours, key=cv2.contourArea)
    px, py, pw, ph = cv2.boundingRect(paper)
    paper_mask = np.zeros_like(white)
    cv2.drawContours(paper_mask, [paper], -1, 255, -1)

    inv = cv2.bitwise_and(255 - gray, paper_mask)
    _, th = cv2.threshold(inv, 50, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs: list[tuple[int, int, int, int, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1200 or area > 45000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (0.65 < w / float(h) < 1.35 and 35 < w < 280 and 35 < h < 280):
            continue
        blobs.append((x + w // 2 + px, y + h // 2 + py, w, h, area))
    blobs.sort(key=lambda item: item[4], reverse=True)
    return [(cx, cy, w, h) for cx, cy, w, h, _ in blobs[:12]]


def detect_markers_from_blobs(bgr: np.ndarray, found: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    for cx, cy, w, h in find_marker_blobs_on_white_paper(bgr):
        pad = int(max(w, h) * 1.8)
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(bgr.shape[1], cx + pad)
        y2 = min(bgr.shape[0], cy + pad)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        local = detect_markers_on_gray(gray)
        for marker_id, pts in local.items():
            shifted = pts.copy()
            shifted[:, 0] += x1
            shifted[:, 1] += y1
            area = cv2.contourArea(shifted.astype(np.float32))
            if marker_id not in found or area > cv2.contourArea(found[marker_id].astype(np.float32)):
                found[marker_id] = shifted
    return found


def marker_paper_corner_point(corners: np.ndarray, paper_center: np.ndarray) -> np.ndarray:
    """Pick the marker vertex farthest from the paper center."""
    distances = np.linalg.norm(corners - paper_center[None, :], axis=1)
    return corners[int(np.argmax(distances))]


def fill_missing_markers_from_blobs(
    bgr: np.ndarray,
    found: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    if len(found) >= 4:
        return found

    centers = {marker_id: pts.mean(axis=0) for marker_id, pts in found.items()}
    blobs = find_marker_blobs_on_white_paper(bgr)
    extra_centers: list[np.ndarray] = []
    for cx, cy, w, h in blobs:
        center = np.array([cx, cy], dtype=np.float32)
        if all(np.linalg.norm(center - c) > max(w, h) * 0.8 for c in centers.values()):
            if all(np.linalg.norm(center - e) > max(w, h) * 0.8 for e in extra_centers):
                extra_centers.append(center)

    missing = [marker_id for marker_id in EXPECTED_IDS if marker_id not in found]
    for marker_id, center in zip(missing, extra_centers):
        half = 20.0
        square = np.array(
            [
                [center[0] - half, center[1] - half],
                [center[0] + half, center[1] - half],
                [center[0] + half, center[1] + half],
                [center[0] - half, center[1] + half],
            ],
            dtype=np.float32,
        )
        found[marker_id] = square
    return found


def estimate_paper_pose(found: dict[int, np.ndarray], paper_long_edge_px: int = 2000) -> PaperPose:
    detected_ids = sorted(found)
    missing_ids = [marker_id for marker_id in EXPECTED_IDS if marker_id not in found]

    marker_centers = np.array([found[mid].mean(axis=0) for mid in detected_ids], dtype=np.float32)
    paper_center = marker_centers.mean(axis=0) if len(marker_centers) else np.array([np.nan, np.nan])

    corner_points: dict[str, np.ndarray] = {}
    for marker_id, label in CORNER_IDS.items():
        if marker_id in found:
            corner_points[label] = marker_paper_corner_point(found[marker_id], paper_center)

    # Need at least 3 corners for affine; 4 for homography.
    src_pts: list[np.ndarray] = []
    dst_pts: list[np.ndarray] = []
    aspect = 297.0 / 210.0  # A4 portrait canonical frame
    width = paper_long_edge_px
    height = int(round(width * aspect))
    canonical = {
        "TL": np.array([0.0, 0.0]),
        "TR": np.array([width - 1.0, 0.0]),
        "BR": np.array([width - 1.0, height - 1.0]),
        "BL": np.array([0.0, height - 1.0]),
    }
    for label, dst in canonical.items():
        if label in corner_points:
            src_pts.append(corner_points[label])
            dst_pts.append(dst)

    homography = None
    rectified = None
    angle_deg = float("nan")
    center = np.array([np.nan, np.nan])

    if len(src_pts) >= 4:
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        homography, _ = cv2.findHomography(src, dst, method=0)
    elif len(src_pts) == 3:
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        affine, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC)
        if affine is not None:
            homography = np.vstack([affine, [0.0, 0.0, 1.0]])

    if homography is not None and "TL" in corner_points and "TR" in corner_points:
        top_vec = corner_points["TR"] - corner_points["TL"]
        angle_deg = math.degrees(math.atan2(top_vec[1], top_vec[0]))
        paper_corners = np.array(
            [corner_points[label] for label in ("TL", "TR", "BR", "BL") if label in corner_points],
            dtype=np.float32,
        )
        center = paper_corners.mean(axis=0)

    return PaperPose(
        detected_ids=detected_ids,
        missing_ids=missing_ids,
        corner_points_image=corner_points,
        homography_image_to_paper=homography,
        paper_size_px=(width, height),
        angle_deg=angle_deg,
        center_image=center,
        rectified_bgr=rectified,
    )


def rectify_paper(bgr: np.ndarray, pose: PaperPose) -> Optional[np.ndarray]:
    if pose.homography_image_to_paper is None:
        return None
    width, height = pose.paper_size_px
    return cv2.warpPerspective(bgr, pose.homography_image_to_paper, (width, height))


def draw_result(bgr: np.ndarray, found: dict[int, np.ndarray], pose: PaperPose) -> np.ndarray:
    vis = bgr.copy()
    dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)
    if found:
        corners_list = [np.expand_dims(found[mid], 0) for mid in sorted(found)]
        ids = np.array([[mid] for mid in sorted(found)], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(vis, corners_list, ids)

    for label, pt in pose.corner_points_image.items():
        p = tuple(np.round(pt).astype(int))
        cv2.circle(vis, p, 12, (0, 255, 255), 3)
        cv2.putText(vis, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    if all(label in pose.corner_points_image for label in ("TL", "TR", "BR", "BL")):
        quad = np.array(
            [pose.corner_points_image[label] for label in ("TL", "TR", "BR", "BL")],
            dtype=np.int32,
        )
        cv2.polylines(vis, [quad], True, (255, 0, 0), 3)

    if not np.isnan(pose.center_image[0]):
        c = tuple(np.round(pose.center_image).astype(int))
        cv2.drawMarker(vis, c, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.putText(
            vis,
            f"angle={pose.angle_deg:.1f}deg",
            (c[0] + 20, c[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    status = f"detected IDs: {pose.detected_ids}; missing: {pose.missing_ids}"
    cv2.putText(vis, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return vis


def capture_usb_frame(
    device: str,
    width: int,
    height: int,
    warmup_frames: int,
    rotate_180: bool,
) -> np.ndarray:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame = None
    for _ in range(warmup_frames):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("Failed to read frame from camera")
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def elapsed_microseconds(save_date: str) -> int:
    current = datetime.now()
    target = datetime.strptime(save_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    return int((current - target).total_seconds() * 1_000_000)


def localize_paper(bgr: np.ndarray) -> tuple[dict[int, np.ndarray], PaperPose]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found = detect_markers_on_gray(gray)
    found = detect_markers_from_blobs(bgr, found)
    found = {marker_id: pts for marker_id, pts in found.items() if marker_id in EXPECTED_IDS}
    found = fill_missing_markers_from_blobs(bgr, found)
    pose = estimate_paper_pose(found)
    pose.rectified_bgr = rectify_paper(bgr, pose)
    return found, pose


def save_outputs(
    out_dir: Path,
    raw_bgr: np.ndarray,
    found: dict[int, np.ndarray],
    pose: PaperPose,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "raw": str(out_dir / f"capture_{stamp}.jpg"),
        "overlay": str(out_dir / f"overlay_{stamp}.jpg"),
        "meta": str(out_dir / f"pose_{stamp}.json"),
    }
    cv2.imwrite(paths["raw"], raw_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(paths["overlay"], draw_result(raw_bgr, found, pose))

    if pose.rectified_bgr is not None:
        paths["rectified"] = str(out_dir / f"rectified_{stamp}.jpg")
        cv2.imwrite(paths["rectified"], pose.rectified_bgr)

    meta = {
        "timestamp": stamp,
        "detected_ids": pose.detected_ids,
        "missing_ids": pose.missing_ids,
        "angle_deg": None if math.isnan(pose.angle_deg) else pose.angle_deg,
        "center_image": None if np.isnan(pose.center_image[0]) else pose.center_image.tolist(),
        "corner_points_image": {k: v.tolist() for k, v in pose.corner_points_image.items()},
        "paper_size_px": list(pose.paper_size_px),
        "homography_ready": pose.homography_image_to_paper is not None and len(pose.detected_ids) == 4,
    }
    paths["meta"] = str(out_dir / f"pose_{stamp}.json")
    Path(paths["meta"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def write_collect_header(csv_path: Path) -> None:
    if csv_path.exists():
        return
    csv_path.write_text(
        "timestamp,image_file,detected_ids,missing_ids,homography_ready,angle_deg,center_x,center_y\n",
        encoding="utf-8",
    )


def append_collect_pose(csv_path: Path, timestamp: int, image_file: str, pose: PaperPose) -> None:
    detected = "|".join(str(item) for item in pose.detected_ids)
    missing = "|".join(str(item) for item in pose.missing_ids)
    homography_ready = pose.homography_image_to_paper is not None and len(pose.detected_ids) == 4
    angle = "" if math.isnan(pose.angle_deg) else f"{pose.angle_deg:.6f}"
    center_x = "" if np.isnan(pose.center_image[0]) else f"{pose.center_image[0]:.3f}"
    center_y = "" if np.isnan(pose.center_image[1]) else f"{pose.center_image[1]:.3f}"
    with csv_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{timestamp},{image_file},{detected},{missing},{int(homography_ready)},"
            f"{angle},{center_x},{center_y}\n"
        )


def collect_stream(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir) if args.session_dir else (
        DEFAULT_DATA_COLLECT_ROOT / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    )
    save_date = args.save_date or session_dir.parent.name
    image_dir = session_dir / args.image_subdir
    state_dir = session_dir / args.state_subdir
    image_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    csv_path = state_dir / "paper_aruco_pose.csv"
    write_collect_header(csv_path)

    meta_path = session_dir / "paper_aruco_meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(),
                    "device": args.device,
                    "width": args.width,
                    "height": args.height,
                    "collect_hz": args.collect_hz,
                    "image_subdir": args.image_subdir,
                    "state_subdir": args.state_subdir,
                    "rotate_180": args.rotate_180,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    for _ in range(max(0, args.warmup_frames)):
        cap.read()

    period = 1.0 / max(0.1, args.collect_hz)
    count = 0
    last_status = time.monotonic()
    print(f"Paper ArUco collect started: {session_dir}", flush=True)
    try:
        while args.max_frames <= 0 or count < args.max_frames:
            loop_start = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("WARN: failed to read paper camera frame", flush=True)
                time.sleep(period)
                continue
            if args.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            timestamp = elapsed_microseconds(save_date)
            found, pose = localize_paper(frame)
            count += 1
            image_name = f"{timestamp}.{count:06d}.jpg"
            image_path = image_dir / image_name
            if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]):
                raise RuntimeError(f"cv2.imwrite failed: {image_path}")
            append_collect_pose(csv_path, timestamp, image_name, pose)

            if args.save_overlay:
                overlay_path = image_dir / f"{timestamp}.{count:06d}.overlay.jpg"
                cv2.imwrite(str(overlay_path), draw_result(frame, found, pose))

            now = time.monotonic()
            if now - last_status >= args.status_period_sec:
                print(
                    f"Paper ArUco collect: saved={count}, detected={pose.detected_ids}, missing={pose.missing_ids}",
                    flush=True,
                )
                last_status = now

            sleep_s = period - (time.monotonic() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Paper ArUco collect stopped by KeyboardInterrupt", flush=True)
    finally:
        cap.release()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize paper pose using four ArUco corner markers.")
    parser.add_argument("--device", default="/dev/video1", help="USB camera device path")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--rotate-180", action="store_true", default=True)
    parser.add_argument("--no-rotate-180", action="store_false", dest="rotate_180")
    parser.add_argument("--image", default="", help="Use existing image instead of capturing")
    parser.add_argument("--no-capture", action="store_true", help="Alias of --image if provided")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--collect", action="store_true", help="Continuously save timestamped paper camera frames for training alignment")
    parser.add_argument("--session-dir", default="", help="Data-collect session dir; images are saved below this directory")
    parser.add_argument("--save-date", default="", help="YYYY-MM-DD used for microsecond timestamps; default derives from session dir")
    parser.add_argument("--image-subdir", default="camera_paper_aruco")
    parser.add_argument("--state-subdir", default="paper_state")
    parser.add_argument("--collect-hz", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until interrupted")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--save-overlay", action="store_true", help="Also save debug overlay images")
    parser.add_argument("--status-period-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.collect:
        return collect_stream(args)

    out_dir = Path(args.out_dir)

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"Cannot read image: {args.image}")
    else:
        bgr = capture_usb_frame(args.device, args.width, args.height, args.warmup_frames, args.rotate_180)

    found, pose = localize_paper(bgr)
    paths = save_outputs(out_dir, bgr, found, pose)

    print("=== Paper ArUco localization ===")
    print(f"Detected IDs: {pose.detected_ids}")
    print(f"Missing IDs:  {pose.missing_ids}")
    if not math.isnan(pose.angle_deg):
        print(f"Paper top-edge angle: {pose.angle_deg:.2f} deg")
        print(f"Paper center (px): ({pose.center_image[0]:.1f}, {pose.center_image[1]:.1f})")
    print(f"Homography ready (4/4 markers): {pose.homography_image_to_paper is not None and len(pose.detected_ids) == 4}")
    for key, path in paths.items():
        print(f"{key}: {path}")

    if len(pose.detected_ids) < 4:
        print("\nWARN: fewer than 4 markers decoded. Check lighting, focus, marker size, and white border.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
