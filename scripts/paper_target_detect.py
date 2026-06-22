#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect horizontal frames and filled vertical-stroke targets on rectified paper images."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from paper_target_shapes import PaperTemplateSpec, shu_stroke_polygon


@dataclass
class DetectedTarget:
    target_id: str
    score: float
    center_x: float
    center_y: float
    width: float
    height: float
    bbox: tuple[int, int, int, int]
    contour: list[list[float]]
    frame_id: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def _gray_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _preprocess(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        7,
    )
    # Merge thin printed outlines into solid blobs for contour analysis.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)


def detect_frames(binary: np.ndarray, min_area_ratio: float = 0.01) -> list[tuple[int, int, int, int]]:
    h, w = binary.shape[:2]
    min_area = h * w * min_area_ratio
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frames: list[tuple[int, int, int, int, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < w * 0.25 or bh < h * 0.05:
            continue
        aspect = bw / float(bh)
        if aspect < 1.5:
            continue
        frames.append((x, y, bw, bh, area))
    frames.sort(key=lambda item: (item[1], item[0]))
    return [(x, y, bw, bh) for x, y, bw, bh, _ in frames]


def contour_shape_features(cnt: np.ndarray) -> dict[str, float]:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0 or area <= 0:
        return {}
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    rect_area = float(w * h)
    return {
        "area": float(area),
        "width": float(w),
        "height": float(h),
        "aspect": float(h / w),
        "extent": float(area / rect_area),
        "solidity": float(area / hull_area) if hull_area > 0 else 0.0,
        "center_x": float(x + w / 2.0),
        "center_y": float(y + h / 2.0),
    }


def score_vertical_stroke(features: dict[str, float], image_area: float) -> float:
    if not features:
        return 0.0
    aspect = features["aspect"]
    extent = features["extent"]
    solidity = features["solidity"]
    area = features["area"]
    min_area = max(120.0, image_area * 0.00005)

    if area < min_area:
        return 0.0
    if aspect < 2.0:
        return 0.0
    if solidity < 0.80:
        return 0.0
    if extent < 0.35 or extent > 0.80:
        return 0.0

    aspect_score = 1.0 - min(abs(aspect - 3.8) / 3.8, 1.0)
    extent_score = 1.0 - min(abs(extent - 0.58) / 0.58, 1.0)
    solidity_score = min(solidity, 1.0)
    return float(0.45 * aspect_score + 0.35 * extent_score + 0.20 * solidity_score)


def reference_shape_score(cnt: np.ndarray, ref_width: float, ref_height: float) -> float:
    ref = shu_stroke_polygon(0.0, 0.0, ref_width, ref_height).astype(np.float32)
    ref[:, 0] -= ref[:, 0].min()
    ref[:, 1] -= ref[:, 1].min()
    probe = cnt.reshape(-1, 2).astype(np.float32)
    x, y, _, _ = cv2.boundingRect(probe.astype(np.int32))
    probe[:, 0] -= x
    probe[:, 1] -= y
    if ref.size == 0 or probe.size == 0:
        return 0.0
    # Smaller Hu distance is better.
    distance = cv2.matchShapes(ref, probe, cv2.CONTOURS_MATCH_I1, 0.0)
    return float(max(0.0, 1.0 - distance))


def detect_targets_in_image(
    image: np.ndarray,
    spec: PaperTemplateSpec | None = None,
    min_score: float = 0.35,
) -> list[DetectedTarget]:
    scale = 1.0
    if min(image.shape[0], image.shape[1]) < 400:
        scale = 400.0 / float(min(image.shape[0], image.shape[1]))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = _gray_image(image)
    binary = _preprocess(gray)
    image_area = float(gray.shape[0] * gray.shape[1])
    frames = detect_frames(binary)
    if not frames:
        frames = [(0, 0, gray.shape[1], gray.shape[0])]

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray, tuple[int, int, int, int], str]] = []

    ref_width = 110.0
    ref_height = 360.0
    if spec is not None and spec.frames and spec.frames[0].targets:
        ref = spec.frames[0].targets[0]
        ref_width = ref.width
        ref_height = ref.height

    for frame_idx, (fx, fy, fw, fh) in enumerate(frames):
        frame_id = f"frame_{frame_idx + 1:03d}"
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w / 2.0
            cy = y + h / 2.0
            if not (fx <= cx <= fx + fw and fy <= cy <= fy + fh):
                continue
            features = contour_shape_features(cnt)
            score = score_vertical_stroke(features, image_area)
            score = 0.6 * score + 0.4 * reference_shape_score(cnt, ref_width, ref_height)
            if score < min_score:
                continue
            frame_area = float(fw * fh)
            if (w * h) > frame_area * 0.40:
                continue
            candidates.append((score, cnt, (x, y, w, h), frame_id))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[DetectedTarget] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    for idx, (score, cnt, bbox, frame_id) in enumerate(candidates, start=1):
        if any(_iou(bbox, used) > 0.35 for used in used_boxes):
            continue
        x, y, w, h = bbox
        if scale != 1.0:
            x = int(round(x / scale))
            y = int(round(y / scale))
            w = int(round(w / scale))
            h = int(round(h / scale))
            cnt_scaled = (cnt.reshape(-1, 2) / scale).astype(float)
        else:
            cnt_scaled = cnt.reshape(-1, 2).astype(float)
        selected.append(
            DetectedTarget(
                target_id=f"detected_{idx:03d}",
                score=score,
                center_x=x + w / 2.0,
                center_y=y + h / 2.0,
                width=float(w),
                height=float(h),
                bbox=(x, y, w, h),
                contour=cnt_scaled.tolist(),
                frame_id=frame_id,
            )
        )
        used_boxes.append(bbox)
    return selected


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


def render_detected_mask(
    paper_width: int,
    paper_height: int,
    detection: DetectedTarget,
) -> np.ndarray:
    mask = np.zeros((paper_height, paper_width), dtype=np.uint8)
    contour = np.array(detection.contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 255)
    return mask


def score_against_reference_mask(
    rectified_image: np.ndarray,
    reference_mask: np.ndarray,
    detection: DetectedTarget | None = None,
) -> float:
    """Score how well ink/fill inside the target matches the standard mask."""
    gray = _gray_image(rectified_image)
    fill = _preprocess(gray)
    if detection is not None:
        det_mask = render_detected_mask(reference_mask.shape[1], reference_mask.shape[0], detection)
        region = (reference_mask > 0) | (det_mask > 0)
    else:
        region = reference_mask > 0
    observed = (fill > 0) & region
    expected = reference_mask > 0
    union = observed | expected
    if union.sum() == 0:
        return 0.0
    return float((observed & expected).sum() / union.sum())


def score_all_targets(
    rectified_image: np.ndarray,
    spec: PaperTemplateSpec,
    detections: list[DetectedTarget],
) -> list[dict]:
    from paper_target_shapes import render_target_mask

    det_by_frame: dict[str, list[DetectedTarget]] = {}
    for det in detections:
        det_by_frame.setdefault(det.frame_id, []).append(det)

    results: list[dict] = []
    for frame_idx, frame in enumerate(spec.frames):
        frame_id = f"frame_{frame_idx + 1:03d}"
        frame_dets = sorted(det_by_frame.get(frame_id, []), key=lambda d: d.score, reverse=True)
        for target_idx, target in enumerate(frame.targets):
            ref_mask = render_target_mask(target, spec.paper_width, spec.paper_height)
            det = frame_dets[target_idx] if target_idx < len(frame_dets) else None
            results.append(
                {
                    "target_id": target.target_id,
                    "frame_id": frame.frame_id,
                    "fill_iou": score_against_reference_mask(rectified_image, ref_mask, det),
                    "detected": det is not None,
                    "detection_score": None if det is None else det.score,
                }
            )
    return results


def draw_detection(image: np.ndarray, detections: list[DetectedTarget], frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in frames:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 128, 0), 2)
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour = np.array(det.contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)
        cv2.putText(
            vis,
            f"{det.target_id} {det.score:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect vertical-stroke targets on rectified paper images.")
    parser.add_argument("--image", required=True, help="Rectified paper image or photo")
    parser.add_argument("--config", default="", help="template_config.json for reference geometry")
    parser.add_argument("--out-dir", default="", help="Directory for overlay and JSON output")
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--score", action="store_true", help="Score fills against standard target masks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    spec = None
    if args.config:
        spec = PaperTemplateSpec.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))

    detections = detect_targets_in_image(image, spec=spec, min_score=args.min_score)
    gray = _gray_image(image)
    frames = detect_frames(_preprocess(gray))
    overlay = draw_detection(image, detections, frames)

    print("=== Target detection ===")
    print(f"frames: {len(frames)}")
    print(f"targets: {len(detections)}")
    for det in detections:
        print(
            f"  {det.target_id} score={det.score:.3f} center=({det.center_x:.1f},{det.center_y:.1f}) "
            f"size=({det.width:.1f}x{det.height:.1f}) frame={det.frame_id}"
        )

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = out_dir / "target_detection_overlay.png"
        json_path = out_dir / "target_detection.json"
        cv2.imwrite(str(overlay_path), overlay)
        if args.score and spec is not None:
            payload = {
                "detections": [det.to_dict() for det in detections],
                "scores": score_all_targets(image, spec, detections),
            }
        else:
            payload = [det.to_dict() for det in detections]
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"overlay: {overlay_path}")
        print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
