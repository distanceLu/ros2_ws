#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry helpers for calligraphy-style vertical stroke (竖) targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

DEFAULT_PAPER_WIDTH = 2000
DEFAULT_PAPER_HEIGHT = 2829


@dataclass
class TargetSpec:
    target_id: str
    center_x: float
    center_y: float
    width: float
    height: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameSpec:
    frame_id: str
    x: float
    y: float
    width: float
    height: float
    targets: list[TargetSpec]

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "targets": [target.to_dict() for target in self.targets],
        }


@dataclass
class PaperTemplateSpec:
    paper_width: int
    paper_height: int
    frames: list[FrameSpec]
    outline_thickness: int = 3
    frame_thickness: int = 2

    def to_dict(self) -> dict:
        return {
            "paper_width": self.paper_width,
            "paper_height": self.paper_height,
            "outline_thickness": self.outline_thickness,
            "frame_thickness": self.frame_thickness,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @staticmethod
    def from_dict(data: dict) -> "PaperTemplateSpec":
        frames = []
        for frame in data["frames"]:
            targets = [TargetSpec(**target) for target in frame["targets"]]
            frames.append(
                FrameSpec(
                    frame_id=frame["frame_id"],
                    x=frame["x"],
                    y=frame["y"],
                    width=frame["width"],
                    height=frame["height"],
                    targets=targets,
                )
            )
        return PaperTemplateSpec(
            paper_width=int(data["paper_width"]),
            paper_height=int(data["paper_height"]),
            frames=frames,
            outline_thickness=int(data.get("outline_thickness", 3)),
            frame_thickness=int(data.get("frame_thickness", 2)),
        )


def shu_stroke_polygon(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    tip_ratio: float = 0.22,
) -> np.ndarray:
    """Return a closed polygon for a round-head vertical stroke (圆头竖).

    Bottom: semicircle. Middle: parallel sides. Top: tapered rounded tip.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    w2 = width / 2.0
    cx = float(center_x)
    cy = float(center_y)
    half_h = height / 2.0
    y_bottom = cy + half_h - w2
    y_top = cy - half_h
    tip_height = max(width * 1.1, height * tip_ratio)
    y_shoulder = y_top + tip_height

    pts: list[list[float]] = []

    # Bottom semicircle, left -> right.
    for theta in np.linspace(np.pi, 0.0, 24):
        pts.append([cx + w2 * np.cos(theta), y_bottom + w2 * np.sin(theta)])

    # Right side up to shoulder.
    pts.append([cx + w2, y_shoulder])

    # Tapered top cap, right -> tip -> left.
    for t in np.linspace(0.0, 1.0, 16):
        x = cx + w2 * (1.0 - t) ** 0.85
        y = y_shoulder - (y_shoulder - y_top) * t
        pts.append([x, y])

    for t in np.linspace(1.0, 0.0, 16):
        x = cx - w2 * (1.0 - t) ** 0.85
        y = y_shoulder - (y_shoulder - y_top) * t
        pts.append([x, y])

    # Left side down to bottom arc start.
    pts.append([cx - w2, y_shoulder])

    contour = np.array(pts, dtype=np.float32)
    return np.round(contour).astype(np.int32)


def iter_targets(spec: PaperTemplateSpec) -> Iterable[TargetSpec]:
    for frame in spec.frames:
        yield from frame.targets


def blank_canvas(spec: PaperTemplateSpec, value: int) -> np.ndarray:
    return np.full((spec.paper_height, spec.paper_width), value, dtype=np.uint8)


def render_frame_outlines(canvas: np.ndarray, spec: PaperTemplateSpec) -> None:
    for frame in spec.frames:
        x1 = int(round(frame.x))
        y1 = int(round(frame.y))
        x2 = int(round(frame.x + frame.width))
        y2 = int(round(frame.y + frame.height))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 80, spec.frame_thickness)


def render_target_outlines(canvas: np.ndarray, spec: PaperTemplateSpec) -> None:
    for target in iter_targets(spec):
        polygon = shu_stroke_polygon(target.center_x, target.center_y, target.width, target.height)
        cv2.polylines(canvas, [polygon], isClosed=True, color=0, thickness=spec.outline_thickness)


def render_target_mask(target: TargetSpec, paper_width: int, paper_height: int) -> np.ndarray:
    mask = np.zeros((paper_height, paper_width), dtype=np.uint8)
    polygon = shu_stroke_polygon(target.center_x, target.center_y, target.width, target.height)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def render_paper_template(spec: PaperTemplateSpec) -> np.ndarray:
    canvas = blank_canvas(spec, 255)
    render_frame_outlines(canvas, spec)
    render_target_outlines(canvas, spec)
    return canvas


def save_template_assets(spec: PaperTemplateSpec, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    template = render_paper_template(spec)
    template_path = out_dir / "paper_template.png"
    cv2.imwrite(str(template_path), template)

    config_path = out_dir / "template_config.json"
    config_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    mask_paths: dict[str, str] = {}
    for target in iter_targets(spec):
        mask = render_target_mask(target, spec.paper_width, spec.paper_height)
        mask_path = mask_dir / f"{target.target_id}.png"
        cv2.imwrite(str(mask_path), mask)
        mask_paths[target.target_id] = str(mask_path)

    preview = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(preview)
    for target in iter_targets(spec):
        mask = render_target_mask(target, spec.paper_width, spec.paper_height)
        overlay[mask > 0] = (0, 180, 0)
    preview = cv2.addWeighted(preview, 0.75, overlay, 0.25, 0)
    preview_path = out_dir / "template_mask_preview.png"
    cv2.imwrite(str(preview_path), preview)

    return {
        "template": str(template_path),
        "config": str(config_path),
        "preview": str(preview_path),
        "masks": mask_paths,
    }


def default_demo_spec() -> PaperTemplateSpec:
    """Three horizontal frames, each with one 竖 target."""
    paper_width = DEFAULT_PAPER_WIDTH
    paper_height = DEFAULT_PAPER_HEIGHT
    frame_w = 1500
    frame_h = 520
    frame_x = (paper_width - frame_w) // 2
    row_ys = [520, 1180, 1840]
    targets: list[FrameSpec] = []
    for idx, y in enumerate(row_ys, start=1):
        target_id = f"target_{idx:03d}"
        targets.append(
            FrameSpec(
                frame_id=f"frame_{idx:03d}",
                x=frame_x,
                y=y,
                width=frame_w,
                height=frame_h,
                targets=[
                    TargetSpec(
                        target_id=target_id,
                        center_x=paper_width / 2.0,
                        center_y=y + frame_h / 2.0,
                        width=110,
                        height=360,
                    )
                ],
            )
        )
    return PaperTemplateSpec(
        paper_width=paper_width,
        paper_height=paper_height,
        frames=targets,
    )
