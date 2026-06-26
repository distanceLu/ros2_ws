#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Episode 级毛笔笔迹图像评分（奖励函数）。

设计目标
--------
给真机强化学习提供一个 episode 结束时的图像 reward。

输入
----
- before_bgr : episode 开始前的纸面相机图
- after_bgr  : episode 结束后的纸面相机图

输出
----
dict (ScoreResult):
    valid, reason, mode,
    coverage, overflow, shape_score, verticality, length_score,
    score, reward, ink_area, target_area, vis

评分模式
--------
模块按以下顺序尝试评分，第一个成功者胜出：

1. ink_shape 模式（auto 默认优先，对市售练习纸最稳健）
   - 不依赖 target 检测，直接对墨迹自身形状评分：
       aspect_score   : h/w 接近理想竖线 aspect
       verticality    : PCA 主轴与竖直方向夹角越小越高
       straightness   : PCA 主轴残差越小越高（侧向偏离）
       length_score   : 主轴长度接近 TARGET_LENGTH_PX
   - 综合 score = 100 * (0.3*aspect + 0.2*verticality + 0.3*straightness + 0.2*length)

2. target_overlap 模式（fallback；适用于打印模板纸且 target 唯一可识别）
   - 在 before 图检测预印胶囊形竖线轮廓作为候选 target mask。
   - 若墨迹与某个 target mask 重叠且 target 面积与墨迹面积相当，用 coverage/overflow/shape 评分。
   - 含可靠性守卫：target 面积 < 15% * ink_area 时视为误检，不采用。

方法细节
--------
1. 两张图统一降采样到 WORKING_WIDTH。
2. 识别纸面 interior mask（浅色区域 + 内缩）。
3. before/after 灰度差分 > INK_DIFF_TH 得到变暗区域。
4. HSV 颜色过滤排除机械臂假阳性：
   - after 像素 V < INK_V_MAX 且 S < INK_S_MAX（墨迹深黑低饱和）；
   - before 像素灰度 > INK_BEFORE_MIN_GRAY（排除 arm 静态暗区）。
5. 竖直形态学开运算 (1, VERTICAL_OPEN_LEN) 分离细长墨迹与粗短 arm 夹具块。
6. 连通块形状过滤：保留 width <= INK_MAX_WIDTH 且 aspect >= INK_MIN_ASPECT 的竖直块。

已知局限
--------
- 机械臂夹具在所有帧都进入画面时，差分会捕获夹具移动假阳性；上述过滤能去掉大部分。
- 推荐 episode 结束后让机械臂抬离纸面到画面外再拍 after 帧，可获得更可靠的图像 reward。
- 不依赖 ArUco。

所有图像操作只读，不会修改调用方传入的数组。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ----------------------------- 可调参数 -----------------------------

WORKING_WIDTH = 1600

# 墨迹差分
INK_DIFF_TH = 25
INK_V_MAX = 90
INK_S_MAX = 85
INK_BEFORE_MIN_GRAY = 100
INK_MIN_AREA_PX = 40
INK_OPEN_KERNEL = 3
INK_CLOSE_KERNEL = 5

# 竖直形态学开运算
VERTICAL_OPEN_LEN = 35
INK_MAX_WIDTH = 70
INK_MIN_ASPECT = 2.0

# 预印轮廓（pill）检测 —— target_overlap 模式
PAPER_GRAY_MIN = 150
PAPER_GRAY_MAX = 255
PAPER_ERODE_PX = 20
PILL_GRAY_MIN = 90
PILL_GRAY_MAX = 150
PILL_MIN_AREA_PX = 600
PILL_MIN_ASPECT = 1.8
PILL_MAX_ASPECT = 8.0
PILL_MAX_WIDTH = 120
PILL_BORDER_MARGIN = 8
PILL_CLOSE_KERNEL = 9
PILL_CLOSE_ITERS = 3

# 容忍区域与 ROI —— target_overlap 模式
TOLERANT_DILATE_PX = 6
ROI_EXPAND_PX = 60

# ink_shape 模式参数
TARGET_LENGTH_PX = 220         # 理想竖线长度（像素，WORKING 坐标系）
TARGET_ASPECT = 3.5            # 理想竖线 h/w
MAX_VERTICALITY_DEG = 15.0     # 主轴与竖直方向夹角超过此值则 verticality=0
MAX_STRAIGHT_RESID_PX = 30.0   # PCA 主轴残差中位数超过此值则 straightness=0

# 评分权重 —— target_overlap 模式
W_COVERAGE = 70.0
W_SHAPE_T = 30.0
W_OVERFLOW_PENALTY = 80.0

# 评分权重 —— ink_shape 模式
W_ASPECT = 0.20
W_VERTICALITY = 0.15
W_STRAIGHTNESS = 0.25
W_LENGTH = 0.40


# ----------------------------- 数据结构 -----------------------------


@dataclass
class PillTarget:
    target_id: str
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int = 0
    tolerant_mask: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.area = int((self.mask > 0).sum())
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * TOLERANT_DILATE_PX + 1, 2 * TOLERANT_DILATE_PX + 1)
        )
        self.tolerant_mask = cv2.dilate(self.mask, kernel)


@dataclass
class ScoreResult:
    valid: bool
    reason: str = ""
    mode: str = ""              # "target_overlap" / "ink_shape" / ""
    target_id: str = ""
    target_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0
    overflow: float = 0.0
    shape_score: float = 0.0
    verticality: float = 0.0
    straightness_ink: float = 0.0
    length_score: float = 0.0
    aspect_score: float = 0.0
    main_axis_deg: float = 0.0
    main_length_px: float = 0.0
    score: float = 0.0
    reward: float = 0.0
    ink_area: int = 0
    target_area: int = 0
    vis: Optional[np.ndarray] = None

    def to_dict(self, include_vis: bool = False) -> dict:
        d = {
            "valid": self.valid,
            "reason": self.reason,
            "mode": self.mode,
            "target_id": self.target_id,
            "target_bbox": list(self.target_bbox),
            "coverage": round(self.coverage, 4),
            "overflow": round(self.overflow, 4),
            "shape_score": round(self.shape_score, 4),
            "verticality": round(self.verticality, 4),
            "straightness_ink": round(self.straightness_ink, 4),
            "length_score": round(self.length_score, 4),
            "aspect_score": round(self.aspect_score, 4),
            "main_axis_deg": round(self.main_axis_deg, 2),
            "main_length_px": round(self.main_length_px, 2),
            "score": round(self.score, 2),
            "reward": round(self.reward, 4),
            "ink_area": self.ink_area,
            "target_area": self.target_area,
        }
        if include_vis and self.vis is not None:
            d["vis_shape"] = list(self.vis.shape)
        return d


# ----------------------------- 内部工具 -----------------------------


def _resize_to_working(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("输入必须是 BGR 三通道图")
    h, w = bgr.shape[:2]
    if w == WORKING_WIDTH:
        return bgr.copy(), 1.0
    scale = WORKING_WIDTH / float(w)
    out = cv2.resize(bgr, (WORKING_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _paper_interior_mask(working_bgr: np.ndarray) -> np.ndarray:
    gray = _gray(working_bgr)
    h, w = gray.shape
    paper = cv2.inRange(gray, PAPER_GRAY_MIN, PAPER_GRAY_MAX)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.1 * w * h:
        return np.zeros((h, w), dtype=np.uint8)
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [biggest], -1, 255, -1)
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PAPER_ERODE_PX, PAPER_ERODE_PX))
    return cv2.erode(filled, erode_k)


def _detect_pill_targets(working_bgr: np.ndarray, paper_int: np.ndarray) -> list[PillTarget]:
    gray = _gray(working_bgr)
    h, w = gray.shape

    bin_outline = cv2.inRange(gray, PILL_GRAY_MIN, PILL_GRAY_MAX) & paper_int
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PILL_CLOSE_KERNEL, PILL_CLOSE_KERNEL))
    bin_outline = cv2.morphologyEx(bin_outline, cv2.MORPH_CLOSE, kernel, iterations=PILL_CLOSE_ITERS)
    bin_outline = cv2.morphologyEx(bin_outline, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bin_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < PILL_MIN_AREA_PX:
            continue
        if bw < 8 or bh < 16:
            continue
        if bw > PILL_MAX_WIDTH:
            continue
        aspect = bh / float(bw)
        if aspect < PILL_MIN_ASPECT or aspect > PILL_MAX_ASPECT:
            continue
        if (x < PILL_BORDER_MARGIN or y < PILL_BORDER_MARGIN
                or x + bw > w - PILL_BORDER_MARGIN or y + bh > h - PILL_BORDER_MARGIN):
            continue
        candidates.append((int(area), (x, y, bw, bh), cnt))

    candidates.sort(key=lambda item: item[0], reverse=True)

    targets: list[PillTarget] = []
    for idx, (_, bbox, cnt) in enumerate(candidates, start=1):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        targets.append(PillTarget(target_id=f"target_{idx:03d}", bbox=bbox, mask=mask))
    return targets


def _extract_ink_mask(
    working_before: np.ndarray,
    working_after: np.ndarray,
    paper_int: np.ndarray,
) -> np.ndarray:
    gb = _gray(working_before)
    ga = _gray(working_after)
    hsv_a = cv2.cvtColor(working_after, cv2.COLOR_BGR2HSV)

    diff = cv2.GaussianBlur((gb.astype(np.int16) - ga.astype(np.int16)).astype(np.int16), (5, 5), 0)
    diff_mask = (diff > INK_DIFF_TH).astype(np.uint8) * 255

    dark_ink = cv2.inRange(hsv_a, (0, 0, 0), (180, INK_S_MAX, INK_V_MAX))
    before_not_dark = (gb > INK_BEFORE_MIN_GRAY).astype(np.uint8) * 255

    ink = diff_mask & dark_ink & before_not_dark & paper_int

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INK_OPEN_KERNEL, INK_OPEN_KERNEL))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INK_CLOSE_KERNEL, INK_CLOSE_KERNEL))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_close)

    vk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, VERTICAL_OPEN_LEN))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_close)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = np.zeros_like(ink)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < INK_MIN_AREA_PX:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw > INK_MAX_WIDTH:
            continue
        if bh / max(bw, 1) < INK_MIN_ASPECT:
            continue
        out[labels == i] = 255
    return out


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _select_target(ink_mask: np.ndarray, targets: list[PillTarget]) -> Optional[PillTarget]:
    if not targets or (ink_mask > 0).sum() == 0:
        return None
    best: Optional[PillTarget] = None
    best_overlap = 0
    for tgt in targets:
        overlap = int(np.count_nonzero((ink_mask > 0) & (tgt.mask > 0)))
        if overlap > best_overlap:
            best_overlap = overlap
            best = tgt
    return best


def _shape_score_iou(ink_mask: np.ndarray, target_mask: np.ndarray) -> float:
    ink_contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tgt_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ink_contours or not tgt_contours:
        return 0.0
    ink_cnt = max(ink_contours, key=cv2.contourArea)
    tgt_cnt = max(tgt_contours, key=cv2.contourArea)
    if cv2.contourArea(ink_cnt) < 10 or cv2.contourArea(tgt_cnt) < 10:
        return 0.0
    distance = cv2.matchShapes(tgt_cnt, ink_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    return float(max(0.0, 1.0 - distance))


def _ink_shape_metrics(ink_mask: np.ndarray) -> dict:
    """对墨迹 mask 做形状分析，返回 ink_shape 模式所需指标。"""
    cnt = _largest_contour(ink_mask)
    if cnt is None or cv2.contourArea(cnt) < INK_MIN_AREA_PX:
        return {}
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = h / max(w, 1)

    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean, eig = cv2.PCACompute(pts, mean=None)
    if mean is None or eig is None or eig.shape[0] < 2:
        return {}
    centered = pts - mean[0]
    proj1 = centered @ eig[0]      # 沿主轴投影
    recon1 = proj1.reshape(-1, 1) * eig[0].reshape(1, 2)
    resid = centered - recon1
    resid_mag = np.linalg.norm(resid, axis=1)
    main_length = float(proj1.max() - proj1.min())

    # 主轴方向：eig[0] = (ex, ey)，与竖直 (0, 1) 的夹角
    ex, ey = float(eig[0][0]), float(eig[0][1])
    # 竖直方向单位向量 (0, 1)。取 |dot|，因为主轴正负方向都对
    cos_vert = abs(ey) / math.sqrt(ex * ex + ey * ey)
    angle_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_vert))))

    aspect_score = float(max(0.0, 1.0 - abs(aspect - TARGET_ASPECT) / TARGET_ASPECT))
    verticality = float(max(0.0, 1.0 - angle_deg / MAX_VERTICALITY_DEG))
    median_resid = float(np.median(resid_mag))
    straightness = float(max(0.0, 1.0 - median_resid / MAX_STRAIGHT_RESID_PX))
    length_score = float(min(1.0, main_length / TARGET_LENGTH_PX))

    return {
        "aspect": float(aspect),
        "aspect_score": aspect_score,
        "verticality": verticality,
        "main_axis_deg": angle_deg,
        "main_length_px": main_length,
        "median_resid_px": median_resid,
        "straightness": straightness,
        "length_score": length_score,
        "bbox": (x, y, w, h),
    }


def _make_visualization_target(
    working_bgr: np.ndarray,
    ink_mask: np.ndarray,
    target: PillTarget,
) -> np.ndarray:
    vis = working_bgr.copy()
    target_region = target.mask > 0
    tolerant_region = target.tolerant_mask > 0
    ink_region = ink_mask > 0

    inside = ink_region & target_region
    overflow = ink_region & (~tolerant_region)
    missed = target_region & (~ink_region)

    overlay = vis.copy()
    overlay[inside] = (0, 255, 0)
    overlay[overflow] = (0, 0, 255)
    overlay[missed] = (255, 128, 0)
    vis = cv2.addWeighted(vis, 0.4, overlay, 0.6, 0)

    x, y, w, h = target.bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(vis, target.target_id, (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def _make_visualization_ink_shape(
    working_bgr: np.ndarray,
    ink_mask: np.ndarray,
    metrics: dict,
) -> np.ndarray:
    vis = working_bgr.copy()
    ink_region = ink_mask > 0
    overlay = vis.copy()
    overlay[ink_region] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)

    cnt = _largest_contour(ink_mask)
    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.drawContours(vis, [cnt], -1, (255, 0, 0), 2)
        # 画主轴
        mean, eig = cv2.PCACompute(cnt.reshape(-1, 2).astype(np.float32), mean=None)
        if mean is not None and eig is not None:
            cx, cy = float(mean[0][0]), float(mean[0][1])
            dx, dy = float(eig[0][0]), float(eig[0][1])
            norm = math.sqrt(dx * dx + dy * dy) + 1e-6
            dx, dy = dx / norm, dy / norm
            L = metrics.get("main_length_px", 100.0) / 2.0
            p1 = (int(cx - dx * L), int(cy - dy * L))
            p2 = (int(cx + dx * L), int(cy + dy * L))
            cv2.line(vis, p1, p2, (0, 0, 255), 2)

    label = (
        f"asp={metrics.get('aspect', 0):.2f} "
        f"ang={metrics.get('main_axis_deg', 0):.1f} "
        f"len={metrics.get('main_length_px', 0):.0f}"
    )
    cv2.putText(vis, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


# ----------------------------- 主接口 -----------------------------


def score_episode(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    generate_vis: bool = True,
    prefer_mode: str = "auto",
) -> ScoreResult:
    """对一次 episode 的 before/after 纸面图评分。

    prefer_mode: "auto"（默认，优先 ink_shape，必要时回退 target_overlap）
                 "ink_shape"（仅墨迹形状模式，对市售练习纸最稳健）
                 "target_overlap"（仅 target 模式，需打印模板纸且 target 唯一可识别）
    """
    if before_bgr is None or after_bgr is None:
        return ScoreResult(valid=False, reason="before/after image is None")

    try:
        wb, _ = _resize_to_working(before_bgr)
        wa, _ = _resize_to_working(after_bgr)
    except Exception as exc:
        return ScoreResult(valid=False, reason=f"resize failed: {exc}")

    if wb.shape[:2] != wa.shape[:2]:
        h = min(wb.shape[0], wa.shape[0])
        w = min(wb.shape[1], wa.shape[1])
        wb = wb[:h, :w]
        wa = wa[:h, :w]

    paper_int = _paper_interior_mask(wb)
    if (paper_int > 0).sum() == 0:
        return ScoreResult(valid=False, reason="paper region not detected in before image")

    ink = _extract_ink_mask(wb, wa, paper_int)
    ink_area = int((ink > 0).sum())
    if ink_area == 0:
        return ScoreResult(
            valid=False,
            reason="no new ink detected (before/after diff empty after filtering)",
            vis=wb.copy() if generate_vis else None,
        )

    # ink_shape 模式（auto 默认优先，对市售练习纸最稳健）
    if prefer_mode in ("auto", "ink_shape"):
        metrics = _ink_shape_metrics(ink)
        if metrics:
            score = 100.0 * (
                W_ASPECT * metrics["aspect_score"]
                + W_VERTICALITY * metrics["verticality"]
                + W_STRAIGHTNESS * metrics["straightness"]
                + W_LENGTH * metrics["length_score"]
            )
            score = float(max(0.0, min(100.0, score)))
            vis = _make_visualization_ink_shape(wa, ink, metrics) if generate_vis else None

            return ScoreResult(
                valid=True,
                mode="ink_shape",
                aspect_score=metrics["aspect_score"],
                verticality=metrics["verticality"],
                main_axis_deg=metrics["main_axis_deg"],
                main_length_px=metrics["main_length_px"],
                straightness_ink=metrics["straightness"],
                length_score=metrics["length_score"],
                score=score,
                reward=score / 100.0,
                ink_area=ink_area,
                vis=vis,
            )

        if prefer_mode == "ink_shape":
            return ScoreResult(
                valid=False,
                reason="ink_shape metrics unavailable (ink contour too small)",
                ink_area=ink_area,
                vis=wb.copy() if generate_vis else None,
            )

    # target_overlap 模式（fallback；适用于打印模板纸且每个 target 唯一可识别）
    if prefer_mode in ("auto", "target_overlap"):
        targets = _detect_pill_targets(wb, paper_int)
        target = _select_target(ink, targets) if targets else None

        # 可靠性守卫：target 面积应与 ink 面积相当，否则很可能是误检 pill
        if target is not None and target.area >= max(50, int(0.15 * ink_area)):
            target_area = max(1, target.area)
            roi_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ROI_EXPAND_PX + 1, 2 * ROI_EXPAND_PX + 1)
            )
            roi = cv2.dilate(target.mask, roi_kernel)
            ink_roi = cv2.bitwise_and(ink, roi)

            coverage = float(np.count_nonzero((ink_roi > 0) & (target.mask > 0))) / target_area
            ink_roi_area = max(1, int((ink_roi > 0).sum()))
            overflow = float(
                np.count_nonzero((ink_roi > 0) & (target.tolerant_mask == 0))
            ) / ink_roi_area
            shape_score = _shape_score_iou(ink_roi, target.mask)

            raw = W_COVERAGE * coverage + W_SHAPE_T * shape_score - W_OVERFLOW_PENALTY * overflow
            score = float(max(0.0, min(100.0, raw)))
            vis = _make_visualization_target(wa, ink_roi, target) if generate_vis else None

            return ScoreResult(
                valid=True,
                mode="target_overlap",
                target_id=target.target_id,
                target_bbox=target.bbox,
                coverage=coverage,
                overflow=overflow,
                shape_score=shape_score,
                score=score,
                reward=score / 100.0,
                ink_area=ink_area,
                target_area=target.area,
                vis=vis,
            )

        if prefer_mode == "target_overlap":
            return ScoreResult(
                valid=False,
                reason="target_overlap mode requested but no reliable target overlaps ink",
                ink_area=ink_area,
                vis=wb.copy() if generate_vis else None,
            )

    return ScoreResult(
        valid=False,
        reason="no scoring mode succeeded",
        ink_area=ink_area,
        vis=wb.copy() if generate_vis else None,
    )


# ----------------------------- CLI 自检 -----------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(description="毛笔笔迹 episode 图像评分自检")
    parser.add_argument("--before", required=True, help="before 图路径")
    parser.add_argument("--after", required=True, help="after 图路径")
    parser.add_argument("--out-dir", default="", help="输出目录（存 reward.json 和 vis.png）")
    parser.add_argument("--mode", default="auto", choices=["auto", "target_overlap", "ink_shape"],
                        help="评分模式")
    args = parser.parse_args()

    before = cv2.imread(args.before)
    after = cv2.imread(args.after)
    if before is None:
        print(f"无法读取 before: {args.before}")
        return 1
    if after is None:
        print(f"无法读取 after: {args.after}")
        return 1

    result = score_episode(before, after, generate_vis=True, prefer_mode=args.mode)
    print("=== ink_reward score ===")
    print(json.dumps(result.to_dict(include_vis=True), indent=2, ensure_ascii=False))

    if args.out_dir and result.vis is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "reward_vis.png"), result.vis)
        (out_dir / "reward.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"saved: {out_dir / 'reward_vis.png'}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    sys.exit(_cli())
