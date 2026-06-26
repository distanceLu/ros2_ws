#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workspace_safety.py - build and check TCP workspace limits.

Typical usage:
  # Build a conservative first workspace from an existing tool_pose.csv.
  python3 scripts/workspace_safety.py from-csv \
    --csv data_collect/2026-06-15/19-49-26/robot_state/tool_pose.csv \
    --out scripts/workspace_limits.json \
    --margin-mm 10 \
    --rot-margin-deg 2

  # Teach a local task frame and workspace bounds from /tool_pos.
  python3 scripts/workspace_safety.py teach --out scripts/workspace_limits.json

  # Verify a trajectory against a workspace config.
  python3 scripts/workspace_safety.py check-csv \
    --workspace scripts/workspace_limits.json \
    --csv data_collect/2026-06-15/19-49-26/robot_state/tool_pose.csv
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
LIMIT_EPSILON = 1e-9


def _prepend_env_path(var_name: str, value: str) -> None:
    current = os.environ.get(var_name, "")
    items = [item for item in current.split(os.pathsep) if item]
    if value in items:
        return
    os.environ[var_name] = value if not current else value + os.pathsep + current


def bootstrap_local_ros_paths() -> None:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    install_root = WORKSPACE_ROOT / "install"
    candidates: list[Path] = []
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


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def mul(a: list[float], scale: float) -> list[float]:
    return [x * scale for x in a]


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: list[float], label: str) -> list[float]:
    length = norm(a)
    if length < 1e-9:
        raise ValueError(f"{label} is too short to define a direction")
    return [x / length for x in a]


def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def rad(deg: float) -> float:
    return math.radians(deg)


@dataclass
class Workspace:
    data: dict[str, Any]

    @property
    def origin(self) -> list[float]:
        return [float(v) for v in self.data["frame"]["origin"]]

    @property
    def axes(self) -> list[list[float]]:
        return [[float(v) for v in axis] for axis in self.data["frame"]["axes"]]

    @property
    def position_limits(self) -> dict[str, list[float]]:
        return self.data["position_limits"]

    @property
    def orientation_limits(self) -> dict[str, list[float]]:
        return self.data["orientation_limits"]

    @property
    def tool_clearance_m(self) -> float:
        return float(self.data.get("safety", {}).get("tool_clearance_m", 0.0))

    @property
    def position_clearance_m(self) -> dict[str, float]:
        safety = self.data.get("safety", {})
        fallback = self.tool_clearance_m
        configured = safety.get("position_clearance_m", {})
        if not isinstance(configured, dict):
            configured = {}
        return {
            axis_name: float(configured.get(axis_name, fallback))
            for axis_name in ("x", "y", "z")
        }

    @property
    def check_orientation(self) -> bool:
        return bool(self.data.get("safety", {}).get("check_orientation", True))

    @property
    def path_check_step_m(self) -> float:
        return float(self.data.get("safety", {}).get("path_check_step_m", 0.002))

    def world_to_local(self, xyz: list[float]) -> list[float]:
        delta = sub(xyz, self.origin)
        return [dot(delta, axis) for axis in self.axes]

    def local_to_world(self, local: list[float]) -> list[float]:
        xyz = self.origin
        for value, axis in zip(local, self.axes):
            xyz = add(xyz, mul(axis, value))
        return xyz

    def effective_position_limits(self, margin_m: float = 0.0) -> dict[str, list[float]]:
        clearance = self.position_clearance_m
        limits: dict[str, list[float]] = {}
        for axis_name in ("x", "y", "z"):
            low, high = [float(v) for v in self.position_limits[axis_name]]
            effective_margin_m = margin_m + clearance[axis_name]
            limits[axis_name] = [low + effective_margin_m, high - effective_margin_m]
        return limits

    def contains_pose(self, pose: list[float], margin_m: float = 0.0) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if len(pose) != 6:
            return False, [f"pose length must be 6, got {len(pose)}"]
        for idx, value in enumerate(pose):
            if not math.isfinite(float(value)):
                return False, [f"pose[{idx}] is not finite: {value}"]
        local = self.world_to_local(pose[:3])
        effective_limits = self.effective_position_limits(margin_m=margin_m)
        for axis_name, value in zip(("x", "y", "z"), local):
            low, high = effective_limits[axis_name]
            if value < low - LIMIT_EPSILON:
                reasons.append(f"{axis_name}_local below {low:.6f}: {value:.6f}")
            if value > high + LIMIT_EPSILON:
                reasons.append(f"{axis_name}_local above {high:.6f}: {value:.6f}")

        if self.check_orientation:
            for axis_name, value in zip(("rx", "ry", "rz"), pose[3:]):
                low, high = [float(v) for v in self.orientation_limits[axis_name]]
                if value < low:
                    reasons.append(f"{axis_name} below {low:.6f}: {value:.6f}")
                if value > high:
                    reasons.append(f"{axis_name} above {high:.6f}: {value:.6f}")
        return not reasons, reasons

    def contains_pose_raw_position(self, pose: list[float], tolerance_m: float = 0.0) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if len(pose) != 6:
            return False, [f"pose length must be 6, got {len(pose)}"]
        for idx, value in enumerate(pose):
            if not math.isfinite(float(value)):
                return False, [f"pose[{idx}] is not finite: {value}"]
        local = self.world_to_local(pose[:3])
        for axis_name, value in zip(("x", "y", "z"), local):
            low, high = [float(v) for v in self.position_limits[axis_name]]
            if value < low - tolerance_m - LIMIT_EPSILON:
                reasons.append(f"{axis_name}_local below raw {low:.6f}: {value:.6f}")
            if value > high + tolerance_m + LIMIT_EPSILON:
                reasons.append(f"{axis_name}_local above raw {high:.6f}: {value:.6f}")
        return not reasons, reasons

    def contains_segment_raw_position(
        self,
        start_pose: list[float],
        end_pose: list[float],
        step_m: float,
        tolerance_m: float = 0.0,
    ) -> tuple[bool, list[str]]:
        distance = norm(sub(end_pose[:3], start_pose[:3]))
        samples = max(1, int(math.ceil(distance / max(step_m, 1e-6))))
        for idx in range(samples + 1):
            ratio = idx / samples
            pose = [
                start_pose[col] + (end_pose[col] - start_pose[col]) * ratio
                for col in range(6)
            ]
            ok, reasons = self.contains_pose_raw_position(pose, tolerance_m=tolerance_m)
            if not ok:
                return False, [f"segment sample {idx}/{samples}: {reason}" for reason in reasons]
        return True, []

    def contains_segment(
        self,
        start_pose: list[float],
        end_pose: list[float],
        step_m: float,
        margin_m: float = 0.0,
    ) -> tuple[bool, list[str]]:
        distance = norm(sub(end_pose[:3], start_pose[:3]))
        samples = max(1, int(math.ceil(distance / max(step_m, 1e-6))))
        for idx in range(samples + 1):
            ratio = idx / samples
            pose = [
                start_pose[col] + (end_pose[col] - start_pose[col]) * ratio
                for col in range(6)
            ]
            ok, reasons = self.contains_pose(pose, margin_m=margin_m)
            if not ok:
                return False, [f"segment sample {idx}/{samples}: {reason}" for reason in reasons]
        return True, []


def load_workspace(path: Path) -> Workspace:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return Workspace(data)


def write_workspace(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tool_pose_csv(csv_path: Path) -> list[list[float]]:
    poses: list[list[float]] = []
    with csv_path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("timestamp"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 7:
                continue
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            poses.append(values[1:7])
    if not poses:
        raise ValueError(f"No valid tool poses found in {csv_path}")
    return poses


def load_waypoint_csv(csv_path: Path) -> list[tuple[str, list[float]]]:
    waypoints: list[tuple[str, list[float]]] = []
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["label", "x", "y", "z", "rx", "ry", "rz"]
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"Waypoint CSV must contain columns: {', '.join(required)}")
        for row_idx, row in enumerate(reader, start=2):
            label = (row.get("label") or f"waypoint_{row_idx}").strip()
            try:
                pose = [float(row[name]) for name in required[1:]]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid waypoint row {row_idx}: {row}") from exc
            waypoints.append((label, pose))
    if not waypoints:
        raise ValueError(f"No valid waypoints found in {csv_path}")
    return waypoints


def min_max(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def expand_limits(limits: list[float], margin: float) -> list[float]:
    return [limits[0] - margin, limits[1] + margin]


def pose_bounds(poses: list[list[float]], position_margin_m: float, orientation_margin_rad: float) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    position_limits = {
        axis: expand_limits(min_max([pose[idx] for pose in poses]), position_margin_m)
        for idx, axis in enumerate(("x", "y", "z"))
    }
    orientation_limits = {
        axis: expand_limits(min_max([pose[idx + 3] for pose in poses]), orientation_margin_rad)
        for idx, axis in enumerate(("rx", "ry", "rz"))
    }
    return position_limits, orientation_limits


def make_world_workspace(
    poses: list[list[float]],
    source: str,
    position_margin_m: float,
    orientation_margin_rad: float,
    tool_clearance_m: float,
    path_check_step_m: float,
) -> dict[str, Any]:
    position_limits, orientation_limits = pose_bounds(poses, position_margin_m, orientation_margin_rad)
    return {
        "version": 1,
        "type": "oriented_box",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame": {
            "source": "world_from_csv",
            "origin": [0.0, 0.0, 0.0],
            "axes": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "position_limits": position_limits,
        "orientation_limits": orientation_limits,
        "safety": {
            "position_margin_m": position_margin_m,
            "orientation_margin_rad": orientation_margin_rad,
            "tool_clearance_m": tool_clearance_m,
            "path_check_step_m": path_check_step_m,
            "notes": "Check x/y/z for every model target and sampled path point. Z is mandatory.",
        },
        "metadata": {
            "source": source,
            "num_poses": len(poses),
            "coordinate_units": "m, rad",
        },
    }


def make_frame_from_points(origin_pose: list[float], x_pose: list[float], y_pose: list[float]) -> dict[str, Any]:
    origin = origin_pose[:3]
    x_axis = normalize(sub(x_pose[:3], origin), "P0 -> Px")
    raw_y = sub(y_pose[:3], origin)
    y_without_x = sub(raw_y, mul(x_axis, dot(raw_y, x_axis)))
    y_axis = normalize(y_without_x, "P0 -> Py projected perpendicular to X")
    z_axis = normalize(cross(x_axis, y_axis), "X cross Y")
    y_axis = normalize(cross(z_axis, x_axis), "orthogonalized Y")
    return {
        "source": "taught_from_tool_pos",
        "origin": origin,
        "axes": [x_axis, y_axis, z_axis],
        "teach_points": {
            "origin_p0": origin_pose,
            "x_direction_px": x_pose,
            "y_direction_py": y_pose,
        },
    }


def make_taught_workspace(
    frame: dict[str, Any],
    boundary_poses: list[list[float]],
    position_margin_m: float,
    orientation_margin_rad: float,
    tool_clearance_m: float,
    path_check_step_m: float,
) -> dict[str, Any]:
    workspace = Workspace(
        {
            "frame": frame,
            "position_limits": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "orientation_limits": {"rx": [0.0, 0.0], "ry": [0.0, 0.0], "rz": [0.0, 0.0]},
        }
    )
    local_points = [workspace.world_to_local(pose[:3]) for pose in boundary_poses]
    position_limits = {
        axis: expand_limits(min_max([point[idx] for point in local_points]), position_margin_m)
        for idx, axis in enumerate(("x", "y", "z"))
    }
    orientation_limits = {
        axis: expand_limits(min_max([pose[idx + 3] for pose in boundary_poses]), orientation_margin_rad)
        for idx, axis in enumerate(("rx", "ry", "rz"))
    }
    return {
        "version": 1,
        "type": "oriented_box",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame": frame,
        "position_limits": position_limits,
        "orientation_limits": orientation_limits,
        "safety": {
            "position_margin_m": position_margin_m,
            "orientation_margin_rad": orientation_margin_rad,
            "tool_clearance_m": tool_clearance_m,
            "path_check_step_m": path_check_step_m,
            "notes": "Taught local box. Keep z range deliberately bounded.",
        },
        "metadata": {
            "source": "/tool_pos teach",
            "num_boundary_poses": len(boundary_poses),
            "coordinate_units": "m, rad",
        },
    }


def format_pose(pose: list[float]) -> str:
    return (
        f"x={pose[0]:.6f}, y={pose[1]:.6f}, z={pose[2]:.6f}, "
        f"rx={pose[3]:.6f}, ry={pose[4]:.6f}, rz={pose[5]:.6f}"
    )


def summarize_workspace(data: dict[str, Any]) -> str:
    pos = data["position_limits"]
    ori = data["orientation_limits"]
    return "\n".join(
        [
            "Position limits in workspace frame:",
            f"  x: {pos['x'][0]:.6f} .. {pos['x'][1]:.6f} m",
            f"  y: {pos['y'][0]:.6f} .. {pos['y'][1]:.6f} m",
            f"  z: {pos['z'][0]:.6f} .. {pos['z'][1]:.6f} m",
            "Orientation limits:",
            f"  rx: {ori['rx'][0]:.6f} .. {ori['rx'][1]:.6f} rad",
            f"  ry: {ori['ry'][0]:.6f} .. {ori['ry'][1]:.6f} rad",
            f"  rz: {ori['rz'][0]:.6f} .. {ori['rz'][1]:.6f} rad",
        ]
    )


def workspace_size_lines(workspace: Workspace, margin_m: float = 0.0) -> list[str]:
    raw = workspace.position_limits
    effective = workspace.effective_position_limits(margin_m=margin_m)
    lines = [
        "Workspace size:",
    ]
    volume = 1.0
    effective_volume = 1.0
    for axis_name in ("x", "y", "z"):
        low, high = [float(v) for v in raw[axis_name]]
        eff_low, eff_high = effective[axis_name]
        width = high - low
        eff_width = max(0.0, eff_high - eff_low)
        volume *= max(0.0, width)
        effective_volume *= eff_width
        lines.append(
            f"  {axis_name}: raw={width * 1000.0:.1f} mm, "
            f"effective={eff_width * 1000.0:.1f} mm "
            f"({eff_low:.6f} .. {eff_high:.6f} m local)"
        )
    lines.append(f"  raw volume: {volume * 1e6:.1f} cm^3")
    lines.append(f"  effective volume: {effective_volume * 1e6:.1f} cm^3")
    clearance = workspace.position_clearance_m
    lines.append(
        "  position clearance: "
        f"x={clearance['x'] * 1000.0:.1f} mm, "
        f"y={clearance['y'] * 1000.0:.1f} mm, "
        f"z={clearance['z'] * 1000.0:.1f} mm"
    )
    lines.append(f"  orientation check: {'enabled' if workspace.check_orientation else 'disabled'}")
    return lines


def midpoint(limits: dict[str, list[float]]) -> list[float]:
    return [(limits[axis][0] + limits[axis][1]) / 2.0 for axis in ("x", "y", "z")]


def pose_from_local(workspace: Workspace, local_xyz: list[float]) -> list[float]:
    orientation = [
        (float(workspace.orientation_limits[axis][0]) + float(workspace.orientation_limits[axis][1])) / 2.0
        for axis in ("rx", "ry", "rz")
    ]
    return workspace.local_to_world(local_xyz) + orientation


def probe_workspace_points(workspace: Workspace, outside_offset_m: float) -> list[tuple[str, list[float], bool]]:
    limits = workspace.effective_position_limits()
    center = midpoint(limits)
    probes: list[tuple[str, list[float], bool]] = [("center", center, True)]
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        low_point = list(center)
        low_point[axis_idx] = limits[axis_name][0]
        high_point = list(center)
        high_point[axis_idx] = limits[axis_name][1]
        below_point = list(center)
        below_point[axis_idx] = limits[axis_name][0] - outside_offset_m
        above_point = list(center)
        above_point[axis_idx] = limits[axis_name][1] + outside_offset_m
        probes.extend(
            [
                (f"{axis_name}_min_inside", low_point, True),
                (f"{axis_name}_max_inside", high_point, True),
                (f"{axis_name}_below_outside", below_point, False),
                (f"{axis_name}_above_outside", above_point, False),
            ]
        )
    return probes


def inspection_waypoints(workspace: Workspace, inset_m: float) -> list[tuple[str, list[float]]]:
    limits = workspace.effective_position_limits(margin_m=inset_m)
    low = [limits[axis][0] for axis in ("x", "y", "z")]
    high = [limits[axis][1] for axis in ("x", "y", "z")]
    center = midpoint(limits)
    z = center[2]
    local_points = [
        ("center", center),
        ("xy_corner_1", [low[0], low[1], z]),
        ("xy_corner_2", [high[0], low[1], z]),
        ("xy_corner_3", [high[0], high[1], z]),
        ("xy_corner_4", [low[0], high[1], z]),
        ("z_min_center", [center[0], center[1], low[2]]),
        ("z_max_center", [center[0], center[1], high[2]]),
    ]
    return [(label, pose_from_local(workspace, local_xyz)) for label, local_xyz in local_points]


def sample_poses(node: Any, sample_sec: float) -> list[list[float]]:
    deadline = time.time() + sample_sec
    node.samples.clear()
    while time.time() < deadline:
        node.rclpy.spin_once(node, timeout_sec=0.05)
    return list(node.samples)


def average_pose(poses: list[list[float]]) -> list[float]:
    if not poses:
        raise RuntimeError("No /tool_pos samples received")
    return [sum(pose[idx] for pose in poses) / len(poses) for idx in range(6)]


def prompt_capture(node: Any, label: str, sample_sec: float) -> list[float]:
    input(f"\nMove TCP to {label}, then press Enter to record.")
    poses = sample_poses(node, sample_sec)
    pose = average_pose(poses)
    print(f"Recorded {label}: {format_pose(pose)}")
    return pose


def cmd_from_csv(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    poses = load_tool_pose_csv(csv_path)
    data = make_world_workspace(
        poses=poses,
        source=str(csv_path),
        position_margin_m=args.margin_mm / 1000.0,
        orientation_margin_rad=rad(args.rot_margin_deg),
        tool_clearance_m=args.tool_clearance_mm / 1000.0,
        path_check_step_m=args.path_step_mm / 1000.0,
    )
    write_workspace(Path(args.out), data)
    print(f"Wrote workspace: {args.out}")
    print(summarize_workspace(data))
    return 0


def cmd_check_csv(args: argparse.Namespace) -> int:
    workspace = load_workspace(Path(args.workspace))
    poses = load_tool_pose_csv(Path(args.csv))
    margin_m = args.extra_margin_mm / 1000.0
    step_m = (args.path_step_mm / 1000.0) if args.path_step_mm is not None else workspace.path_check_step_m
    pose_failures = 0
    segment_failures = 0
    first_failure: Optional[str] = None

    for idx, pose in enumerate(poses):
        if args.position_only:
            ok, reasons = workspace.contains_pose_raw_position(pose, tolerance_m=margin_m)
        else:
            ok, reasons = workspace.contains_pose(pose, margin_m=margin_m)
        if not ok:
            pose_failures += 1
            if first_failure is None:
                first_failure = f"pose {idx}: {reasons[0]}"

    for idx in range(len(poses) - 1):
        if args.position_only:
            ok, reasons = workspace.contains_segment_raw_position(
                poses[idx], poses[idx + 1], step_m=step_m, tolerance_m=margin_m
            )
        else:
            ok, reasons = workspace.contains_segment(poses[idx], poses[idx + 1], step_m=step_m, margin_m=margin_m)
        if not ok:
            segment_failures += 1
            if first_failure is None:
                first_failure = f"segment {idx}->{idx + 1}: {reasons[0]}"

    print(f"Checked poses: {len(poses)}")
    print(f"Pose failures: {pose_failures}")
    print(f"Segment failures: {segment_failures}")
    if first_failure:
        print(f"First failure: {first_failure}")
        return 2
    print("Workspace check passed.")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    workspace = load_workspace(Path(args.workspace))
    print(summarize_workspace(workspace.data))
    print()
    print("\n".join(workspace_size_lines(workspace, margin_m=args.extra_margin_mm / 1000.0)))
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    workspace = load_workspace(Path(args.workspace))
    outside_offset_m = args.outside_offset_mm / 1000.0
    failures = 0
    print("\n".join(workspace_size_lines(workspace)))
    print("\nProbe results:")
    for label, local_xyz, expected in probe_workspace_points(workspace, outside_offset_m):
        pose = pose_from_local(workspace, local_xyz)
        ok, reasons = workspace.contains_pose(pose)
        status = "PASS" if ok == expected else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"  {status} {label}: expected={'inside' if expected else 'outside'}, "
            f"actual={'inside' if ok else 'outside'}"
        )
        if reasons:
            print(f"    reason: {reasons[0]}")

    if failures:
        print(f"\nSelf-test failed: {failures} unexpected results.")
        return 2
    print("\nSelf-test passed.")
    return 0


def cmd_export_inspection_csv(args: argparse.Namespace) -> int:
    workspace = load_workspace(Path(args.workspace))
    waypoints = inspection_waypoints(workspace, inset_m=args.inset_mm / 1000.0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["label,x,y,z,rx,ry,rz"]
    for label, pose in waypoints:
        lines.append(label + "," + ",".join(f"{value:.9f}" for value in pose))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote inspection waypoints: {out_path}")
    print("These are dry-run waypoints only. Move the robot through them only with a separate low-speed controller and manual supervision.")
    return 0


def validate_waypoint_path(
    workspace: Workspace,
    waypoints: list[tuple[str, list[float]]],
    start_pose: Optional[list[float]],
    allow_entry_from_raw: bool = False,
    entry_tolerance_m: float = 0.0,
) -> None:
    previous_label = "current_pose"
    previous_pose = start_pose
    for label, pose in waypoints:
        ok, reasons = workspace.contains_pose(pose)
        if not ok:
            raise RuntimeError(f"Waypoint {label} is outside workspace: {reasons[0]}")
        if previous_pose is not None:
            ok, reasons = workspace.contains_segment(
                previous_pose,
                pose,
                step_m=workspace.path_check_step_m,
            )
            if not ok:
                if allow_entry_from_raw and previous_label == "current_pose":
                    start_raw_ok, start_raw_reasons = workspace.contains_pose_raw_position(
                        previous_pose,
                        tolerance_m=entry_tolerance_m,
                    )
                    segment_raw_ok, segment_raw_reasons = workspace.contains_segment_raw_position(
                        previous_pose,
                        pose,
                        step_m=workspace.path_check_step_m,
                        tolerance_m=entry_tolerance_m,
                    )
                    if start_raw_ok and segment_raw_ok:
                        print(
                            "WARNING: current pose is outside the effective workspace "
                            "but inside the raw taught workspace tolerance; allowing first entry move."
                        )
                    else:
                        reason = start_raw_reasons[0] if not start_raw_ok else segment_raw_reasons[0]
                        raise RuntimeError(f"Entry segment {previous_label}->{label} leaves raw workspace: {reason}")
                else:
                    raise RuntimeError(f"Segment {previous_label}->{label} leaves workspace: {reasons[0]}")
        previous_label = label
        previous_pose = pose


class InspectionExecutor:
    def __init__(self, service_name: str, pose_topic: str, timeout_sec: float):
        bootstrap_local_ros_paths()
        import rclpy
        from common_interface.msg import TcpPos
        from common_interface.srv import Move
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.Move = Move
        self.node = Node("workspace_inspection_runner")
        self.latest_pose: Optional[list[float]] = None
        self.node.create_subscription(TcpPos, pose_topic, self._on_pose, 10)
        self.move_client = self.node.create_client(Move, service_name)
        if not self.move_client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"Move service unavailable: {service_name}")

    def _on_pose(self, msg: Any) -> None:
        self.latest_pose = [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz]

    def get_current_pose(self, timeout_sec: float) -> list[float]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.latest_pose is not None:
                return list(self.latest_pose)
        raise RuntimeError("No current pose received from /tool_pos")

    def move_absolute(self, pose: list[float], timeout_sec: float) -> bool:
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
            self.node.get_logger().error(f"Move failed: {future.exception()}")
            return False
        return True

    def shutdown(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def cmd_run_inspection(args: argparse.Namespace) -> int:
    workspace = load_workspace(Path(args.workspace))
    waypoints = load_waypoint_csv(Path(args.waypoints))
    print(f"Loaded {len(waypoints)} inspection waypoints.")

    executor: Optional[InspectionExecutor] = None
    current_pose: Optional[list[float]] = None
    if args.execute or not args.skip_current_check:
        executor = InspectionExecutor(args.service, args.pose_topic, args.timeout_sec)
        current_pose = executor.get_current_pose(args.pose_timeout_sec)
        print(f"Current pose: {format_pose(current_pose)}")

    try:
        validate_waypoint_path(
            workspace,
            waypoints,
            start_pose=None if args.skip_current_check else current_pose,
            allow_entry_from_raw=args.allow_entry_from_raw,
            entry_tolerance_m=args.entry_tolerance_mm / 1000.0,
        )
        print("Workspace path validation passed.")

        for idx, (label, pose) in enumerate(waypoints, start=1):
            print(f"[{idx}/{len(waypoints)}] {label}: {format_pose(pose)}")

        if not args.execute:
            print("Dry-run complete: no robot motion command was sent. Add --execute to move.")
            return 0

        print("\nWARNING: This will move the robot through the listed waypoints.")
        print("Use the lowest available controller speed, keep clear of the robot, and keep emergency stop ready.")
        if not args.yes:
            answer = input("Type RUN to start inspection motion: ").strip()
            if answer != "RUN":
                print("Aborted.")
                return 1

        if executor is None:
            executor = InspectionExecutor(args.service, args.pose_topic, args.timeout_sec)
        previous_pose = current_pose
        for idx, (label, pose) in enumerate(waypoints, start=1):
            if previous_pose is not None:
                ok, reasons = workspace.contains_segment(previous_pose, pose, workspace.path_check_step_m)
                if not ok:
                    if args.allow_entry_from_raw and idx == 1:
                        start_raw_ok, start_raw_reasons = workspace.contains_pose_raw_position(
                            previous_pose,
                            tolerance_m=args.entry_tolerance_mm / 1000.0,
                        )
                        segment_raw_ok, segment_raw_reasons = workspace.contains_segment_raw_position(
                            previous_pose,
                            pose,
                            workspace.path_check_step_m,
                            tolerance_m=args.entry_tolerance_mm / 1000.0,
                        )
                        if not start_raw_ok or not segment_raw_ok:
                            reason = start_raw_reasons[0] if not start_raw_ok else segment_raw_reasons[0]
                            raise RuntimeError(f"Live entry segment failed before {label}: {reason}")
                        print("WARNING: first move enters the effective workspace from the raw taught boundary.")
                    else:
                        raise RuntimeError(f"Live segment check failed before {label}: {reasons[0]}")
            print(f"\n[{idx}/{len(waypoints)}] Move to {label}: {format_pose(pose)}")
            if not args.yes:
                answer = input("Press Enter to move, or type q to stop: ").strip().lower()
                if answer in {"q", "quit", "stop"}:
                    print("Stopped by user.")
                    return 1
            if not executor.move_absolute(pose, args.timeout_sec):
                return 2
            if args.settle_sec > 0:
                time.sleep(args.settle_sec)
            previous_pose = executor.get_current_pose(args.pose_timeout_sec)
    finally:
        if executor is not None:
            executor.shutdown()
    return 0


def cmd_serve_safe_move(args: argparse.Namespace) -> int:
    bootstrap_local_ros_paths()
    import rclpy
    from common_interface.msg import TcpPos
    from common_interface.srv import Move
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    class SafeMoveProxyNode(Node):
        def __init__(self) -> None:
            super().__init__("workspace_safe_move_proxy")
            self.workspace = load_workspace(Path(args.workspace))
            self.latest_pose: Optional[list[float]] = None
            self.callback_group = ReentrantCallbackGroup()
            self.create_subscription(TcpPos, args.pose_topic, self._on_pose, 10)
            self.move_client = self.create_client(
                Move,
                args.real_service,
                callback_group=self.callback_group,
            )
            self.safe_service = self.create_service(
                Move,
                args.safe_service,
                self._on_safe_move,
                callback_group=self.callback_group,
            )

        def _on_pose(self, msg: Any) -> None:
            self.latest_pose = [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz]

        def _reject(self, reason: str, response: Any) -> Any:
            self.get_logger().error(f"拒绝安全移动: {reason}")
            return response

        def _validate_target(self, target: list[float]) -> tuple[bool, str]:
            if self.latest_pose is None:
                return False, f"尚未收到当前位姿 {args.pose_topic}"
            ok, reasons = self.workspace.contains_pose(target)
            if not ok:
                return False, reasons[0]
            step_m = norm(sub(target[:3], self.latest_pose[:3]))
            if not args.allow_large_steps and step_m > args.max_step_m:
                return False, f"单步位移过大: {step_m * 1000.0:.2f} mm > {args.max_step_m * 1000.0:.2f} mm"
            ok, reasons = self.workspace.contains_segment(
                self.latest_pose,
                target,
                step_m=self.workspace.path_check_step_m,
            )
            if not ok:
                return False, reasons[0]
            return True, ""

        def _forward_to_robot(self, request: Any) -> bool:
            if not self.move_client.service_is_ready():
                if not self.move_client.wait_for_service(timeout_sec=args.service_timeout_sec):
                    self.get_logger().error(f"真实运动服务不可用: {args.real_service}")
                    return False
            future = self.move_client.call_async(request)
            done = threading.Event()
            future.add_done_callback(lambda _: done.set())
            if not done.wait(timeout=args.move_timeout_sec):
                self.get_logger().error(f"真实运动服务超时: {args.real_service}")
                return False
            if future.result() is None:
                self.get_logger().error(f"真实运动服务调用失败: {future.exception()}")
                return False
            return True

        def _on_safe_move(self, request: Any, response: Any) -> Any:
            target = [
                float(request.a),
                float(request.b),
                float(request.c),
                float(request.d),
                float(request.e),
                float(request.f),
            ]
            ok, reason = self._validate_target(target)
            if not ok:
                return self._reject(reason, response)

            self.get_logger().info(f"安全移动通过: {format_pose(target)}")
            if args.dry_run:
                self.get_logger().info("dry-run: 未转发到真实 /mov_jog")
                return response
            self._forward_to_robot(request)
            return response

    if not rclpy.ok():
        rclpy.init()
    node = SafeMoveProxyNode()
    try:
        if not node.move_client.wait_for_service(timeout_sec=args.service_timeout_sec):
            raise RuntimeError(f"真实运动服务不可用: {args.real_service}")
        node.get_logger().info(f"安全服务已启动: {args.safe_service} -> {args.real_service}")
        node.get_logger().info(f"workspace: {args.workspace}")
        node.get_logger().info(
            f"max_step={args.max_step_m * 1000.0:.1f} mm, "
            f"allow_large_steps={args.allow_large_steps}, dry_run={args.dry_run}"
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.remove_node(node)
            executor.shutdown()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def cmd_serve_zmq_filter(args: argparse.Namespace) -> int:
    bootstrap_local_ros_paths()
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("缺少 pyzmq，请在 ROS Python 环境安装 pyzmq 后再运行 serve-zmq-filter") from exc

    import rclpy
    from common_interface.msg import TcpPos
    from common_interface.srv import Move
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    class ZmqSafetyFilterNode(Node):
        def __init__(self) -> None:
            super().__init__("workspace_zmq_safety_filter")
            self.workspace = load_workspace(Path(args.workspace))
            self.latest_pose: Optional[list[float]] = None
            self.in_flight = False
            self.accepted_count = 0
            self.rejected_count = 0
            self.dropped_count = 0
            self.last_seq: Optional[int] = None
            self.last_run_id: Optional[int] = None

            self.callback_group = ReentrantCallbackGroup()
            self.create_subscription(TcpPos, args.pose_topic, self._on_pose, 10)
            self.move_client = self.create_client(
                Move,
                args.real_service,
                callback_group=self.callback_group,
            )

            self.zmq_context = zmq.Context.instance()
            self.zmq_socket = self.zmq_context.socket(zmq.PULL)
            self.zmq_socket.setsockopt(zmq.RCVHWM, 1)
            self.zmq_socket.setsockopt(zmq.CONFLATE, 1)
            self.zmq_socket.bind(args.zmq_bind)
            self.create_timer(args.poll_period_sec, self._poll_zmq)
            self.create_timer(args.status_period_sec, self._log_status)

        def _on_pose(self, msg: Any) -> None:
            self.latest_pose = [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz]

        def _reject(self, code: str, detail: str) -> None:
            self.rejected_count += 1
            self.get_logger().warning(f"丢弃 infer 目标: {code}: {detail}")

        def _parse_message(self, message: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[float], Optional[list[float]], Optional[str]]:
            run_id_value = message.get("run_id")
            run_id = int(run_id_value) if run_id_value is not None else None
            seq_value = message.get("seq")
            seq = int(seq_value) if seq_value is not None else None
            timestamp_value = message.get("timestamp")
            timestamp = float(timestamp_value) if timestamp_value is not None else None
            pose_raw = message.get("pose")
            if not isinstance(pose_raw, list) or len(pose_raw) != 6:
                return run_id, seq, timestamp, None, "pose must be a 6-value list"
            try:
                pose = [float(value) for value in pose_raw]
            except (TypeError, ValueError) as exc:
                return run_id, seq, timestamp, None, f"pose contains non-float values: {exc}"
            return run_id, seq, timestamp, pose, None

        def _begin_infer_run(self, run_id: Optional[int], seq: Optional[int]) -> None:
            if run_id is not None:
                if run_id != self.last_run_id:
                    self.last_run_id = run_id
                    self.last_seq = None
                    self.get_logger().info(f"检测到新 infer 会话: run_id={run_id}")
                return
            if seq == 1 and self.last_seq is not None:
                self.last_seq = None
                self.get_logger().info("检测到 seq 从 1 重新开始，重置 infer 会话")

        def _is_stale(self, timestamp: Optional[float]) -> bool:
            if timestamp is None:
                return bool(args.require_timestamp)
            return (time.time() - timestamp) > args.max_target_age_sec

        def _validate_target(
            self,
            run_id: Optional[int],
            seq: Optional[int],
            timestamp: Optional[float],
            target: list[float],
        ) -> tuple[bool, str, str]:
            self._begin_infer_run(run_id, seq)
            if seq is not None and self.last_seq is not None and seq <= self.last_seq:
                return False, "OLD_SEQ", f"seq={seq} <= last_seq={self.last_seq}"
            if self._is_stale(timestamp):
                age = float("nan") if timestamp is None else time.time() - timestamp
                return False, "STALE_TARGET", f"age={age:.3f}s > {args.max_target_age_sec:.3f}s"
            if self.latest_pose is None:
                return False, "NO_CURRENT_POSE", f"尚未收到 {args.pose_topic}"
            ok, reasons = self.workspace.contains_pose(target)
            if not ok:
                return False, "OUT_OF_WORKSPACE", reasons[0]
            step_m = norm(sub(target[:3], self.latest_pose[:3]))
            if not args.allow_large_steps and step_m > args.max_step_m:
                return False, "STEP_TOO_LARGE", f"{step_m * 1000.0:.2f} mm > {args.max_step_m * 1000.0:.2f} mm"
            ok, reasons = self.workspace.contains_segment(
                self.latest_pose,
                target,
                step_m=self.workspace.path_check_step_m,
            )
            if not ok:
                return False, "PATH_OUT_OF_WORKSPACE", reasons[0]
            return True, "OK", ""

        def _forward_to_robot(self, target: list[float]) -> None:
            if self.in_flight:
                self.dropped_count += 1
                self.get_logger().warning("上一条 /mov_jog 尚未完成，丢弃当前合法目标以避免命令积压")
                return
            if not self.move_client.service_is_ready():
                if not self.move_client.wait_for_service(timeout_sec=args.service_timeout_sec):
                    self._reject("MOVE_SERVICE_UNAVAILABLE", args.real_service)
                    return

            request = Move.Request()
            request.a = float(target[0])
            request.b = float(target[1])
            request.c = float(target[2])
            request.d = float(target[3])
            request.e = float(target[4])
            request.f = float(target[5])
            request.block = bool(args.block)
            request.name = ""

            if args.dry_run:
                self.accepted_count += 1
                self.get_logger().info(f"dry-run 安全通过: {format_pose(target)}")
                return

            self.in_flight = True
            future = self.move_client.call_async(request)

            def on_done(done_future: Any) -> None:
                self.in_flight = False
                if done_future.result() is None:
                    self._reject("MOVE_FAILED", str(done_future.exception()))
                    return
                self.accepted_count += 1
                self.get_logger().info(f"已转发安全目标: {format_pose(target)}")

            future.add_done_callback(on_done)

        def _poll_zmq(self) -> None:
            latest_message: Optional[dict[str, Any]] = None
            drained = 0
            while True:
                try:
                    latest_message = self.zmq_socket.recv_json(flags=zmq.NOBLOCK)
                    drained += 1
                except zmq.Again:
                    break
                except ValueError as exc:
                    self._reject("BAD_JSON", str(exc))
                    break

            if latest_message is None:
                return
            if drained > 1:
                self.dropped_count += drained - 1

            run_id, seq, timestamp, target, parse_error = self._parse_message(latest_message)
            if parse_error is not None or target is None:
                self._reject("INVALID_MESSAGE", parse_error or "unknown parse error")
                return

            ok, code, detail = self._validate_target(run_id, seq, timestamp, target)
            if not ok:
                self._reject(code, detail)
                return

            if seq is not None:
                self.last_seq = seq
            self._forward_to_robot(target)

        def _log_status(self) -> None:
            self.get_logger().info(
                "ZMQ安全过滤状态: "
                f"accepted={self.accepted_count}, rejected={self.rejected_count}, "
                f"dropped={self.dropped_count}, last_seq={self.last_seq}"
            )

        def destroy_node(self) -> bool:
            self.zmq_socket.close(linger=0)
            return super().destroy_node()

    if not rclpy.ok():
        rclpy.init()
    node = ZmqSafetyFilterNode()
    try:
        if not node.move_client.wait_for_service(timeout_sec=args.service_timeout_sec):
            raise RuntimeError(f"真实运动服务不可用: {args.real_service}")
        node.get_logger().info(f"ZMQ安全过滤已启动: {args.zmq_bind} -> {args.real_service}")
        node.get_logger().info(f"workspace: {args.workspace}")
        node.get_logger().info(
            f"max_age={args.max_target_age_sec:.3f}s, max_step={args.max_step_m * 1000.0:.1f}mm, "
            f"poll={args.poll_period_sec:.3f}s, dry_run={args.dry_run}"
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.remove_node(node)
            executor.shutdown()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def cmd_teach(args: argparse.Namespace) -> int:
    bootstrap_local_ros_paths()
    import rclpy
    from common_interface.msg import TcpPos
    from rclpy.node import Node

    class ToolPoseSampler(Node):
        def __init__(self) -> None:
            super().__init__("workspace_safety_teach")
            self.rclpy = rclpy
            self.samples: list[list[float]] = []
            self.create_subscription(TcpPos, args.topic, self._on_pose, 10)

        def _on_pose(self, msg: Any) -> None:
            self.samples.append([msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz])

    rclpy.init()
    node = ToolPoseSampler()
    try:
        print("Teach local workspace frame from /tool_pos.")
        print("P0 is the workspace origin. Px defines +X. Py defines the +Y side of the work plane.")
        origin_pose = prompt_capture(node, "P0 origin", args.sample_sec)
        x_pose = prompt_capture(node, "Px on +X direction", args.sample_sec)
        y_pose = prompt_capture(node, "Py on +Y direction", args.sample_sec)
        frame = make_frame_from_points(origin_pose, x_pose, y_pose)

        print("\nNow record boundary poses. Include low-Z and high-Z limits; XY alone is not enough.")
        print("Record at least the corners or face centers of the allowed 3D volume. Type 'done' when finished.")
        boundary_poses: list[list[float]] = [origin_pose, x_pose, y_pose]
        while True:
            value = input(f"Boundary #{len(boundary_poses) + 1}, Enter=record, done=finish: ").strip().lower()
            if value in {"done", "d", "q", "quit"}:
                break
            poses = sample_poses(node, args.sample_sec)
            pose = average_pose(poses)
            boundary_poses.append(pose)
            print(f"Recorded boundary #{len(boundary_poses)}: {format_pose(pose)}")

        if len(boundary_poses) < args.min_boundary_points:
            raise RuntimeError(
                f"Need at least {args.min_boundary_points} boundary poses, got {len(boundary_poses)}"
            )

        data = make_taught_workspace(
            frame=frame,
            boundary_poses=boundary_poses,
            position_margin_m=args.margin_mm / 1000.0,
            orientation_margin_rad=rad(args.rot_margin_deg),
            tool_clearance_m=args.tool_clearance_mm / 1000.0,
            path_check_step_m=args.path_step_mm / 1000.0,
        )
        z_low, z_high = data["position_limits"]["z"]
        if z_high - z_low < args.min_z_range_mm / 1000.0:
            print(
                f"WARNING: taught z range is only {(z_high - z_low) * 1000.0:.2f} mm. "
                "Record explicit low-Z and high-Z boundaries."
            )
        write_workspace(Path(args.out), data)
        print(f"\nWrote workspace: {args.out}")
        print(summarize_workspace(data))
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and check TCP workspace limits")
    sub = parser.add_subparsers(dest="command", required=True)

    from_csv = sub.add_parser("from-csv", help="Build initial workspace limits from tool_pose.csv")
    from_csv.add_argument("--csv", required=True)
    from_csv.add_argument("--out", default=str(SCRIPT_DIR / "workspace_limits.json"))
    from_csv.add_argument("--margin-mm", type=float, default=10.0)
    from_csv.add_argument("--rot-margin-deg", type=float, default=2.0)
    from_csv.add_argument("--tool-clearance-mm", type=float, default=0.0)
    from_csv.add_argument("--path-step-mm", type=float, default=2.0)
    from_csv.set_defaults(func=cmd_from_csv)

    teach = sub.add_parser("teach", help="Teach local workspace frame and bounds from /tool_pos")
    teach.add_argument("--out", default=str(SCRIPT_DIR / "workspace_limits.json"))
    teach.add_argument("--topic", default="/tool_pos")
    teach.add_argument("--sample-sec", type=float, default=0.3)
    teach.add_argument("--margin-mm", type=float, default=5.0)
    teach.add_argument("--rot-margin-deg", type=float, default=2.0)
    teach.add_argument("--tool-clearance-mm", type=float, default=0.0)
    teach.add_argument("--path-step-mm", type=float, default=2.0)
    teach.add_argument("--min-boundary-points", type=int, default=6)
    teach.add_argument("--min-z-range-mm", type=float, default=5.0)
    teach.set_defaults(func=cmd_teach)

    check_csv = sub.add_parser("check-csv", help="Check tool_pose.csv against workspace limits")
    check_csv.add_argument("--workspace", default=str(SCRIPT_DIR / "workspace_limits.json"))
    check_csv.add_argument("--csv", required=True)
    check_csv.add_argument("--path-step-mm", type=float, default=None)
    check_csv.add_argument("--extra-margin-mm", type=float, default=0.0)
    check_csv.add_argument("--position-only", action="store_true", help="Only check x/y/z raw workspace bounds; ignore orientation and tool clearance")
    check_csv.set_defaults(func=cmd_check_csv)

    describe = sub.add_parser("describe", help="Print workspace size and effective limits")
    describe.add_argument("--workspace", default=str(SCRIPT_DIR / "workspace_limits.json"))
    describe.add_argument("--extra-margin-mm", type=float, default=0.0)
    describe.set_defaults(func=cmd_describe)

    self_test = sub.add_parser("self-test", help="Probe inside and outside points without moving the robot")
    self_test.add_argument("--workspace", default=str(SCRIPT_DIR / "workspace_limits.json"))
    self_test.add_argument("--outside-offset-mm", type=float, default=5.0)
    self_test.set_defaults(func=cmd_self_test)

    export_inspection = sub.add_parser("export-inspection-csv", help="Export inner-box inspection waypoints")
    export_inspection.add_argument("--workspace", default=str(SCRIPT_DIR / "workspace_limits.json"))
    export_inspection.add_argument("--out", default=str(SCRIPT_DIR / "workspace_inspection_waypoints.csv"))
    export_inspection.add_argument("--inset-mm", type=float, default=20.0)
    export_inspection.set_defaults(func=cmd_export_inspection_csv)

    run_inspection = sub.add_parser("run-inspection", help="Dry-run or execute workspace inspection waypoints")
    run_inspection.add_argument("--workspace", default=str(SCRIPT_DIR / "workspace_limits.json"))
    run_inspection.add_argument("--waypoints", default=str(SCRIPT_DIR / "workspace_inspection_waypoints.csv"))
    run_inspection.add_argument("--service", default="/mov_jog")
    run_inspection.add_argument("--pose-topic", default="/tool_pos")
    run_inspection.add_argument("--timeout-sec", type=float, default=30.0)
    run_inspection.add_argument("--pose-timeout-sec", type=float, default=3.0)
    run_inspection.add_argument("--settle-sec", type=float, default=0.5)
    run_inspection.add_argument("--skip-current-check", action="store_true")
    run_inspection.add_argument("--allow-entry-from-raw", action="store_true")
    run_inspection.add_argument("--entry-tolerance-mm", type=float, default=1.0)
    run_inspection.add_argument("--execute", action="store_true")
    run_inspection.add_argument("--yes", action="store_true", help="Do not prompt before each move")
    run_inspection.set_defaults(func=cmd_run_inspection)

    serve_safe = sub.add_parser("serve-safe-move", help="Run a safe Move proxy service before /mov_jog")
    serve_safe.add_argument("--workspace", default=str(SCRIPT_DIR / "workspace_limits.json"))
    serve_safe.add_argument("--pose-topic", default="/tool_pos")
    serve_safe.add_argument("--safe-service", default="/safe_mov_jog")
    serve_safe.add_argument("--real-service", default="/mov_jog")
    serve_safe.add_argument("--service-timeout-sec", type=float, default=10.0)
    serve_safe.add_argument("--move-timeout-sec", type=float, default=30.0)
    serve_safe.add_argument("--max-step-m", type=float, default=0.01)
    serve_safe.add_argument("--allow-large-steps", action="store_true")
    serve_safe.add_argument("--dry-run", action="store_true", help="Validate requests but do not call the real move service")
    serve_safe.set_defaults(func=cmd_serve_safe_move)

    serve_zmq = sub.add_parser("serve-zmq-filter", help="Run a ZMQ PULL safety filter for infer target poses")
    serve_zmq.add_argument("--workspace", default=str(SCRIPT_DIR / "workspace_limits.json"))
    serve_zmq.add_argument("--pose-topic", default="/tool_pos")
    serve_zmq.add_argument("--real-service", default="/mov_jog")
    serve_zmq.add_argument("--zmq-bind", default="tcp://127.0.0.1:5555")
    serve_zmq.add_argument("--poll-period-sec", type=float, default=0.02)
    serve_zmq.add_argument("--status-period-sec", type=float, default=5.0)
    serve_zmq.add_argument("--max-target-age-sec", type=float, default=0.5)
    serve_zmq.add_argument("--require-timestamp", action="store_true")
    serve_zmq.add_argument("--service-timeout-sec", type=float, default=10.0)
    serve_zmq.add_argument("--max-step-m", type=float, default=0.01)
    serve_zmq.add_argument("--allow-large-steps", action="store_true")
    serve_zmq.add_argument("--block", action="store_true")
    serve_zmq.add_argument("--dry-run", action="store_true", help="Validate requests but do not call the real move service")
    serve_zmq.set_defaults(func=cmd_serve_zmq_filter)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
