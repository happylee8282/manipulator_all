#!/usr/bin/env python3
"""Generate a robot-profile TCP path from Step 2 geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..configuration import DEFAULT_CONFIG, load_config, output_path, resolve_path
from ..control.common.geometry import (
    continuous_quaternions,
    cumulative_distance,
    quaternion_step_deg,
    resample_corresponding_geometry,
    surface_frames,
    transform_points,
)
from .clearance_geometry import surface_offset_directions
from .path_profile import surface_clearance_profile_mm


REQUIRED_COLUMNS = (
    "global_x_mm", "global_y_mm", "global_z_mm",
    "surface_center_x_mm", "surface_center_y_mm", "surface_center_z_mm",
    "outer_x_mm", "outer_y_mm", "outer_z_mm",
    "inner_x_mm", "inner_y_mm", "inner_z_mm",
)


def read_step2_csv(path: Path) -> dict[str, np.ndarray]:
    """Read centre and paired boundary geometry in Step 2 model-local mm."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"Step 2 CSV needs at least two data rows: {path}")
    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"Step 2 CSV is missing columns: {missing}")

    def xyz(names: Sequence[str]) -> np.ndarray:
        return np.asarray([[float(row[name]) for name in names] for row in rows], float)

    result = {
        "global_path_mm": xyz(REQUIRED_COLUMNS[0:3]),
        "surface_center_mm": xyz(REQUIRED_COLUMNS[3:6]),
        "outer_edge_mm": xyz(REQUIRED_COLUMNS[6:9]),
        "inner_edge_mm": xyz(REQUIRED_COLUMNS[9:12]),
    }
    for key, values in result.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} contains NaN or Inf")
    return result


def read_pcd_xyz(path: Path) -> np.ndarray:
    """Read XYZ from an ASCII or uncompressed binary PCD file."""

    with path.open("rb") as stream:
        lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD has no DATA line: {path}")
            decoded = line.decode("ascii").strip()
            lines.append(decoded)
            if decoded.upper().startswith("DATA "):
                mode = decoded.split(maxsplit=1)[1].lower()
                offset = stream.tell()
                break
    header: dict[str, list[str]] = {}
    for line in lines:
        if line and not line.startswith("#"):
            parts = line.split()
            header[parts[0].upper()] = parts[1:]
    fields = header.get("FIELDS", header.get("FIELD", []))
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PCD must contain x, y, z")
    if mode == "ascii":
        data = np.loadtxt(path, skiprows=len(lines), ndmin=2)
        return np.column_stack([data[:, fields.index(axis)] for axis in "xyz"])
    if mode != "binary":
        raise ValueError(f"unsupported PCD mode: {mode}")
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    if any(count != 1 for count in counts):
        raise ValueError("PCD COUNT other than one is unsupported")
    scalars = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
        ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4",
    }
    dtype = np.dtype(
        [(name, scalars[(kind.upper(), size)]) for name, kind, size in zip(fields, types, sizes)]
    )
    count = int(header.get("POINTS", header["WIDTH"])[0])
    records = np.fromfile(path, dtype=dtype, count=count, offset=offset)
    return np.column_stack([records[axis].astype(float) for axis in "xyz"])


def _linear_section(start: np.ndarray, end: np.ndarray, resolution_mm: float) -> np.ndarray:
    distance_mm = float(np.linalg.norm(end - start) * 1000.0)
    intervals = max(1, int(math.ceil(distance_mm / resolution_mm - 1.0e-12)))
    return np.linspace(start, end, intervals + 1)


def build_complete_trajectory(
    bonding_position: np.ndarray,
    surface: np.ndarray,
    quaternions: np.ndarray,
    tangent: np.ndarray,
    width: np.ndarray,
    tool_z: np.ndarray,
    settings: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Attach collision-safe approach, explicit endpoint dwell, and retreat."""

    approach_distance = float(settings["approach_distance_mm"]) * 1.0e-3
    retreat_distance = float(settings["retreat_distance_mm"]) * 1.0e-3
    resolution = float(settings["approach_resolution_mm"])
    approach = _linear_section(
        bonding_position[0] - approach_distance * tool_z[0], bonding_position[0], resolution
    )[:-1]
    retreat = _linear_section(
        bonding_position[-1], bonding_position[-1] - retreat_distance * tool_z[-1], resolution
    )[1:]
    positions = np.vstack((approach, bonding_position[:1], bonding_position, bonding_position[-1:], retreat))
    phase = np.asarray(
        ["approach"] * len(approach)
        + ["pre_bond_settle"]
        + ["bonding"] * len(bonding_position)
        + ["post_bond_settle"]
        + ["retreat"] * len(retreat)
    )
    source_index = np.r_[
        np.zeros(len(approach) + 1, dtype=int),
        np.arange(len(bonding_position)),
        np.full(len(retreat) + 1, len(bonding_position) - 1, dtype=int),
    ]

    def extend(values: np.ndarray) -> np.ndarray:
        return np.vstack(
            (
                np.repeat(values[:1], len(approach) + 1, axis=0),
                values,
                np.repeat(values[-1:], len(retreat) + 1, axis=0),
            )
        )

    return {
        "positions_m": positions,
        "quaternions_xyzw": extend(quaternions),
        "surface_points_m": extend(surface),
        "tool_z_world": extend(tool_z),
        "surface_tangents_world": extend(tangent),
        "surface_width_world": extend(width),
        "phase": phase,
        "source_index": source_index,
        "bonding_positions_m": bonding_position,
        "bonding_surface_points_m": surface,
        "bonding_quaternions_xyzw": quaternions,
        "bonding_tool_z_world": tool_z,
    }


def generate(config_path: Path | str = DEFAULT_CONFIG) -> dict[str, Path]:
    """Generate surface-normal clearances and collision-safe TCP poses."""

    config, root = load_config(config_path)
    csv_path = resolve_path(root, config["input"]["step2_csv"])
    pcd_path = resolve_path(root, config["input"]["step2_pcd"])
    output_dir = resolve_path(root, config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    original = read_step2_csv(csv_path)
    pcd = read_pcd_xyz(pcd_path)
    midpoint_error = np.linalg.norm(
        original["surface_center_mm"]
        - 0.5 * (original["outer_edge_mm"] + original["inner_edge_mm"]),
        axis=1,
    )
    if float(np.max(midpoint_error)) > 1.0e-3:
        raise ValueError("Step 2 surface centre is not the midpoint of the boundaries")
    pcd_distance, _ = cKDTree(pcd).query(original["global_path_mm"])

    settings = config["cartesian_pose"]
    geometry = resample_corresponding_geometry(
        original, float(settings["interpolation_spacing_mm"])
    )
    transform = np.asarray(config["frame_transform"]["mesh_local_mm_to_world_m"], float)
    task_offset = np.asarray(config["frame_transform"]["task_world_offset_m"], float)
    transformed = {
        key: transform_points(geometry[key], transform) + task_offset
        for key in (
            "global_path_mm",
            "surface_center_mm",
            "outer_edge_mm",
            "inner_edge_mm",
        )
    }
    surface = transformed["surface_center_mm"]
    clearance_mm = surface_clearance_profile_mm(surface, settings)
    offset_direction = surface_offset_directions(
        surface,
        transformed["outer_edge_mm"],
        transformed["inner_edge_mm"],
        settings,
    )
    bonding_position = surface + clearance_mm[:, None] * 1.0e-3 * offset_direction
    tangent, width, tool_z, rotations = surface_frames(
        bonding_position,
        transformed["outer_edge_mm"],
        transformed["inner_edge_mm"],
        settings,
    )
    quaternion = continuous_quaternions(rotations)
    trajectory = build_complete_trajectory(
        bonding_position, surface, quaternion, tangent, width, tool_z, settings
    )

    compensation = settings["command_compensation"]
    command_offset = (
        np.asarray(compensation["world_xyz_mm"], float) * 1.0e-3
        if bool(compensation["enabled"])
        else np.zeros(3)
    )
    trajectory["command_positions_m"] = trajectory["positions_m"] + command_offset
    trajectory["command_compensation_m"] = np.repeat(
        command_offset[None, :], len(trajectory["positions_m"]), axis=0
    )
    trajectory["bonding_surface_clearance_mm"] = clearance_mm
    trajectory["bonding_surface_offset_direction_world"] = offset_direction

    archive_path = output_path(config, root, "cartesian_archive")
    csv_output = output_path(config, root, "cartesian_csv")
    metrics_path = output_path(config, root, "cartesian_metrics")
    np.savez_compressed(archive_path, **trajectory)
    _write_csv(csv_output, trajectory)

    tilt = np.degrees(
        np.arccos(np.clip(tool_z @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))
    )
    q_step = quaternion_step_deg(quaternion)
    displacement = bonding_position - surface
    actual_clearance = np.sum(displacement * offset_direction, axis=1) * 1000.0
    world_up = np.asarray(settings["world_up"], float)
    world_up /= np.linalg.norm(world_up)
    world_z_clearance = displacement @ world_up * 1000.0
    metrics = {
        "status": "PASS",
        "input_csv": str(csv_path),
        "input_pcd": str(pcd_path),
        "input_waypoints": len(original["global_path_mm"]),
        "bonding_pose_count": len(bonding_position),
        "total_pose_count": len(trajectory["positions_m"]),
        "bonding_path_length_mm": float(
            cumulative_distance(bonding_position)[-1] * 1000.0
        ),
        "step2_csv_to_pcd_nearest_mm": _stats(pcd_distance),
        "surface_midpoint_formula_error_mm": _stats(midpoint_error),
        "surface_clearance_mm": _stats(actual_clearance),
        "surface_offset_direction": str(
            settings.get("surface_offset_direction", "world_z")
        ),
        "world_z_clearance_mm": _stats(world_z_clearance),
        "surface_clearance_profile": dict(settings.get("surface_offset_profile", {})),
        "nozzle_tilt_from_world_down_deg": _stats(tilt),
        "quaternion_step_deg": _stats(q_step),
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0))
        ),
        "command_compensation_enabled": bool(compensation["enabled"]),
        "command_compensation_world_mm": (command_offset * 1000.0).tolist(),
        "important_note": (
            "The TCP clearance is measured along the configured free-space "
            "offset direction."
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"Cartesian poses: {len(trajectory['positions_m'])} total, "
        f"{len(bonding_position)} bonding"
    )
    print(f"Bonding length: {metrics['bonding_path_length_mm']:.3f} mm")
    print(
        "Surface clearance: "
        f"{np.min(actual_clearance):.3f}..{np.max(actual_clearance):.3f} mm"
    )
    print(f"Quaternion max step: {metrics['quaternion_step_deg']['max']:.6f} deg")
    print(f"Wrote: {archive_path}")
    return {"archive": archive_path, "csv": csv_output, "metrics": metrics_path}


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _write_csv(path: Path, trajectory: Mapping[str, np.ndarray]) -> None:
    fields = (
        "index", "phase", "source_index", "target_x_m", "target_y_m", "target_z_m",
        "command_x_m", "command_y_m", "command_z_m", "qx", "qy", "qz", "qw",
        "surface_x_m", "surface_y_m", "surface_z_m", "tool_z_x", "tool_z_y", "tool_z_z",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(len(trajectory["positions_m"])):
            writer.writerow(
                (
                    index,
                    trajectory["phase"][index],
                    int(trajectory["source_index"][index]),
                    *trajectory["positions_m"][index],
                    *trajectory["command_positions_m"][index],
                    *trajectory["quaternions_xyzw"][index],
                    *trajectory["surface_points_m"][index],
                    *trajectory["tool_z_world"][index],
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
