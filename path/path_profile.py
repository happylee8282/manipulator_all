"""Spatially varying nozzle clearance for the selected glasses path."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


AXIS_INDEX = {"world_x": 0, "world_y": 1, "world_z": 2}


def _smootherstep(values: np.ndarray) -> np.ndarray:
    """C2-continuous 0..1 blend with flat slope and curvature at both ends."""

    blend = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return blend**3 * (blend * (blend * 6.0 - 15.0) + 10.0)


def _three_zone_clearance_mm(
    distance_mm: np.ndarray, profile: Mapping[str, Any]
) -> np.ndarray:
    """Blend middle, slope-entry, and steep-region clearance plateaus."""

    middle = float(profile["middle_offset_mm"])
    entry = float(profile["slope_entry_offset_mm"])
    steep = float(profile["steep_offset_mm"])
    first_start = float(profile["middle_to_entry_start_abs_mm"])
    first_end = float(profile["middle_to_entry_end_abs_mm"])
    second_start = float(profile["entry_to_steep_start_abs_mm"])
    second_end = float(profile["entry_to_steep_end_abs_mm"])
    values = (
        middle,
        entry,
        steep,
        first_start,
        first_end,
        second_start,
        second_end,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("three-zone surface-offset profile values must be finite")
    if min(middle, entry, steep) <= 0.0:
        raise ValueError("three-zone surface-offset clearances must be positive")
    if not (0.0 <= first_start < first_end <= second_start < second_end):
        raise ValueError(
            "three-zone boundaries must satisfy 0 <= first_start < first_end "
            "<= second_start < second_end"
        )

    first_blend = _smootherstep(
        (distance_mm - first_start) / (first_end - first_start)
    )
    second_blend = _smootherstep(
        (distance_mm - second_start) / (second_end - second_start)
    )
    clearance = middle + (entry - middle) * first_blend
    return clearance + (steep - entry) * second_blend


def surface_clearance_profile_mm(
    surface_points_world_m: np.ndarray, settings: Mapping[str, Any]
) -> np.ndarray:
    """Return one smooth surface clearance per bonding waypoint.

    The glasses path runs from the right edge through the bridge/centre to the
    left edge.  A quintic smootherstep keeps the proven edge clearance while
    lowering only the centre, without introducing slope or curvature jumps.
    """

    points = np.asarray(surface_points_world_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("surface points must be an Nx3 array with at least two rows")
    if not np.all(np.isfinite(points)):
        raise ValueError("surface points contain NaN or Inf")

    fallback = float(settings["surface_offset_mm"])
    profile = settings.get("surface_offset_profile", {})
    if not isinstance(profile, Mapping) or not bool(profile.get("enabled", False)):
        return np.full(len(points), fallback, dtype=float)

    axis_name = str(profile.get("axis", "world_x"))
    if axis_name not in AXIS_INDEX:
        raise ValueError(f"unsupported surface-offset profile axis: {axis_name}")
    centre = float(profile.get("center_world_mm", 0.0))
    coordinate_mm = points[:, AXIS_INDEX[axis_name]] * 1000.0
    distance = np.abs(coordinate_mm - centre)
    mode = str(profile.get("mode", "two_zone"))
    if mode == "three_zone":
        return _three_zone_clearance_mm(distance, profile)
    if mode != "two_zone":
        raise ValueError(f"unsupported surface-offset profile mode: {mode}")

    # Backward-compatible centre/edge profile used by earlier experiments.
    centre_offset = float(profile["center_offset_mm"])
    edge_offset = float(profile.get("edge_offset_mm", fallback))
    transition_start = float(profile["transition_start_abs_mm"])
    full_edge = float(profile["full_edge_abs_mm"])
    values = (centre, centre_offset, edge_offset, transition_start, full_edge)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("surface-offset profile values must be finite")
    if centre_offset <= 0.0 or edge_offset <= 0.0:
        raise ValueError("surface-offset profile clearances must be positive")
    if transition_start < 0.0 or full_edge <= transition_start:
        raise ValueError("full_edge_abs_mm must exceed transition_start_abs_mm")

    blend = _smootherstep(
        (distance - transition_start) / (full_edge - transition_start)
    )
    return centre_offset + (edge_offset - centre_offset) * blend
