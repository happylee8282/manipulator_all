"""Clearance directions for a TCP that must not intersect the glasses surface."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from ..control.common.geometry import normalize_rows, surface_frames


def surface_offset_directions(
    surface: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray,
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Return unit vectors pointing from the green path toward free space.

    ``world_z`` preserves the old visual-height behaviour. ``surface_normal``
    preserves the requested shortest distance on steep sections, which is the
    meaningful clearance for non-contact operation.
    """

    mode = str(settings.get("surface_offset_direction", "world_z"))
    world_up = np.asarray(settings["world_up"], dtype=float)
    world_up /= np.linalg.norm(world_up)
    if mode == "world_z":
        return np.repeat(world_up[None, :], len(surface), axis=0)
    if mode != "surface_normal":
        raise ValueError(f"unsupported surface-offset direction: {mode}")

    tangent, width, _, _ = surface_frames(surface, outer, inner, settings)
    downward_normal = normalize_rows(np.cross(tangent, width), "surface normal")
    preferred = np.asarray(settings["preferred_nozzle_direction_world"], dtype=float)
    preferred /= np.linalg.norm(preferred)
    downward_normal[(downward_normal @ preferred) < 0.0] *= -1.0
    return -downward_normal
