"""Pure numerical geometry used to turn a Step 2 curve into TCP poses.

Quaternion ordering is XYZW throughout this package, matching ROS geometry_msgs.
Rotation matrices store tool X, Y, Z axes in their columns.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def normalize_rows(values: np.ndarray, name: str = "vectors") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1.0e-12):
        bad = np.flatnonzero(norms[:, 0] < 1.0e-12)[:10].tolist()
        raise ValueError(f"{name} contains zero vectors at indices {bad}")
    return values / norms


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("a path needs at least two points")
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def transform_points(points_mm: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply the configured homogeneous mm-to-world-m transformation."""

    points_mm = np.asarray(points_mm, dtype=np.float64)
    homogeneous = np.column_stack((points_mm, np.ones(len(points_mm))))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def resample_corresponding_geometry(
    geometry: Mapping[str, np.ndarray], spacing_mm: float
) -> dict[str, np.ndarray]:
    """Subdivide every source segment while preserving all original vertices."""

    if spacing_mm <= 0.0:
        raise ValueError("interpolation spacing must be positive")
    source = np.asarray(geometry["global_path_mm"], dtype=np.float64)
    source_s = cumulative_distance(source)
    if np.any(np.diff(source_s) <= 1.0e-12):
        raise ValueError("Step 2 path contains duplicate consecutive points")
    sections = []
    for left, right in zip(source_s[:-1], source_s[1:]):
        intervals = max(1, int(math.ceil((right - left) / spacing_mm - 1.0e-12)))
        sections.append(np.linspace(left, right, intervals + 1)[:-1])
    target_s = np.r_[np.concatenate(sections), source_s[-1]]
    result: dict[str, np.ndarray] = {"distance_mm": target_s}
    for key in ("global_path_mm", "surface_center_mm", "outer_edge_mm", "inner_edge_mm"):
        points = np.asarray(geometry[key], dtype=np.float64)
        result[key] = np.column_stack(
            [np.interp(target_s, source_s, points[:, axis]) for axis in range(3)]
        )
    return result


def smooth_unit_vectors(vectors: np.ndarray, requested_window: int) -> np.ndarray:
    # Smooth the measured vector components first and normalize afterwards.
    # Normalizing every raw derivative before the filter would weight slow and
    # fast stations equally and subtly change the already validated Step 3
    # nozzle orientation near high-curvature sections.
    values = np.asarray(vectors, dtype=np.float64)
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window = requested_window if requested_window % 2 else requested_window + 1
    window = min(maximum, window)
    if window < 5:
        return normalize_rows(values)
    filtered = np.column_stack(
        [savgol_filter(values[:, axis], window, 3, mode="interp") for axis in range(3)]
    )
    return normalize_rows(filtered, "smoothed vectors")


def clamp_to_cone(
    vectors: np.ndarray,
    axis: np.ndarray,
    maximum_angle_deg: float,
    soft_transition_deg: float = 0.0,
) -> np.ndarray:
    """Preserve azimuth while smoothly limiting vectors around an axis."""

    values = normalize_rows(vectors, "cone input")
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if not 0.0 <= maximum_angle_deg < 90.0:
        raise ValueError("maximum cone angle must be in [0, 90)")
    if not 0.0 <= soft_transition_deg < max(maximum_angle_deg, 1.0e-12):
        raise ValueError("soft cone transition must be smaller than the cone angle")
    dots = np.clip(values @ axis, -1.0, 1.0)
    incoming = np.arccos(dots)
    maximum = math.radians(maximum_angle_deg)
    transition = math.radians(soft_transition_deg)
    outgoing = np.minimum(incoming, maximum)
    if transition > 0.0:
        start = maximum - transition
        selected = incoming > start
        outgoing[selected] = start + transition * np.tanh(
            (incoming[selected] - start) / transition
        )
    result = values.copy()
    for index in np.flatnonzero(outgoing < incoming - 1.0e-12):
        lateral = values[index] - dots[index] * axis
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1.0e-12:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(reference, axis)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            lateral = reference - np.dot(reference, axis) * axis
            lateral_norm = np.linalg.norm(lateral)
        result[index] = (
            math.cos(float(outgoing[index])) * axis
            + math.sin(float(outgoing[index])) * lateral / lateral_norm
        )
    return normalize_rows(result, "cone result")


def surface_frames(
    path_world: np.ndarray,
    outer_world: np.ndarray,
    inner_world: np.ndarray,
    settings: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute tangent, width, downward nozzle axis and rotation matrices.

    The measured ribbon normal is ``tangent x (outer->inner)``.  It is made
    continuous, oriented toward world-down, smoothed, and cone-limited.  Roll
    around the nozzle axis uses a stable world reference to reduce wrist flips.
    """

    distance = cumulative_distance(path_world)
    tangent = np.gradient(path_world, distance, axis=0, edge_order=1)
    tangent = smooth_unit_vectors(tangent, int(settings["tangent_smoothing_window"]))
    width = np.asarray(inner_world) - np.asarray(outer_world)
    width -= np.sum(width * tangent, axis=1, keepdims=True) * tangent
    width = smooth_unit_vectors(width, int(settings["normal_smoothing_window"]))
    width -= np.sum(width * tangent, axis=1, keepdims=True) * tangent
    width = normalize_rows(width, "outer-to-inner width")
    normal = normalize_rows(np.cross(tangent, width), "ribbon normal")

    preferred = np.asarray(settings["preferred_nozzle_direction_world"], dtype=float)
    preferred /= np.linalg.norm(preferred)
    normal[(normal @ preferred) < 0.0] *= -1.0
    normal = smooth_unit_vectors(normal, int(settings["normal_smoothing_window"]))
    normal = clamp_to_cone(
        normal,
        preferred,
        float(settings["max_tilt_from_world_down_deg"]),
        float(settings["cone_soft_transition_deg"]),
    )
    for index in range(1, len(normal)):
        if np.dot(normal[index - 1], normal[index]) < 0.0:
            normal[index] *= -1.0
    if np.any(normal @ preferred <= 0.0):
        raise ValueError("the nozzle direction leaves the downward hemisphere")

    roll_reference = np.asarray(settings["roll_reference_world"], dtype=float)
    roll_reference /= np.linalg.norm(roll_reference)
    tool_x = np.repeat(roll_reference[None, :], len(normal), axis=0)
    tool_x -= np.sum(tool_x * normal, axis=1, keepdims=True) * normal
    weak = np.linalg.norm(tool_x, axis=1) < 1.0e-6
    if np.any(weak):
        tool_x[weak] = tangent[weak] - np.sum(
            tangent[weak] * normal[weak], axis=1, keepdims=True
        ) * normal[weak]
    tool_x = normalize_rows(tool_x, "tool X")
    tool_y = normalize_rows(np.cross(normal, tool_x), "tool Y")
    tool_x = normalize_rows(np.cross(tool_y, normal), "orthogonal tool X")
    roll = math.radians(float(settings["roll_offset_deg"]))
    if abs(roll) > 1.0e-12:
        old_x, old_y = tool_x.copy(), tool_y.copy()
        tool_x = math.cos(roll) * old_x + math.sin(roll) * old_y
        tool_y = -math.sin(roll) * old_x + math.cos(roll) * old_y
    rotations = np.stack((tool_x, tool_y, normal), axis=2)
    width = normalize_rows(np.cross(normal, tangent), "final surface width")
    return tangent, width, normal, rotations


def continuous_quaternions(rotations: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(rotations).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return normalize_rows(quaternions, "quaternion")


def quaternion_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return shortest orientation error for corresponding XYZW quaternions."""

    first = normalize_rows(np.atleast_2d(first), "first quaternion")
    second = normalize_rows(np.atleast_2d(second), "second quaternion")
    dots = np.clip(np.abs(np.sum(first * second, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def quaternion_step_deg(quaternions: np.ndarray) -> np.ndarray:
    return quaternion_error_deg(quaternions[:-1], quaternions[1:])
