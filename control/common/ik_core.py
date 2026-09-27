"""Pure IK branch scoring utilities, independent from ROS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class BranchState:
    q: np.ndarray
    previous_delta: np.ndarray
    total_cost: float
    parent: "BranchState | None"
    pose_index: int


def equivalent_near_reference(
    candidate: Sequence[float],
    reference: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray | None:
    """Choose the legal 2-pi equivalent closest to the preceding state."""

    candidate = np.asarray(candidate, float)
    reference = np.asarray(reference, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    result = np.empty_like(candidate)
    for index, value in enumerate(candidate):
        equivalents = value + 2.0 * np.pi * np.arange(-3, 4)
        legal = equivalents[(equivalents >= lower[index]) & (equivalents <= upper[index])]
        if not len(legal):
            return None
        result[index] = legal[np.argmin(np.abs(legal - reference[index]))]
    return result


def incremental_branch_cost(
    candidate: np.ndarray,
    previous: np.ndarray,
    previous_delta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Mapping[str, Any],
) -> float:
    """Penalize motion, curvature, limits, and an optional wrist joint."""

    delta = candidate - previous
    default_weights = [1.0, 1.2, 1.1, 0.7, 0.8, 0.5] if len(candidate) == 6 else np.ones(len(candidate))
    joint_weights = np.asarray(weights.get("joint_weights", default_weights), float)
    if joint_weights.shape != candidate.shape or not np.all(np.isfinite(joint_weights)) or np.any(joint_weights < 0):
        raise ValueError("joint_weights must be finite non-negative values matching the robot joints")
    motion = float(np.sum(joint_weights * delta**2))
    curvature = float(np.sum((delta - previous_delta) ** 2))
    span = np.maximum(upper - lower, 1.0e-9)
    margin = np.minimum(candidate - lower, upper - candidate) / span
    limit = float(np.sum(1.0 / np.maximum(margin, 0.02) ** 2))
    wrist_index = weights.get("wrist_joint_index", 4 if len(candidate) == 6 else None)
    if wrist_index is not None and not 0 <= int(wrist_index) < len(candidate):
        raise ValueError("wrist_joint_index is outside the robot joint array")
    wrist = 0.0 if wrist_index is None else float(1.0 / (np.sin(candidate[int(wrist_index)]) ** 2 + 0.02))
    return (
        float(weights["joint_motion"]) * motion
        + float(weights["joint_acceleration"]) * curvature
        + float(weights["joint_limit"]) * limit
        + float(weights["wrist_singularity"]) * wrist
    )


def seed_variants(
    seed: Sequence[float], count: int, amplitude: float,
    lower: Sequence[float], upper: Sequence[float],
    direction: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """Deterministic IK starts preserving the tuned six-axis perturbations."""

    seed = np.asarray(seed, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if seed.ndim != 1 or not seed.size or lower.shape != seed.shape or upper.shape != seed.shape:
        raise ValueError("seed and joint bounds must be equal nonempty vectors")
    if not all(np.all(np.isfinite(values)) for values in (seed, lower, upper)) or np.any(lower > upper):
        raise ValueError("seed and joint bounds must be finite and ordered")
    if count < 1 or not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("seed count must be positive and amplitude finite non-negative")
    directions = np.resize([1.0, -1.0, 0.7, -0.6, 0.8, -0.5], seed.size) if direction is None else np.asarray(direction, float)
    if directions.shape != seed.shape or not np.all(np.isfinite(directions)):
        raise ValueError("seed perturbation direction must match the robot joints")
    variants = [seed.copy()]
    for attempt in range(1, count):
        sign = 1.0 if attempt % 2 else -1.0
        multiplier = 1 + (attempt - 1) // 2
        variants.append(np.clip(seed + sign * multiplier * amplitude * directions, lower, upper))
    return variants


def reconstruct_branch(state: BranchState) -> np.ndarray:
    """Follow parent pointers from the selected final branch."""

    values = []
    current: BranchState | None = state
    while current is not None and current.pose_index >= 0:
        values.append(current.q)
        current = current.parent
    values.reverse()
    return np.asarray(values, dtype=float)
