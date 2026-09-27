"""Jerk-checked time parameterization and quintic Hermite interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class TimedTrajectory:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    time_s: np.ndarray
    maximum_velocity: np.ndarray
    maximum_acceleration: np.ndarray
    maximum_jerk: np.ndarray
    global_time_scale: float


def quintic_coefficients(
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Return six coefficients for p(u), u in [0, 1]."""

    if duration <= 0.0:
        raise ValueError("quintic segment duration must be positive")
    q0, v0, a0, q1, v1, a1 = map(
        lambda value: np.asarray(value, dtype=np.float64), (q0, v0, a0, q1, v1, a1)
    )
    delta = q1 - q0
    t = float(duration)
    coefficients = np.empty((6, len(q0)), dtype=np.float64)
    coefficients[0] = q0
    coefficients[1] = v0 * t
    coefficients[2] = 0.5 * a0 * t**2
    coefficients[3] = (
        10.0 * delta - (6.0 * v0 + 4.0 * v1) * t - (1.5 * a0 - 0.5 * a1) * t**2
    )
    coefficients[4] = (
        -15.0 * delta + (8.0 * v0 + 7.0 * v1) * t + (1.5 * a0 - a1) * t**2
    )
    coefficients[5] = (
        6.0 * delta - (3.0 * v0 + 3.0 * v1) * t - (0.5 * a0 - 0.5 * a1) * t**2
    )
    return coefficients


def evaluate_quintic(
    coefficients: np.ndarray, duration: float, elapsed: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate position, velocity, acceleration, and jerk in one segment."""

    c = np.asarray(coefficients, dtype=np.float64)
    t = float(duration)
    u = float(np.clip(elapsed / t, 0.0, 1.0))
    powers = np.array([1.0, u, u**2, u**3, u**4, u**5])
    position = powers @ c
    velocity = (
        np.array([1.0, 2.0 * u, 3.0 * u**2, 4.0 * u**3, 5.0 * u**4])
        @ c[1:]
    ) / t
    acceleration = (
        np.array([2.0, 6.0 * u, 12.0 * u**2, 20.0 * u**3]) @ c[2:]
    ) / t**2
    jerk = (np.array([6.0, 24.0 * u, 60.0 * u**2]) @ c[3:]) / t**3
    return position, velocity, acceleration, jerk


def sample_timed_trajectory(
    trajectory: TimedTrajectory, elapsed: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a complete timestamped trajectory with quintic interpolation."""

    time_s = trajectory.time_s
    if elapsed <= time_s[0]:
        return (
            trajectory.positions[0].copy(),
            trajectory.velocities[0].copy(),
            trajectory.accelerations[0].copy(),
            np.zeros(trajectory.positions.shape[1]),
        )
    if elapsed >= time_s[-1]:
        return (
            trajectory.positions[-1].copy(),
            trajectory.velocities[-1].copy(),
            trajectory.accelerations[-1].copy(),
            np.zeros(trajectory.positions.shape[1]),
        )
    right = int(np.searchsorted(time_s, elapsed, side="right"))
    left = right - 1
    duration = float(time_s[right] - time_s[left])
    coefficients = quintic_coefficients(
        trajectory.positions[left],
        trajectory.velocities[left],
        trajectory.accelerations[left],
        trajectory.positions[right],
        trajectory.velocities[right],
        trajectory.accelerations[right],
        duration,
    )
    return evaluate_quintic(coefficients, duration, elapsed - time_s[left])


def _nominal_interval_times(
    q: np.ndarray,
    xyz: np.ndarray,
    phases: np.ndarray,
    settings: Mapping[str, object],
    endpoint_settle_time_s: float,
) -> np.ndarray:
    distance_mm = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000.0
    bonding = (phases[:-1] == "bonding") & (phases[1:] == "bonding")
    speed = np.where(
        bonding,
        float(settings["bonding_speed_mm_s"]),
        float(settings["travel_speed_mm_s"]),
    )
    dt = np.maximum(distance_mm / speed, float(settings["minimum_segment_time_s"]))
    pre_dwell = (phases[:-1] == "pre_bond_settle") | (phases[1:] == "pre_bond_settle")
    post_dwell = (phases[:-1] == "post_bond_settle") | (phases[1:] == "post_bond_settle")
    stationary = distance_mm <= 1.0e-9
    dt[stationary & (pre_dwell | post_dwell)] = endpoint_settle_time_s

    velocity_limit = np.asarray(settings["velocity_limit_rad_s"], float)
    required = np.max(np.abs(np.diff(q, axis=0)) / velocity_limit[None, :], axis=1)
    dt = np.maximum(dt, required)
    return dt


def _waypoint_derivatives(q: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use one C2 cubic spline only to obtain consistent waypoint derivatives."""

    velocity = np.empty_like(q)
    acceleration = np.empty_like(q)
    for joint in range(q.shape[1]):
        spline = CubicSpline(time_s, q[:, joint], bc_type=((1, 0.0), (1, 0.0)))
        velocity[:, joint] = spline(time_s, 1)
        acceleration[:, joint] = spline(time_s, 2)
    velocity[0] = 0.0
    velocity[-1] = 0.0
    return velocity, acceleration


def _sample_maxima(
    q: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    time_s: np.ndarray,
    period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_count = q.shape[1]
    maximum_v = np.zeros(joint_count)
    maximum_a = np.zeros(joint_count)
    maximum_j = np.zeros(joint_count)
    minimum_q = np.full(joint_count, np.inf)
    maximum_q = np.full(joint_count, -np.inf)
    for index, duration in enumerate(np.diff(time_s)):
        coefficients = quintic_coefficients(
            q[index], velocity[index], acceleration[index],
            q[index + 1], velocity[index + 1], acceleration[index + 1], duration,
        )
        count = max(3, int(np.ceil(duration / period)) + 1)
        for elapsed in np.linspace(0.0, duration, count):
            position, v, a, j = evaluate_quintic(coefficients, duration, elapsed)
            maximum_v = np.maximum(maximum_v, np.abs(v))
            maximum_a = np.maximum(maximum_a, np.abs(a))
            maximum_j = np.maximum(maximum_j, np.abs(j))
            minimum_q = np.minimum(minimum_q, position)
            maximum_q = np.maximum(maximum_q, position)
    return maximum_v, maximum_a, maximum_j, minimum_q, maximum_q


def time_parameterize(
    joint_positions: np.ndarray,
    cartesian_positions_m: np.ndarray,
    phases: Sequence[str],
    settings: Mapping[str, object],
    endpoint_settle_time_s: float,
    lower_limits: Sequence[float] | None = None,
    upper_limits: Sequence[float] | None = None,
) -> TimedTrajectory:
    """Create a C2 quintic trajectory and globally stretch it until limits pass.

    A common time scale preserves constant Cartesian feedrate inside the
    bonding stroke.  Velocity, acceleration, and jerk are checked on densely
    sampled quintic segments, not only at waypoints.
    """

    q = np.asarray(joint_positions, dtype=np.float64)
    xyz = np.asarray(cartesian_positions_m, dtype=np.float64)
    phase = np.asarray(phases, dtype=str)
    if q.ndim != 2 or len(q) != len(xyz) or len(q) != len(phase) or len(q) < 2:
        raise ValueError("joint, Cartesian, and phase arrays must have matching lengths")
    dt = _nominal_interval_times(q, xyz, phase, settings, endpoint_settle_time_s)
    initial_duration = float(np.sum(dt))
    velocity_limit = np.asarray(settings["velocity_limit_rad_s"], float)
    acceleration_limit = np.asarray(settings["acceleration_limit_rad_s2"], float)
    jerk_limit = np.asarray(settings["jerk_limit_rad_s3"], float)
    period = float(settings["verification_sample_period_s"])
    safety = float(settings["global_safety_scale"])

    for _ in range(12):
        time_s = np.r_[0.0, np.cumsum(dt)]
        velocity, acceleration = _waypoint_derivatives(q, time_s)
        # A configured settle interval is a true hold, not merely two equal
        # position waypoints with spline overshoot between them.  Force qdot
        # and qddot to zero on both sides so its quintic is exactly constant.
        distance_mm = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000.0
        settle = (
            (distance_mm <= 1.0e-9)
            & (
                (phase[:-1] == "pre_bond_settle")
                | (phase[1:] == "pre_bond_settle")
                | (phase[:-1] == "post_bond_settle")
                | (phase[1:] == "post_bond_settle")
            )
        )
        stop_indices = np.unique(
            np.r_[0, len(q) - 1, np.flatnonzero(settle), np.flatnonzero(settle) + 1]
        )
        velocity[stop_indices] = 0.0
        acceleration[stop_indices] = 0.0
        max_v, max_a, max_j, min_q, max_q = _sample_maxima(
            q, velocity, acceleration, time_s, period
        )
        ratio = max(
            1.0,
            float(np.max(max_v / velocity_limit)),
            float(np.sqrt(np.max(max_a / acceleration_limit))),
            float(np.cbrt(np.max(max_j / jerk_limit))),
        )
        if ratio <= 1.000001:
            break
        dt *= ratio * safety
    else:
        raise RuntimeError("jerk-aware global time scaling did not converge")

    if lower_limits is not None and upper_limits is not None:
        lower = np.asarray(lower_limits, float)
        upper = np.asarray(upper_limits, float)
        if np.any(min_q < lower - 1.0e-9) or np.any(max_q > upper + 1.0e-9):
            raise ValueError("quintic interpolation overshoots a joint position limit")
    return TimedTrajectory(
        positions=q,
        velocities=velocity,
        accelerations=acceleration,
        time_s=time_s,
        maximum_velocity=max_v,
        maximum_acceleration=max_a,
        maximum_jerk=max_j,
        global_time_scale=float(time_s[-1] / initial_duration),
    )
