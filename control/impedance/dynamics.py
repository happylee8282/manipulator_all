"""Offline diagonal spring/damper wrench, without robot dynamics or actuation."""

from __future__ import annotations

import numpy as np


def spring_damper_wrench(pose_error, velocity_error, *, stiffness, damping) -> np.ndarray:
    """Compute K * error + D * velocity_error, all in a common Cartesian frame.

    Pose error is [translation (m), rotation-vector error (rad)]. Velocity error
    is [linear (m/s), angular (rad/s)]. Rotational error must be computed by the
    caller using rotation geometry, not by subtracting Euler angles. Output is
    [force (N), torque (Nm)]; this is not a joint torque command.
    """

    arrays = [np.asarray(value, dtype=float) for value in (pose_error, velocity_error, stiffness, damping)]
    if any(value.shape != (6,) or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("pose, velocity, stiffness and damping must be finite arrays of shape (6,)")
    position, velocity, stiffness_values, damping_values = arrays
    if np.any(stiffness_values < 0) or np.any(damping_values < 0):
        raise ValueError("stiffness and damping must be non-negative")
    return stiffness_values * position + damping_values * velocity
