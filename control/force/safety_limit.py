"""Offline wrench validity and magnitude checks, not a hardware safety device."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WrenchLimits:
    force_norm_n: float
    torque_norm_nm: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in (self.force_norm_n, self.torque_norm_nm)):
            raise ValueError("force and torque limits must be finite and positive")


def validate_wrench(wrench, limits: WrenchLimits) -> np.ndarray:
    values = np.asarray(wrench, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("wrench must contain six finite SI values")
    if np.linalg.norm(values[:3]) > limits.force_norm_n:
        raise RuntimeError("force magnitude exceeds configured limit")
    if np.linalg.norm(values[3:]) > limits.torque_norm_nm:
        raise RuntimeError("torque magnitude exceeds configured limit")
    return values.copy()
