"""Pure first-order filtering of SI wrenches in an unchanged reference frame."""

from __future__ import annotations

import math

import numpy as np


def filter_wrench(previous, measured, *, dt_s: float, cutoff_hz: float) -> np.ndarray:
    previous = np.asarray(previous, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if previous.shape != (6,) or measured.shape != (6,):
        raise ValueError("wrench shape must be (6,): Fx,Fy,Fz [N], Tx,Ty,Tz [Nm]")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(measured)):
        raise ValueError("wrench values must be finite")
    if not math.isfinite(dt_s) or dt_s <= 0 or not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("dt_s and cutoff_hz must be finite and positive")
    alpha = -math.expm1(-2.0 * math.pi * cutoff_hz * dt_s)
    return previous + alpha * (measured - previous)
