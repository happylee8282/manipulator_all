"""File-based interlock shared by the Isaac monitor and ROS executor."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a JSON status file without exposing a partially written value."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_guard_status(
    path: Path,
    *,
    timeout_s: float,
    expected_snapshot_time: float | None,
) -> dict[str, Any]:
    """Return a fresh settled-glass status or raise the interlock error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Isaac glasses monitor is unavailable: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid glasses monitor JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("glasses monitor status must be a JSON object")
    age = time.time() - float(value.get("updated_wall_time", 0.0))
    if age < -1.0 or age > timeout_s:
        raise RuntimeError(
            f"Isaac glasses monitor is stale ({age:.3f} s; limit {timeout_s:.3f} s)"
        )
    if not bool(value.get("settled")) or bool(value.get("kinematic", True)):
        raise RuntimeError(
            "glasses are not a settled dynamic rigid body "
            f"(state={value.get('state')}, kinematic={value.get('kinematic')})"
        )
    captured = value.get("snapshot_captured_wall_time")
    if (
        expected_snapshot_time is not None
        and captured is not None
        and abs(float(captured) - float(expected_snapshot_time)) > 1.0e-6
    ):
        raise RuntimeError("monitor and Cartesian path use different glasses snapshots")
    if bool(value.get("violation")):
        raise RuntimeError(
            "GLASSES_MOVED: "
            f"translation={float(value.get('translation_from_snapshot_mm', 0.0)):.6f} mm, "
            f"rotation={float(value.get('rotation_from_snapshot_deg', 0.0)):.6f} deg, "
            f"classification={value.get('classification', 'unknown')}"
        )
    return value


def write_guard_command(path: Path, *, active: bool, phase: str) -> None:
    """Tell the Isaac monitor which robot-motion phase is active."""

    atomic_json(
        path,
        {
            "schema_version": 1,
            "execution_active": active,
            "phase": phase,
            "updated_wall_time": time.time(),
            "pid_note": "written by guarded ROS executor",
        },
    )
