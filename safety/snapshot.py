"""Configuration loading with a mandatory settled-glasses USD snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SNAPSHOT_SCHEMA_VERSION = 1


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"settled-glasses snapshot is missing: {path}\n"
            "Launch the Isaac scene and wait for GLASS_SETTLED_READY."
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def validate_snapshot(
    snapshot: Mapping[str, Any], snapshot_path: Path, expected_stage: Path
) -> np.ndarray:
    if int(snapshot.get("schema_version", -1)) != SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported snapshot schema in {snapshot_path}")
    if snapshot.get("status") != "SETTLED_DYNAMIC" or not bool(snapshot.get("stable")):
        raise RuntimeError(f"glasses snapshot is not dynamically settled: {snapshot_path}")
    if bool(snapshot.get("kinematic", True)):
        raise RuntimeError(
            "glasses snapshot is kinematic and cannot reveal robot-contact errors; "
            f"recapture it as a dynamic rigid body: {snapshot_path}"
        )
    recorded_stage = Path(str(snapshot.get("stage", ""))).expanduser().resolve()
    if recorded_stage != expected_stage:
        raise RuntimeError(
            f"snapshot stage mismatch: recorded={recorded_stage}, expected={expected_stage}"
        )
    if not expected_stage.is_file():
        raise RuntimeError(f"configured Isaac stage does not exist: {expected_stage}")
    recorded_mtime = int(snapshot.get("stage_mtime_ns", -1))
    if recorded_mtime != expected_stage.stat().st_mtime_ns:
        raise RuntimeError(
            "the Isaac stage changed after the glasses snapshot was captured; "
            "restart the Isaac settle runner before regenerating the path"
        )
    matrix = np.asarray(snapshot.get("mesh_local_mm_to_world_m"), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"snapshot transform must be a finite 4x4 matrix: {snapshot_path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise RuntimeError(f"snapshot transform has an invalid homogeneous row: {snapshot_path}")
    scales = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    if np.any(scales < 1.0e-7) or np.any(scales > 0.1):
        raise RuntimeError(
            f"unexpected glass mesh scale {scales.tolist()} in {snapshot_path}"
        )
    return matrix

