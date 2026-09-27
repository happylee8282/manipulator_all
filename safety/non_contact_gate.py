"""Hard execution gates for 0.5 mm non-contact commissioning."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..configuration import output_path, resolve_path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"NON_CONTACT_GATE: {message}")


def validate_non_contact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the static safety contract without requiring generated files."""

    safety = config.get("non_contact_safety", {})
    required = bool(safety.get("required", False))
    if not required:
        return {"required": False}

    minimum = float(safety["minimum_tcp_clearance_mm"])
    tolerance = float(safety.get("uniform_clearance_tolerance_mm", 0.001))
    _require(math.isfinite(minimum) and minimum > 0.0, "invalid minimum clearance")
    _require(math.isfinite(tolerance) and tolerance >= 0.0, "invalid clearance tolerance")

    collision = config["collision_scene"]
    _require(bool(collision["publish_glass"]), "glass collision mesh must be published")
    if bool(safety.get("require_nozzle_glass_collision_check", True)):
        _require(
            not bool(collision["allow_nozzle_glass_contact"]),
            "nozzle-glass contact must not be allowed in MoveIt",
        )

    cartesian = config["cartesian_pose"]
    required_direction = str(
        safety.get("required_surface_offset_direction", "surface_normal")
    )
    configured_direction = str(cartesian.get("surface_offset_direction", "world_z"))
    _require(
        configured_direction == required_direction,
        "surface-offset direction must be "
        f"{required_direction!r}, got {configured_direction!r}",
    )
    if bool(safety.get("require_uniform_clearance", True)):
        profile = cartesian.get("surface_offset_profile", {})
        _require(
            not bool(profile.get("enabled", False)),
            "surface-offset profile must be disabled for uniform clearance",
        )
        fixed = float(cartesian["surface_offset_mm"])
        _require(
            fixed + tolerance >= minimum,
            f"configured clearance {fixed:g} mm is below {minimum:g} mm",
        )
    return {
        "required": True,
        "minimum_tcp_clearance_mm": minimum,
        "uniform_clearance_tolerance_mm": tolerance,
        "surface_offset_direction": configured_direction,
        "require_isaac_sdf_collision": bool(
            safety.get("require_isaac_sdf_collision", True)
        ),
        "minimum_isaac_sdf_resolution": int(
            safety.get("minimum_isaac_sdf_resolution", 1024)
        ),
        "nozzle_glass_contact_allowed": bool(collision["allow_nozzle_glass_contact"]),
    }


def validate_isaac_sdf_diagnostics(
    diagnostics: Mapping[str, Any], safety: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the high-detail PhysX shapes proven by the full simulation run."""

    if not bool(safety.get("require_isaac_sdf_collision", True)):
        return {"required": False}
    minimum = int(safety.get("minimum_isaac_sdf_resolution", 1024))
    _require(minimum > 0, "invalid minimum Isaac SDF resolution")
    glass = diagnostics.get("glass_collision", {})
    tool = diagnostics.get("fixed_tool", {}).get("collision_shape", {})
    for name, shape in (("glass", glass), ("nozzle", tool)):
        _require(
            shape.get("approximation") == "sdf",
            f"Isaac {name} collider must use SDF, got {shape.get('approximation')!r}",
        )
        resolution = int(shape.get("resolution", -1))
        _require(
            resolution >= minimum,
            f"Isaac {name} SDF resolution {resolution} is below {minimum}",
        )
    return {"required": True, "minimum_resolution": minimum, "status": "PASS"}


def validate_non_contact_artifacts(
    config: Mapping[str, Any], root: Path, config_path: Path
) -> dict[str, Any]:
    """Reject motion unless fresh Cartesian and collision-proof artifacts agree."""

    contract = validate_non_contact_config(config)
    if not contract["required"]:
        return contract

    cartesian_archive = output_path(config, root, "cartesian_archive")
    cartesian_metrics = output_path(config, root, "cartesian_metrics")
    joint_archive = output_path(config, root, "joint_archive")
    solver_metrics = output_path(config, root, "solver_metrics")
    snapshot = resolve_path(root, config["frame_transform"]["glass_snapshot"])
    safety = config["non_contact_safety"]
    isaac_diagnostics = resolve_path(
        root, safety.get("isaac_diagnostics_file", "runtime/isaac_diagnostics.json")
    )
    paths = (
        config_path,
        snapshot,
        cartesian_archive,
        cartesian_metrics,
        joint_archive,
        solver_metrics,
        isaac_diagnostics,
    )
    for path in paths:
        _require(path.is_file(), f"required proof artifact is missing: {path}")

    config_time = config_path.stat().st_mtime_ns
    snapshot_time = snapshot.stat().st_mtime_ns
    cartesian_time = min(
        cartesian_archive.stat().st_mtime_ns, cartesian_metrics.stat().st_mtime_ns
    )
    joint_time = joint_archive.stat().st_mtime_ns
    solver_time = solver_metrics.stat().st_mtime_ns
    isaac_time = isaac_diagnostics.stat().st_mtime_ns
    _require(
        cartesian_time >= max(config_time, snapshot_time),
        "Cartesian path is older than the config or settled-glass snapshot. "
        "Keep the Isaac runner open, then run "
        "ros2 launch lee_manipulator_part task.launch.py robot:=<robot> operation:=prepare",
    )
    _require(joint_time >= cartesian_time, "joint path is stale; rerun the IK dry-run")
    _require(solver_time >= joint_time, "solver proof is stale; rerun the IK dry-run")
    _require(
        isaac_time >= snapshot_time,
        "Isaac collision diagnostics are older than the settled-glass snapshot",
    )

    cartesian = json.loads(cartesian_metrics.read_text(encoding="utf-8"))
    solver = json.loads(solver_metrics.read_text(encoding="utf-8"))
    isaac = json.loads(isaac_diagnostics.read_text(encoding="utf-8"))
    sdf = validate_isaac_sdf_diagnostics(isaac, safety)
    _require(cartesian.get("status") == "PASS", "Cartesian generation did not pass")
    _require(
        cartesian.get("surface_offset_direction")
        == contract["surface_offset_direction"],
        "Cartesian proof used the wrong surface-offset direction",
    )
    _require(solver.get("status") == "PASS", "MoveIt dry-run did not pass")
    _require(
        int(solver.get("collision_invalid_state_count", -1)) == 0,
        "MoveIt reported a collision-invalid trajectory state",
    )
    _require(
        solver.get("nozzle_glass_contact_allowed") is False,
        "solver proof did not explicitly disallow nozzle-glass contact",
    )

    clearance = cartesian.get("surface_clearance_mm", {})
    minimum = float(clearance.get("min", float("nan")))
    maximum = float(clearance.get("max", float("nan")))
    required_minimum = float(contract["minimum_tcp_clearance_mm"])
    tolerance = float(contract["uniform_clearance_tolerance_mm"])
    _require(math.isfinite(minimum) and math.isfinite(maximum), "missing clearance proof")
    _require(
        minimum + tolerance >= required_minimum,
        f"generated minimum clearance {minimum:g} mm is below {required_minimum:g} mm",
    )
    if bool(config["non_contact_safety"].get("require_uniform_clearance", True)):
        _require(
            maximum - minimum <= tolerance,
            f"clearance is not uniform ({minimum:g}..{maximum:g} mm)",
        )
    return {
        **contract,
        "status": "PASS",
        "generated_clearance_min_mm": minimum,
        "generated_clearance_max_mm": maximum,
        "collision_invalid_state_count": 0,
        "isaac_sdf": sdf,
    }
