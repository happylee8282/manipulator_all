"""Classify an Isaac glasses-motion stop using controller and pose evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from ..configuration import DEFAULT_CONFIG, output_path, resolve_path


def optional_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def last_tracking_row(path: Path) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except FileNotFoundError:
        return {}
    return rows[-1] if rows else {}


def diagnose(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / str(config["project"]["root"])).resolve()
    status_path = resolve_path(root, config["glass_guard"]["status_file"])
    diagnostics_path = status_path.with_name("isaac_diagnostics.json")
    tracking_path = output_path(config, root, "tracking_csv")
    status = optional_json(status_path)
    diagnostics = optional_json(diagnostics_path)
    tracking = last_tracking_row(tracking_path)
    joints = diagnostics.get("joint_state_at_violation", {})
    mismatches = {
        name: abs(float(item["target_state_mismatch_before_deg"]))
        for name, item in joints.items()
        if item.get("target_state_mismatch_before_deg") is not None
    }
    max_drive_mismatch = max(mismatches.values(), default=0.0)
    joint_tracking = (
        float(tracking["max_abs_joint_error_rad"])
        if tracking.get("max_abs_joint_error_rad")
        else None
    )
    tolerance = float(config["controller"]["path_tolerance_rad"])
    phase = status.get("execution_phase", "UNKNOWN")
    causes: list[dict[str, Any]] = []
    if not bool(status.get("violation")):
        causes.append(
            {"priority": 0, "cause": "NO_RECORDED_GLASS_MOTION", "action": "No failure to diagnose."}
        )
    else:
        if joint_tracking is not None and joint_tracking > tolerance:
            causes.append(
                {
                    "priority": 1,
                    "cause": "JOINT_TRACKING_ERROR",
                    "evidence": f"{joint_tracking:.6f} rad > {tolerance:.6f} rad",
                    "action": "Check Isaac drive gains, state/command rate, and joint-name order.",
                }
            )
        # `target_state_mismatch_before_deg` compares joint state with the
        # authored USD DriveAPI attribute.  The ROS articulation controller
        # writes its live command into PhysX, not back into that USD attribute,
        # so a large value is useful context but is not tracking-error evidence.
        if phase == "RETURN_TO_INITIAL_POSE":
            causes.append(
                {
                    "priority": 2,
                    "cause": "INITIAL_POSE_TRANSITION_COLLISION_OR_CONTROL_ERROR",
                    "action": "Inspect the SRDF step_pose, the current joint state, wrist drive gains, and nearby obstacles.",
                }
            )
        elif phase == "MOVE_TO_PATH_START":
            causes.append(
                {
                    "priority": 2,
                    "cause": "START_TRANSITION_COLLISION_OR_FRAME_ERROR",
                    "action": "Compare the selected robot's world/base transform, tool TCP, and MoveIt start-transition collision scene.",
                }
            )
        elif phase == "PROCESS_PATH":
            causes.append(
                {
                    "priority": 2,
                    "cause": "PROCESS_CONTACT_OR_CLEARANCE_ERROR",
                    "action": "Inspect surface_offset, nozzle collision geometry, allowed contact pair, and the failing station.",
                }
            )
        causes.append(
            {
                "priority": 3,
                "cause": "USD_MOVEIT_COLLISION_MODEL_MISMATCH",
                "action": "Compare the recorded nozzle TCP matrix with the settled glass matrix and MoveIt collision objects.",
            }
        )
    report = {
        "status": "DIAGNOSED" if bool(status.get("violation")) else "NO_VIOLATION",
        "classification": status.get("classification"),
        "execution_phase": phase,
        "glass_translation_mm": status.get("translation_from_snapshot_mm"),
        "glass_rotation_deg": status.get("rotation_from_snapshot_deg"),
        "last_tracking_joint_error_rad": joint_tracking,
        "path_tolerance_rad": tolerance,
        "maximum_drive_target_state_mismatch_deg": max_drive_mismatch,
        "drive_target_observation": (
            "USD_AUTHORED_ATTRIBUTE_NOT_LIVE_ROS_COMMAND"
            if mismatches
            else "NOT_AVAILABLE"
        ),
        "ranked_causes": sorted(causes, key=lambda item: item["priority"]),
        "sources": {
            "monitor": str(status_path),
            "isaac_diagnostics": str(diagnostics_path),
            "tracking": str(tracking_path),
        },
    }
    destination = output_path(config, root, "tracking_metrics").with_name(
        "glass_guard_diagnosis.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    diagnose(parse_args().config)


if __name__ == "__main__":
    main()
