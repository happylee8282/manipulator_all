"""Validate selected robot geometry, task inputs, and runtime readiness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from ..configuration import DEFAULT_CONFIG, load_config, output_path, resolve_path
from .glass_guard import read_guard_status
from .non_contact_gate import validate_non_contact_config
from .model_contract import validate_model_contract


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def raw_config(path: Path) -> tuple[dict, Path]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    check(isinstance(config, dict), "resolved configuration must be a mapping")
    root = (path.parent / str(config["project"]["root"])).resolve()
    return config, root


def validate_joint_settings(config: dict) -> tuple[str, ...]:
    names = tuple(config["project"]["joint_names"])
    check(bool(names) and all(isinstance(name, str) and name for name in names), "joint names must be nonempty strings")
    check(len(set(names)) == len(names), "joint names must be unique")
    count = len(names)
    for section, key in (
        ("ik", "joint_lower_rad"), ("ik", "joint_upper_rad"),
        ("execution", "initial_pose_rad"),
        ("time_parameterization", "velocity_limit_rad_s"),
        ("time_parameterization", "acceleration_limit_rad_s2"),
        ("time_parameterization", "jerk_limit_rad_s3"),
    ):
        values = np.asarray(config[section][key], float)
        check(values.shape == (count,) and bool(np.all(np.isfinite(values))), f"{section}.{key} must contain {count} finite values")
        if section == "time_parameterization":
            check(bool(np.all(values > 0)), f"{section}.{key} must be positive")
    lower, upper = np.asarray(config["ik"]["joint_lower_rad"]), np.asarray(config["ik"]["joint_upper_rad"])
    check(bool(np.all(lower < upper)), "joint lower limits must be below upper limits")
    initial = np.asarray(config["execution"]["initial_pose_rad"])
    check(bool(np.all((initial >= lower) & (initial <= upper))), "initial pose is outside configured joint limits")
    seeds = np.asarray(config["ik"]["initial_seeds_rad"], float)
    check(seeds.ndim == 2 and seeds.shape[0] > 0 and seeds.shape[1] == count and bool(np.all(np.isfinite(seeds))), "initial IK seeds must match the configured joint count")
    check(bool(np.all((seeds >= lower) & (seeds <= upper))), "initial IK seeds are outside configured joint limits")
    return names


def inspect_urdf(path: Path, config: dict) -> dict:
    robot = ET.parse(path).getroot()
    check(robot.tag == "robot", "description is not a URDF robot")
    joints = {item.attrib["name"]: item for item in robot.findall("joint")}
    links = {item.attrib["name"] for item in robot.findall("link")}
    names = validate_joint_settings(config)
    check(all(name in joints for name in names), "URDF is missing configured robot joints; provide expanded URDF")
    base, tcp = config["project"]["base_link"], config["project"]["end_effector_link"]
    check(base in links and tcp in links, "URDF is missing configured base or tool TCP")
    limits = {}
    for index, name in enumerate(names):
        joint = joints[name]
        check(joint.attrib.get("type") in ("revolute", "continuous"), f"position engine requires angular joints: {name}")
        limit = joint.find("limit")
        check(limit is not None, f"URDF joint limit missing: {name}")
        lower, upper = float(config["ik"]["joint_lower_rad"][index]), float(config["ik"]["joint_upper_rad"][index])
        if joint.attrib.get("type") != "continuous":
            urdf_lower, urdf_upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            check(lower >= urdf_lower - 1e-6 and upper <= urdf_upper + 1e-6, f"configured bounds exceed URDF joint limits: {name}")
        velocity = float(limit.attrib.get("velocity", "nan"))
        check(math.isfinite(velocity) and velocity > 0, f"invalid URDF velocity limit: {name}")
        check(float(config["time_parameterization"]["velocity_limit_rad_s"][index]) <= velocity + 1e-6, f"configured velocity exceeds URDF limit: {name}")
        limits[name] = {"lower_rad": lower, "upper_rad": upper, "velocity_rad_s": velocity}
    parent_joints = {joint.find("child").attrib["link"]: joint for joint in joints.values() if joint.find("child") is not None}
    current = tcp
    chain = []
    visited = set()
    while current != base:
        check(current not in visited and current in parent_joints, "TCP does not connect to configured base link")
        visited.add(current)
        joint = parent_joints[current]
        chain.append(joint.attrib["name"])
        parent = joint.find("parent")
        check(parent is not None, "joint parent link is missing")
        current = parent.attrib["link"]
    check(all(name in chain for name in names), "configured arm joints are not all on the base-to-TCP chain")
    return {"robot": robot.attrib.get("name"), "joints": list(names), "base_link": base, "end_effector_link": tcp, "limits": limits}


def runtime_status(config: dict, root: Path, config_path: Path) -> dict:
    if bool(config.get("robot", {}).get("dry_run_only", False)):
        return {"ready": False, "planning_only": True, "reason": "profile supports offline geometry and MoveIt planning; dynamic-scene execution is not commissioned"}
    try:
        loaded, _ = load_config(config_path)
        snapshot = loaded.get("runtime_snapshot", {})
        settings = config["glass_guard"]
        status = read_guard_status(
            resolve_path(root, settings["status_file"]),
            timeout_s=float(settings["status_timeout_s"]),
            expected_snapshot_time=snapshot.get("captured_wall_time"),
        )
        check(bool(status.get("physics_sleeping", False)), "settled glasses have not entered physics sleep")
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        return {"ready": False, "reason": str(error)}
    return {"ready": True, "snapshot": snapshot, "monitor_state": status.get("state"), "physics_sleeping": status.get("physics_sleeping"), "violation": status.get("violation", False)}


def run(config_path: Path, require_runtime: bool = False) -> dict:
    config_path = config_path.expanduser().resolve()
    config, root = raw_config(config_path)
    paths = {
        "stage": resolve_path(root, config["resources"]["stage"]),
        "robot_urdf": resolve_path(root, config["resources"]["robot_urdf"]),
        "step2_csv": resolve_path(root, config["input"]["step2_csv"]),
        "step2_pcd": resolve_path(root, config["input"]["step2_pcd"]),
        "glass_stl": resolve_path(root, config["resources"]["glass_stl"]),
    }
    for path in paths.values():
        check(path.is_file(), f"required file is missing: {path}")
    urdf = inspect_urdf(paths["robot_urdf"], config)
    model_contract = validate_model_contract(config, root)
    contract = validate_non_contact_config(config)
    runtime = runtime_status(config, root, config_path)
    if require_runtime:
        check(runtime["ready"], f"runtime is not ready: {runtime.get('reason', runtime)}")
    report = {
        "status": "READY" if runtime["ready"] else "PLANNING_ONLY" if runtime.get("planning_only") else "STRUCTURE_PASS_RUNTIME_PENDING",
        "config": str(config_path), "inputs": {key: str(value) for key, value in paths.items()},
        "urdf": urdf, "model_contract": model_contract,
        "usd": {"path": str(paths["stage"]), "static_check": "FILE_EXISTS", "composition_verified": False, "note": "robot prim, graph, poses and collision shapes are verified by the running Isaac adapter"},
        "non_contact_safety": contract, "runtime": runtime,
    }
    report_path = output_path(config, root, "cartesian_metrics").with_name("preflight_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    try:
        run(args.config, args.require_runtime)
    except (RuntimeError, ValueError, KeyError, OSError, ET.ParseError) as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
