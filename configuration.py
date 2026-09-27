"""Compose robot, tool, task and backend profiles into one reproducible run."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np
import yaml


def package_root() -> Path:
    configured = os.environ.get("LEE_MANIPULATOR_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "config" / "robots").is_dir():
            raise ValueError(f"LEE_MANIPULATOR_ROOT has no robot profiles: {root}")
        return root
    local = Path(__file__).resolve().parent
    if (local / "config" / "robots").is_dir():
        return local
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory("lee_manipulator_part"))
    for ancestor in share.parents:
        if ancestor.name == "lee_manipulator_part" and (ancestor / "config/robots").is_dir():
            return ancestor
    return share


PROJECT_ROOT = package_root()
DEFAULT_CONFIG = Path(os.environ.get("LEE_MANIPULATOR_RUNS", str(PROJECT_ROOT / "runs"))) / "rv4fl_isaac" / "resolved.yaml"
ROBOT_ALIASES = {"mizb": "rv4fl", "mitsubishi": "rv4fl", "ur": "ur3e"}
_CONFIG_LEASES = []


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def output_path(config: Mapping[str, Any], root: Path, key: str) -> Path:
    return resolve_path(root, config["output"]["directory"]) / config["output"][key]


def read_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _profile(root: Path, category: str, name: str) -> tuple[dict, Path]:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Invalid {category} profile name: {name!r}")
    path = root / "config" / category / f"{name}.yaml"
    if not path.is_file():
        choices = ", ".join(sorted(item.stem for item in path.parent.glob("*.yaml")))
        raise ValueError(f"Unknown {category} profile {name!r}. Available: {choices}")
    return read_mapping(path), path


def validate_config(config: Mapping[str, Any]) -> None:
    joints = config["project"]["joint_names"]
    if not joints or len(joints) != len(set(joints)):
        raise ValueError("joint_names must contain unique joint names")
    size = len(joints)
    arrays = {
        "ik.joint_lower_rad": config["ik"]["joint_lower_rad"],
        "ik.joint_upper_rad": config["ik"]["joint_upper_rad"],
        "execution.initial_pose_rad": config["execution"]["initial_pose_rad"],
    }
    for key in ("velocity_limit_rad_s", "acceleration_limit_rad_s2", "jerk_limit_rad_s3"):
        values = config["time_parameterization"][key]
        arrays[f"time_parameterization.{key}"] = values
        if any(float(value) <= 0 for value in values):
            raise ValueError(f"{key} must be positive")
    for index, seed in enumerate(config["ik"].get("initial_seeds_rad", [])):
        arrays[f"ik.initial_seeds_rad[{index}]"] = seed
    for label, values in arrays.items():
        array = np.asarray(values, dtype=float)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must have {size} finite values")
    lower = np.asarray(config["ik"]["joint_lower_rad"])
    upper = np.asarray(config["ik"]["joint_upper_rad"])
    initial = np.asarray(config["execution"]["initial_pose_rad"])
    if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Joint limits or initial pose are inconsistent")
    if not config["frame_transform"].get("require_glass_snapshot", True):
        if not config["robot"].get("dry_run_only", False):
            raise ValueError("A movable robot profile must require a settled-glass snapshot")


def require_capability(config: Mapping[str, Any], *, execute: bool = False) -> None:
    if not config.get("robot", {}).get("supported", True):
        raise ValueError("Robot profile is a template; configure a supported robot model first")
    backend = config.get("backend", {})
    if not backend.get("supported", False):
        raise ValueError(backend.get("reason", "Backend has not been commissioned"))
    if backend.get("use_sim_time", False):
        raise ValueError("These Isaac bridges publish wall-clock stamps; use_sim_time must remain false")
    controller = config.get("control_mode", {})
    if not controller.get("supported", False):
        raise ValueError(controller.get("reason", "Controller has not been implemented"))
    if not config.get("tool", {}).get("supported", True):
        raise ValueError("Tool is a template; provide a model and driver before selecting it")
    if config.get("task", {}).get("events"):
        raise ValueError("Gripper task events need a segmented executor; this continuous-path task cannot dispatch them")
    if execute and config["robot"].get("dry_run_only", False):
        raise ValueError(
            f"{config['selection']['robot']} supports planning only in this package; "
            "its guarded Isaac scene has not been commissioned"
        )


def compose_config(
    *, robot: str = "rv4fl", tool: str = "auto", backend: str = "isaac",
    controller: str = "position", task: str = "glass_following",
    root: Path | None = None, runs_dir: Path | None = None, speed_scale: float = 1.0,
    asset_workspace: Path | None = None,
    path_dataset: Path | None = None,
) -> dict[str, Any]:
    root = (root or PROJECT_ROOT).resolve()
    robot = ROBOT_ALIASES.get(robot, robot)
    robot_profile, _ = _profile(root, "robots", robot)
    if not robot_profile["robot"].get("supported", True):
        raise ValueError("Robot profile is a template; configure a supported robot model first")
    if tool in ("auto", "nozzle"):
        tool = robot_profile["robot"]["default_tool"]
    selection = dict(robot=robot, tool=tool, backend=backend, controller=controller, task=task)
    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for category, name in (
        ("tasks", task), ("robots", robot), ("tools", tool),
        ("backends", backend), ("controllers", controller),
    ):
        profile, source = _profile(root, category, name)
        config = deep_merge(config, profile.get("settings", {}))
        for key in ("robot", "tool", "backend", "control_mode", "task"):
            if key in profile:
                config[key] = deep_merge(config.get(key, {}), profile[key])
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    compatible = config.get("tool", {}).get("compatible_robots", [])
    if compatible and robot not in compatible:
        raise ValueError(f"Tool {tool!r} is incompatible with robot {robot!r}")
    if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
        raise ValueError("speed_scale must be in (0, 1]")
    default_runs = os.environ.get("LEE_MANIPULATOR_RUNS")
    if runs_dir is None:
        runs_dir = Path(default_runs) if default_runs else root / "runs"
    run_dir = runs_dir.expanduser().resolve() / f"{robot}_{backend}"
    runtime = run_dir / "runtime"
    config["selection"] = selection
    config["selection"]["speed_scale"] = float(speed_scale)
    config["project"]["root"] = str(root)
    if path_dataset is not None:
        dataset = path_dataset.expanduser().resolve()
        config["input"]["step2_csv"] = str(dataset / "step2_1_accurate_global_path.csv")
        config["input"]["step2_pcd"] = str(dataset / "step2_1_accurate_global_path.pcd")
    workspace = (asset_workspace or Path(os.environ.get("LEE_MANIPULATOR_ASSET_WORKSPACE", str(root.parent)))).expanduser().resolve()
    for key, value in config["resources"].items():
        if isinstance(value, str) and value.startswith("../"):
            config["resources"][key] = str((workspace / value[3:]).resolve())
    config["output"]["directory"] = str(run_dir / "output")
    config["frame_transform"]["glass_snapshot"] = str(runtime / "glass_path_snapshot.json")
    config["glass_guard"]["status_file"] = str(runtime / "glass_monitor_status.json")
    config["glass_guard"]["command_file"] = str(runtime / "glass_guard_command.json")
    config["non_contact_safety"]["isaac_diagnostics_file"] = str(runtime / "isaac_diagnostics.json")
    config["controller"]["diagnostics_file"] = str(runtime / "controller_diagnostics.json")
    config["run_directory"] = str(run_dir)
    config["execution"]["execute"] = False
    timing = config["time_parameterization"]
    for key in ("bonding_speed_mm_s", "travel_speed_mm_s"):
        timing[key] *= speed_scale
    for key in ("initial_pose_velocity_scaling", "start_velocity_scaling"):
        config["execution"][key] *= speed_scale
    for key in ("description_file", "srdf_file", "kinematics_file", "joint_limits_file",
                "ompl_file", "moveit_controllers_file"):
        resource = resolve_path(root, config["robot"][key])
        if not resource.is_file():
            raise FileNotFoundError(resource)
        sources[str(resource)] = hashlib.sha256(resource.read_bytes()).hexdigest()
    for value in config["input"].values():
        resource = resolve_path(root, value)
        if resource.is_file():
            sources[str(resource)] = hashlib.sha256(resource.read_bytes()).hexdigest()
    config["configuration_sources"] = sources
    validate_config(config)
    from .safety.model_contract import validate_model_contract

    validate_model_contract(config, root)
    return config


def write_resolved_config(config: Mapping[str, Any], *, hold_lease: bool = False) -> Path:
    path = Path(config["run_directory"]) / "resolved.yaml"
    content = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = (path.parent / ".config.lock").open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_SH)
        identical = path.is_file() and path.read_text(encoding="utf-8") == content
        if not identical:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "This run is active with different settings. Stop its launches before "
                    "changing profiles, or select a different runs_dir."
                ) from None
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=".resolved-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(path)
        if hold_lease:
            fcntl.flock(lease, fcntl.LOCK_SH)
            _CONFIG_LEASES.append(lease)
            lease = None
    finally:
        if lease is not None:
            lease.close()
    return path


def load_config(
    path: Path | str = DEFAULT_CONFIG, *, require_snapshot: bool | None = None,
) -> tuple[dict[str, Any], Path]:
    from .safety.snapshot import read_json, validate_snapshot

    config_path = Path(path).expanduser().resolve()
    config = read_mapping(config_path)
    root = resolve_path(config_path.parent, config["project"]["root"])
    transform = config["frame_transform"]
    required = transform.get("require_glass_snapshot", True) if require_snapshot is None else require_snapshot
    snapshot_path = resolve_path(root, transform["glass_snapshot"])
    if required or snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        matrix = validate_snapshot(snapshot, snapshot_path, resolve_path(root, config["resources"]["stage"]))
        transform["mesh_local_mm_to_world_m"] = matrix.tolist()
        config["runtime_snapshot"] = {
            "path": str(snapshot_path), "captured_wall_time": snapshot.get("captured_wall_time"),
            "glass_prim": snapshot.get("glass_prim"), "glass_mesh_prim": snapshot.get("glass_mesh_prim"),
        }
    return config, root
