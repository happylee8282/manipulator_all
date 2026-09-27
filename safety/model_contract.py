"""Cross-check profile metadata against the installed, expanded robot model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MODEL_CONTRACT: {message}")


def _vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, float)
    _require(vector.shape == (3,) and bool(np.all(np.isfinite(vector))), f"{label} must contain three finite values")
    return vector


def _pose(xyz: Any, rpy: Any, label: str) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = _vector(xyz, f"{label}.xyz")
    matrix[:3, :3] = Rotation.from_euler("xyz", _vector(rpy, f"{label}.rpy")).as_matrix()
    return matrix


def _origin(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    return _pose([float(item) for item in origin.attrib.get("xyz", "0 0 0").split()],
                 [float(item) for item in origin.attrib.get("rpy", "0 0 0").split()], "URDF origin")


def _fixed_transform(description: ET.Element, parent: str, child: str) -> np.ndarray:
    joints = {joint.find("child").attrib["link"]: joint for joint in description.findall("joint")
              if joint.find("child") is not None}
    matrix = np.eye(4)
    visited = set()
    current = child
    while current != parent:
        _require(current not in visited and current in joints, f"no fixed chain {parent} -> {child}")
        visited.add(current)
        joint = joints[current]
        _require(joint.attrib.get("type") == "fixed", f"{parent} -> {child} includes a moving joint")
        matrix = _origin(joint) @ matrix
        current = joint.find("parent").attrib["link"]
    return matrix


def _same_pose(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    _require(bool(np.allclose(actual, expected, rtol=0.0, atol=1e-9)),
             f"{label} does not match the frozen URDF; update the description and profile together")


def validate_model_contract(config: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Reject ignored geometry or controller edits before launching a model."""

    robot, project, tool = config["robot"], config["project"], config["tool"]
    description_path = (root / robot["description_file"]).resolve()
    _require(description_path == (root / config["resources"]["robot_urdf"]).resolve(),
             "resources.robot_urdf and robot.description_file select different models")
    description = ET.parse(description_path).getroot()
    semantic = ET.parse(root / robot["srdf_file"]).getroot()
    _require(description.attrib.get("name") == semantic.attrib.get("name"), "URDF and SRDF robot names differ")
    group_name = project["planning_group"]
    group = next((item for item in semantic.findall("group") if item.attrib.get("name") == group_name), None)
    _require(group is not None, f"planning group {group_name!r} is missing from SRDF")
    chains = group.findall("chain")
    _require(len(chains) == 1 and chains[0].attrib.get("base_link") == project["base_link"]
             and chains[0].attrib.get("tip_link") == project["end_effector_link"],
             "SRDF chain must match project.base_link and project.end_effector_link")
    kinematics = yaml.safe_load((root / robot["kinematics_file"]).read_text(encoding="utf-8"))
    _require(group_name in kinematics, "planning group is missing from kinematics.yaml")
    names = list(project["joint_names"])
    limits = yaml.safe_load((root / robot["joint_limits_file"]).read_text(encoding="utf-8"))
    _require(all(name in limits.get("joint_limits", {}) for name in names), "MoveIt joint limits omit a configured joint")
    controllers = yaml.safe_load((root / robot["moveit_controllers_file"]).read_text(encoding="utf-8"))
    manager = controllers.get("moveit_simple_controller_manager", {})
    matching = []
    for name in manager.get("controller_names", []):
        controller = manager[name]
        action = "/" + "/".join((name.strip("/"), controller.get("action_ns", "").strip("/")))
        if action == "/" + config["controller"]["action_name"].strip("/"):
            matching.append(controller)
    _require(len(matching) == 1, "controller.action_name must match one MoveIt trajectory controller")
    _require(matching[0].get("type") == "FollowJointTrajectory" and matching[0].get("joints") == names,
             "MoveIt controller type or joint order differs from the robot profile")

    base_pose = _pose(robot["base_xyz_m"], robot["base_rpy_rad"], "robot.base")
    _require(bool(np.allclose(base_pose[:3, 3], _vector(config["frame_transform"]["robot_base_world_xyz_m"], "frame_transform.robot_base_world_xyz_m"),
                             rtol=0.0, atol=1e-9)), "robot.base_xyz_m and frame_transform.robot_base_world_xyz_m disagree")
    if robot.get("world_joint_in_description", False):
        _same_pose(_fixed_transform(description, project["planning_frame"], project["base_link"]), base_pose, "robot.base_xyz_m/base_rpy_rad")

    if tool.get("supported", False) and tool.get("type") == "fixed":
        for key in ("parent_link", "link", "tcp_link", "mount_xyz_m", "mount_rpy_rad", "tcp_xyz_m", "mesh_file"):
            _require(key in tool, f"fixed tool metadata is missing {key}")
        _require(tool["tcp_link"] == project["end_effector_link"], "tool.tcp_link differs from the planning TCP")
        _same_pose(_fixed_transform(description, tool["parent_link"], tool["link"]),
                   _pose(tool["mount_xyz_m"], tool["mount_rpy_rad"], "tool.mount"), "tool.mount_xyz_m/mount_rpy_rad")
        _same_pose(_fixed_transform(description, tool["link"], tool["tcp_link"]),
                   _pose(tool["tcp_xyz_m"], tool.get("tcp_rpy_rad", [0, 0, 0]), "tool.tcp"), "tool.tcp_xyz_m/tcp_rpy_rad")
        link = next((item for item in description.findall("link") if item.attrib.get("name") == tool["link"]), None)
        _require(link is not None, "tool link is absent from URDF")
        for category in ("visual", "collision"):
            shape = link.find(category)
            _require(shape is not None and shape.find("geometry/mesh") is not None, f"tool {category} mesh is missing")
            uri = shape.find("geometry/mesh").attrib["filename"]
            expected_uri = "package://lee_manipulator_part/" + tool["mesh_file"]
            _require(uri == expected_uri, f"tool.mesh_file differs from URDF {category} mesh")
            _require((root / tool["mesh_file"]).is_file(), "tool mesh file is missing")
            _same_pose(_origin(shape), _pose(tool.get("mesh_origin_xyz_m", [0, 0, 0]),
                                             tool.get("mesh_origin_rpy_rad", [0, 0, 0]), "tool.mesh_origin"),
                       f"tool.mesh_origin versus {category}")
    return {"status": "PASS", "description": str(description_path), "planning_group": group_name}
