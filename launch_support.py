"""ROS launch composition shared by the public launch entrypoints."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import sys
import tempfile

from .configuration import (
    PROJECT_ROOT, compose_config, read_mapping, require_capability, resolve_path,
    write_resolved_config,
)


_LEASES = []


def _boolean(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise ValueError(f"Expected true or false, got {value!r}")
    return value.lower() == "true"


def _lease(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"Another launch owns {path}; keep only one server per run/domain") from None
    _LEASES.append(stream)


def moveit_parameters(config: dict, root: Path) -> dict:
    robot = config["robot"]
    description = resolve_path(root, robot["description_file"]).read_text(encoding="utf-8")
    semantic = resolve_path(root, robot["srdf_file"]).read_text(encoding="utf-8")
    ompl = read_mapping(resolve_path(root, robot["ompl_file"]))
    controllers = read_mapping(resolve_path(root, robot["moveit_controllers_file"]))
    return {
        "robot_description": description,
        "robot_description_semantic": semantic,
        "robot_description_kinematics": read_mapping(resolve_path(root, robot["kinematics_file"])),
        "robot_description_planning": read_mapping(resolve_path(root, robot["joint_limits_file"])),
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": not config["robot"].get("dry_run_only", False),
        **controllers,
    }


def _moveit_nodes(config: dict, root: Path, *, rviz: bool) -> list:
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue

    parameters = moveit_parameters(config, root)
    parameters["robot_description"] = ParameterValue(parameters["robot_description"], value_type=str)
    parameters["robot_description_semantic"] = ParameterValue(parameters["robot_description_semantic"], value_type=str)
    remappings = [("joint_states", config["controller"]["state_topic"])]
    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": parameters["robot_description"], "use_sim_time": False}],
             remappings=remappings),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[parameters, {"use_sim_time": False}], remappings=remappings),
    ]
    controller_parameters = dict(config["controller"])
    controller_parameters.update(joint_names=config["project"]["joint_names"], use_sim_time=False)
    if not config["robot"].get("dry_run_only", False):
        nodes.append(Node(package="lee_manipulator_part", executable="position_controller", output="screen",
                          parameters=[controller_parameters]))
    robot = config["robot"]
    if not robot.get("world_joint_in_description", False):
        translation = robot["base_xyz_m"]
        rotation = robot["base_rpy_rad"]
        arguments = []
        for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), translation + rotation):
            arguments.extend([f"--{name}", str(value)])
        arguments.extend(["--frame-id", config["project"]["planning_frame"],
                          "--child-frame-id", config["project"]["base_link"]])
        nodes.append(Node(package="tf2_ros", executable="static_transform_publisher", arguments=arguments))
    if rviz:
        arguments = []
        if robot.get("rviz_file"):
            arguments = ["-d", str(resolve_path(root, robot["rviz_file"]))]
        nodes.append(Node(package="rviz2", executable="rviz2", arguments=arguments,
                          parameters=[parameters, {"use_sim_time": False}], remappings=remappings))
    return nodes


def _setup(context, component: str):
    from launch.actions import EmitEvent, ExecuteProcess, LogInfo, RegisterEventHandler, SetEnvironmentVariable
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch.substitutions import LaunchConfiguration

    def value(name):
        return LaunchConfiguration(name).perform(context)

    def checked_exit(event, _context):
        if event.returncode != 0 and not _context.is_shutdown:
            raise RuntimeError(f"Process failed (code {event.returncode}); launch is stopping.")
        return []

    runs_dir = Path(value("runs_dir")).expanduser()
    config = compose_config(robot=value("robot"), tool=value("tool"), backend=value("backend"),
                            controller=value("controller"), task=value("task"), runs_dir=runs_dir,
                            speed_scale=float(value("speed_scale")), asset_workspace=Path(value("asset_workspace")),
                            path_dataset=Path(value("path_dataset")) if value("path_dataset") else None)
    execute = _boolean(value("execute"))
    require_capability(config, execute=execute)
    domain = value("ros_domain_id")
    if domain == "auto":
        domain = str(config["backend"].get("ros_domain_id", "auto"))
    if domain == "auto":
        domain = "31" if config["selection"]["robot"] == "rv4fl" else "32"
    if not domain.isdecimal() or not 0 <= int(domain) <= 232:
        raise ValueError("ros_domain_id must be auto or an integer from 0 through 232")
    actions = [SetEnvironmentVariable("ROS_DOMAIN_ID", domain),
               SetEnvironmentVariable("PYTHONUNBUFFERED", "1")]
    start_scene = component == "isaac" or (component == "system" and _boolean(value("start_isaac")))
    scene_command = None
    if start_scene:
        from .robot.adapters.isaac_adapter import build_command

        interpreter = value("isaac_python")
        if interpreter == "auto":
            interpreter = config["backend"]["python_executable"]
        scene_command = build_command(config, PROJECT_ROOT, interpreter, _boolean(value("gui")))
    if component in ("system", "moveit"):
        _lease(Path(tempfile.gettempdir()) / f"lee_manipulator_part_{os.getuid()}" / f"moveit_domain_{domain}.lock")
    if start_scene:
        _lease(Path(config["run_directory"]) / ".isaac.lock")
    config_path = write_resolved_config(config, hold_lease=True)
    actions.append(LogInfo(msg=f"{config['selection']} | ROS_DOMAIN_ID={domain} | config={config_path}"))
    actions.append(RegisterEventHandler(OnProcessExit(on_exit=checked_exit)))
    if scene_command:
        actions.append(ExecuteProcess(cmd=scene_command, output="screen"))
    if component in ("system", "moveit"):
        actions.extend(_moveit_nodes(config, PROJECT_ROOT, rviz=_boolean(value("launch_rviz"))))
    prepare = _boolean(value("prepare"))
    if component == "task" or (component == "system" and (prepare or execute)):
        operation = value("operation") if component == "task" else "prepare"
        arguments = [sys.executable, "-m", "lee_manipulator_part.workflow", "--config", str(config_path),
                     "--operation", operation, "--ready-timeout", value("ready_timeout")]
        if execute:
            arguments.append("--execute")
        process = ExecuteProcess(cmd=arguments, output="screen")
        if component == "task":
            actions.append(RegisterEventHandler(OnProcessExit(
                target_action=process, on_exit=[EmitEvent(event=Shutdown(reason="Task completed"))])))
        actions.append(process)
    return actions


def generate_launch_description(component: str = "system"):
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    arguments = {
        "robot": ("rv4fl", "rv4fl / ur3e (aliases: mizb / ur)"),
        "tool": ("auto", "Tool profile; auto selects the robot's mounted nozzle"),
        "backend": ("isaac", "Backend profile; real is a non-commissioned template"),
        "controller": ("position", "position; force and impedance are reserved"),
        "task": ("glass_following", "Task profile name"),
        "runs_dir": (os.environ.get("LEE_MANIPULATOR_RUNS", str(PROJECT_ROOT / "runs")), "Writable run root"),
        "asset_workspace": (os.environ.get("LEE_MANIPULATOR_ASSET_WORKSPACE", str(PROJECT_ROOT.parent)),
                            "Workspace containing the existing step1_modelsolution and old_step2 scenes"),
        "speed_scale": ("1.0", "Scale configured speeds by (0,1]"),
        "path_dataset": ("", "Optional directory with the two Step 2-1 input files"),
        "ros_domain_id": ("auto", "auto: RV4FL=31, UR3e=32"),
        "launch_rviz": ("true", "Start RViz"),
        "start_isaac": ("false", "Start the configured scene runner"),
        "isaac_python": (os.environ.get("ISAAC_PYTHON", "auto"),
                         "Isaac Sim Python executable (separate from ROS Python)"),
        "gui": ("true", "Show Isaac viewport"),
        "prepare": ("false", "Wait for scene and plan the entire task"),
        "execute": ("false", "Explicitly allow guarded trajectory execution after preparation"),
        "operation": ("prepare", "task launch: check / generate / solve / prepare / execute"),
        "ready_timeout": ("120.0", "Maximum wait for a settled scene and MoveIt"),
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default, description=description)
          for name, (default, description) in arguments.items()],
        OpaqueFunction(function=lambda context: _setup(context, component)),
    ])
