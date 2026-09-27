#!/usr/bin/env python3
"""Run the RV-4FL scene, dynamically settle the rigid glasses, and guard them.

The opened USD is not saved.  Rigid-body setup, robot pose reset, and settling
controls are authored only in the in-memory session.  Before the
first Play, the robot joint state and drive targets are both reset to the
configured initial pose.  The glasses remain dynamic so an incorrect robot
collision still produces a detectable physical reaction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = PACKAGE_ROOT.parent / "step1_modelsolution" / "rv-4fl_rev_B_step" / "rv4fl_ur_e_0813_nozzle.usda"
DEFAULT_RUNTIME = PACKAGE_ROOT / "runtime"
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_INITIAL_POSE_DEG = (0.0, 0.0, 90.0, 0.0, 0.0, 0.0)
DEFAULT_WRIST_STIFFNESS = (50.0, 20.0)
DEFAULT_ARM_DAMPING = (8.0, 8.0, 10.0, 4.0)
DEFAULT_WRIST_DAMPING = (2.0, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_RUNTIME / "glass_path_snapshot.json")
    parser.add_argument("--status", type=Path, default=DEFAULT_RUNTIME / "glass_monitor_status.json")
    parser.add_argument("--command", type=Path, default=DEFAULT_RUNTIME / "glass_guard_command.json")
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_RUNTIME / "isaac_diagnostics.json")
    parser.add_argument("--ready-file", type=Path, default=DEFAULT_RUNTIME / "isaac.ready")
    parser.add_argument("--glass-prim", default="/World/AI_Glass_Front")
    parser.add_argument("--glass-mesh-prim", default="/World/AI_Glass_Front/node_/mesh_")
    parser.add_argument("--robot-root", default="/World/rv4fl/root_joint")
    parser.add_argument("--tool-prim", default="/World/rv4fl/link6/nozzle_mis_stl")
    parser.add_argument("--command-topic", default="/isaac_joint_commands")
    parser.add_argument("--state-topic", default="/isaac_joint_states")
    parser.add_argument(
        "--initial-pose-deg",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_POSE_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument(
        "--wrist-stiffness",
        type=float,
        nargs=2,
        default=DEFAULT_WRIST_STIFFNESS,
        metavar=("J5", "J6"),
        help="minimum angular-drive stiffness used to prevent tool/wrist sag",
    )
    parser.add_argument(
        "--arm-damping",
        type=float,
        nargs=4,
        default=DEFAULT_ARM_DAMPING,
        metavar=("J1", "J2", "J3", "J4"),
        help="minimum angular-drive damping used to suppress arm oscillation",
    )
    parser.add_argument(
        "--wrist-damping",
        type=float,
        nargs=2,
        default=DEFAULT_WRIST_DAMPING,
        metavar=("J5", "J6"),
        help="minimum angular-drive damping used to prevent tool/wrist sag",
    )
    parser.add_argument("--mass-kg", type=float, default=0.050)
    parser.add_argument("--update-hz", type=float, default=60.0)
    parser.add_argument("--stable-duration-s", type=float, default=3.0)
    parser.add_argument("--settle-timeout-s", type=float, default=60.0)
    parser.add_argument("--stable-linear-speed-m-s", type=float, default=0.00005)
    parser.add_argument("--stable-angular-speed-deg-s", type=float, default=0.10)
    parser.add_argument("--stable-translation-mm", type=float, default=0.020)
    parser.add_argument("--stable-rotation-deg", type=float, default=0.020)
    parser.add_argument("--max-translation-mm", type=float, default=0.10)
    parser.add_argument("--max-rotation-deg", type=float, default=0.10)
    parser.add_argument(
        "--collision-contact-offset-mm",
        type=float,
        default=0.05,
        help=(
            "PhysX contact-generation margin on the glass and nozzle.  This "
            "must stay below the Cartesian surface clearance."
        ),
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=1024,
        help="longest-axis SDF voxel resolution for the glass and nozzle colliders",
    )
    parser.add_argument(
        "--sdf-subgrid-resolution",
        type=int,
        default=8,
        help="sparse SDF block resolution; PhysX recommends 4 to 8",
    )
    parser.add_argument("--decomposition-min-thickness-mm", type=float, default=0.05)
    parser.add_argument("--decomposition-error-percent", type=float, default=1.0)
    parser.add_argument("--decomposition-voxel-resolution", type=int, default=5_000_000)
    parser.add_argument("--decomposition-max-convex-hulls", type=int, default=64)
    parser.add_argument("--solver-position-iterations", type=int, default=16)
    parser.add_argument("--solver-velocity-iterations", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


args = parse_args()
positive = (
    args.mass_kg,
    args.update_hz,
    args.stable_duration_s,
    args.settle_timeout_s,
    args.stable_linear_speed_m_s,
    args.stable_angular_speed_deg_s,
    args.stable_translation_mm,
    args.stable_rotation_deg,
    args.max_translation_mm,
    args.max_rotation_deg,
    args.collision_contact_offset_mm,
    args.sdf_resolution,
    args.sdf_subgrid_resolution,
    args.decomposition_min_thickness_mm,
    args.decomposition_error_percent,
    args.decomposition_voxel_resolution,
    args.decomposition_max_convex_hulls,
    args.solver_position_iterations,
    args.solver_velocity_iterations,
    args.max_seconds,
    *args.wrist_stiffness,
    *args.arm_damping,
    *args.wrist_damping,
)
if min(positive) <= 0.0 or not all(math.isfinite(value) for value in positive):
    raise ValueError("mass, rates, thresholds, and timeouts must be positive")
if not all(math.isfinite(value) for value in args.initial_pose_deg):
    raise ValueError("initial pose values must be finite")

args.stage = args.stage.expanduser().resolve()
for name in ("snapshot", "status", "command", "diagnostics", "ready_file"):
    setattr(args, name, getattr(args, name).expanduser().resolve())
if not args.stage.is_file():
    raise FileNotFoundError(args.stage)

os.environ.setdefault("ROS_DISTRO", "humble")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
isaac_ros_lib = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "isaacsim"
    / "exts"
    / "isaacsim.ros2.bridge"
    / "humble"
    / "lib"
)
if not isaac_ros_lib.is_dir():
    raise FileNotFoundError(f"Isaac ROS 2 bridge libraries not found: {isaac_ros_lib}")
library_paths = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
if str(isaac_ros_lib) not in library_paths:
    restarted_environment = dict(os.environ)
    restarted_environment["LD_LIBRARY_PATH"] = ":".join([*library_paths, str(isaac_ros_lib)])
    os.execve(sys.executable, [sys.executable, *sys.argv], restarted_environment)

from isaacsim import SimulationApp  # noqa: E402


simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1440 if args.gui else 640,
        "height": 900 if args.gui else 480,
        "sync_loads": True,
    }
)

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import omni.physx  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import (  # noqa: E402
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def matrix_column_convention(matrix) -> list[list[float]]:
    # Gf uses row-vector transforms (translation in row 3).  Step 3 uses
    # NumPy column vectors, so the exported homogeneous matrix is transposed.
    return [
        [float(matrix[column][row]) for column in range(4)]
        for row in range(4)
    ]


def world_matrix(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"required USD prim is missing: {prim_path}")
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)


def pose(stage, prim_path: str) -> tuple[list[float], list[float]]:
    matrix = world_matrix(stage, prim_path)
    translation = matrix.ExtractTranslation()
    # The imported glasses root carries the mm->m scale.  Calling
    # ExtractRotationQuat directly on that scaled matrix returns a non-unit,
    # incorrect quaternion, so remove scale/shear before measuring rotation.
    quaternion = matrix.RemoveScaleShear().ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        [float(translation[0]), float(translation[1]), float(translation[2])],
        [float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(quaternion.GetReal())],
    )


def quaternion_distance_deg(first: list[float], second: list[float]) -> float:
    dot = abs(sum(left * right for left, right in zip(first, second)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def configure_graph(stage) -> dict[str, Any]:
    paths = {
        "publisher": "/Graph/ROS_JointStates/PublisherJointState",
        "subscriber": "/Graph/ROS_JointStates/SubscriberJointState",
        "controller": "/Graph/ROS_JointStates/ArticulationController",
    }
    prims = {name: stage.GetPrimAtPath(path) for name, path in paths.items()}
    missing = [paths[name] for name, prim in prims.items() if not prim.IsValid()]
    if missing:
        raise RuntimeError(f"RV-4FL ROS Action Graph prims are missing: {missing}")
    prims["publisher"].GetAttribute("inputs:topicName").Set(args.state_topic)
    prims["subscriber"].GetAttribute("inputs:topicName").Set(args.command_topic)
    for name in ("publisher", "controller"):
        relation = prims[name].GetRelationship("inputs:targetPrim")
        relation.SetTargets([Sdf.Path(args.robot_root)])
    prims["controller"].GetAttribute("inputs:robotPath").Set("")
    return {
        "publisher_topic": args.state_topic,
        "subscriber_topic": args.command_topic,
        "target_prim": args.robot_root,
    }


def configure_sdf_collision(mesh) -> dict[str, Any]:
    """Use a high-detail dynamic SDF instead of an outward convex approximation."""

    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh)
    sdf.CreateSdfResolutionAttr().Set(int(args.sdf_resolution))
    sdf.CreateSdfSubgridResolutionAttr().Set(int(args.sdf_subgrid_resolution))
    sdf.CreateSdfBitsPerSubgridPixelAttr().Set("BitsPerPixel32")
    sdf.CreateSdfNarrowBandThicknessAttr().Set(0.02)
    sdf.CreateSdfMarginAttr().Set(0.01)
    sdf.CreateSdfEnableRemeshingAttr().Set(False)
    sdf.CreateSdfTriangleCountReductionFactorAttr().Set(1.0)
    return {
        "approximation": "sdf",
        "resolution": int(args.sdf_resolution),
        "subgrid_resolution": int(args.sdf_subgrid_resolution),
        "bits_per_subgrid_pixel": 32,
        "narrow_band_thickness": 0.02,
        "margin": 0.01,
        "remeshing": False,
        "triangle_count_reduction_factor": 1.0,
    }


def configure_glasses(stage):
    glass = stage.GetPrimAtPath(args.glass_prim)
    mesh = stage.GetPrimAtPath(args.glass_mesh_prim)
    if not glass.IsValid() or not mesh.IsValid():
        raise RuntimeError(
            f"glasses prim or mesh is missing: {args.glass_prim}, {args.glass_mesh_prim}"
        )
    rigid = UsdPhysics.RigidBodyAPI.Apply(glass)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(False)
    rigid.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    UsdPhysics.MassAPI.Apply(glass).CreateMassAttr().Set(float(args.mass_kg))
    UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr().Set(True)
    # Convex decomposition still expands thin/concave sections enough to make
    # a 0.5 mm TCP path contact early. SDF retains the detailed dynamic mesh.
    collision_shape = configure_sdf_collision(mesh)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
    contact_offset_m = float(args.collision_contact_offset_mm) * 1.0e-3
    physx_collision.CreateContactOffsetAttr().Set(contact_offset_m)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    collision_shape.update(
        {
            "contact_offset_mm": float(args.collision_contact_offset_mm),
            "rest_offset_mm": 0.0,
        }
    )
    return glass, mesh, rigid, collision_shape


def configure_fixed_tool(stage) -> dict[str, Any]:
    """Verify that the nozzle is a collider fixed below link6, not a free body."""

    tool = stage.GetPrimAtPath(args.tool_prim)
    parent = stage.GetPrimAtPath("/World/rv4fl/link6")
    if not tool.IsValid() or not parent.IsValid():
        raise RuntimeError(f"RV-4FL tool or link6 prim is missing: {args.tool_prim}")
    if not parent.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("/World/rv4fl/link6 must be the rigid-body owner of the nozzle")
    separate_rigid_bodies = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(tool)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if separate_rigid_bodies:
        raise RuntimeError(
            "nozzle must not contain a separate rigid body; it is fixed to link6: "
            f"{separate_rigid_bodies}"
        )
    if not tool.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"nozzle collision API is missing: {args.tool_prim}")
    collision_shape = configure_sdf_collision(tool)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(tool)
    contact_offset_m = float(args.collision_contact_offset_mm) * 1.0e-3
    physx_collision.CreateContactOffsetAttr().Set(contact_offset_m)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    return {
        "tool_prim": args.tool_prim,
        "rigid_body_owner": "/World/rv4fl/link6",
        "separate_rigid_body_paths": separate_rigid_bodies,
        "attachment": "FIXED_CHILD_COLLIDER",
        "contact_offset_mm": float(args.collision_contact_offset_mm),
        "rest_offset_mm": 0.0,
        "collision_shape": collision_shape,
    }


def initialize_robot_pose_and_drives(stage) -> dict[str, Any]:
    """Reset state and targets before first Play and strengthen the wrist drives."""

    diagnostics: dict[str, Any] = {}
    wrist_stiffness = dict(zip(JOINT_NAMES[-2:], args.wrist_stiffness))
    minimum_damping = dict(
        zip(JOINT_NAMES, (*args.arm_damping, *args.wrist_damping))
    )
    for name, initial_deg in zip(JOINT_NAMES, args.initial_pose_deg):
        prim = stage.GetPrimAtPath(f"/World/rv4fl/joints/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"RV-4FL joint prim is missing: {name}")
        state_attr = prim.GetAttribute("state:angular:physics:position")
        velocity_attr = prim.GetAttribute("state:angular:physics:velocity")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        target_attr = drive.GetTargetPositionAttr()
        target_velocity_attr = drive.GetTargetVelocityAttr()
        stiffness_attr = drive.GetStiffnessAttr()
        damping_attr = drive.GetDampingAttr()
        state_before = state_attr.Get()
        target_before = target_attr.Get()
        stiffness_before = stiffness_attr.Get()
        damping_before = damping_attr.Get()
        if state_before is None:
            raise RuntimeError(f"{name} has no saved angular state")
        if not velocity_attr.IsValid():
            velocity_attr = prim.CreateAttribute(
                "state:angular:physics:velocity", Sdf.ValueTypeNames.Float
            )
        state_attr.Set(float(initial_deg))
        velocity_attr.Set(0.0)
        target_attr.Set(float(initial_deg))
        target_velocity_attr.Set(0.0)
        if stiffness_before is None or damping_before is None:
            raise RuntimeError(f"{name} angular drive gains are missing")
        damping_attr.Set(max(float(damping_before), float(minimum_damping[name])))
        if name in wrist_stiffness:
            stiffness_attr.Set(max(float(stiffness_before), float(wrist_stiffness[name])))
        diagnostics[name] = {
            "state_before_deg": float(state_before),
            "target_before_deg": None if target_before is None else float(target_before),
            "initial_state_deg": float(state_attr.Get()),
            "initial_target_deg": float(target_attr.Get()),
            "stiffness_before": (
                None if stiffness_before is None else float(stiffness_before)
            ),
            "stiffness_applied": (
                None if stiffness_attr.Get() is None else float(stiffness_attr.Get())
            ),
            "damping_before": None if damping_before is None else float(damping_before),
            "damping_applied": (
                None if damping_attr.Get() is None else float(damping_attr.Get())
            ),
        }
    return diagnostics


def configure_articulation_solver(stage) -> dict[str, Any]:
    """Increase PhysX articulation iterations to reduce visible joint chatter."""

    robot = stage.GetPrimAtPath("/World/rv4fl")
    if not robot.IsValid():
        raise RuntimeError("RV-4FL root prim is missing: /World/rv4fl")
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(robot)
    position = articulation.CreateSolverPositionIterationCountAttr()
    velocity = articulation.CreateSolverVelocityIterationCountAttr()
    position_before = position.Get()
    velocity_before = velocity.Get()
    position.Set(max(int(position_before or 0), int(args.solver_position_iterations)))
    velocity.Set(max(int(velocity_before or 0), int(args.solver_velocity_iterations)))
    return {
        "position_iterations_before": position_before,
        "position_iterations_applied": int(position.Get()),
        "velocity_iterations_before": velocity_before,
        "velocity_iterations_applied": int(velocity.Get()),
    }


def hold_robot_at_current_state(stage) -> dict[str, Any]:
    """Freeze drive targets at the measured state after an emergency pause."""

    diagnostics: dict[str, Any] = {}
    for name in JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"/World/rv4fl/joints/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"RV-4FL joint prim is missing: {name}")
        state_attr = prim.GetAttribute("state:angular:physics:position")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        target_attr = drive.GetTargetPositionAttr()
        state = state_attr.Get()
        target_before = target_attr.Get()
        if state is None:
            raise RuntimeError(f"{name} has no angular state")
        target_attr.Set(float(state))
        drive.GetTargetVelocityAttr().Set(0.0)
        diagnostics[name] = {
            "target_before_deg": (
                None if target_before is None else float(target_before)
            ),
            "state_deg": float(state),
            "hold_target_deg": float(target_attr.Get()),
            "target_state_mismatch_before_deg": (
                None if target_before is None else float(target_before) - float(state)
            ),
        }
    return diagnostics


def execution_command() -> dict[str, Any]:
    try:
        value = json.loads(args.command.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"execution_active": False, "phase": "IDLE"}
    age = time.time() - float(value.get("updated_wall_time", 0.0))
    if age > 10.0:
        return {"execution_active": False, "phase": "STALE_COMMAND"}
    return value


running = True


def request_stop(_signum, _frame) -> None:
    global running
    running = False


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


timeline = None
diagnostics: dict[str, Any] = {}
try:
    enable_extension("isaacsim.ros2.bridge")
    for _ in range(10):
        simulation_app.update()
    context = omni.usd.get_context()
    opened = context.open_stage(str(args.stage))
    if opened is False:
        raise RuntimeError(f"failed to open Isaac stage: {args.stage}")
    while context.get_stage_loading_status()[2] > 0:
        simulation_app.update()
    for _ in range(20):
        simulation_app.update()
    stage = context.get_stage()
    if abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1.0e-12:
        raise RuntimeError("this workflow requires metersPerUnit=1")

    glass, mesh, rigid, glass_collision = configure_glasses(stage)
    diagnostics = {
        "stage": str(args.stage),
        "stage_mtime_ns": args.stage.stat().st_mtime_ns,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "glass_prim": args.glass_prim,
        "glass_mesh_prim": args.glass_mesh_prim,
        "mass_kg": float(args.mass_kg),
        "glass_collision": glass_collision,
        "graph": configure_graph(stage),
        "fixed_tool": configure_fixed_tool(stage),
        "articulation_solver": configure_articulation_solver(stage),
        "robot_initialization": initialize_robot_pose_and_drives(stage),
        "settle": {},
    }
    atomic_json(
        args.command,
        {
            "schema_version": 1,
            "execution_active": False,
            "phase": "IDLE",
            "updated_wall_time": time.time(),
        },
    )
    atomic_json(
        args.status,
        {
            "schema_version": 1,
            "state": "SETTLING",
            "settled": False,
            "kinematic": False,
            "physics_sleeping": False,
            "violation": False,
            "updated_wall_time": time.time(),
        },
    )

    print(
        "[mizb] ROBOT_INITIAL_POSE_READY "
        f"degrees={list(map(float, args.initial_pose_deg))}",
        flush=True,
    )
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    period = 1.0 / args.update_hz
    settle_started = time.monotonic()
    stable_started: float | None = None
    stable_anchor_position: list[float] | None = None
    stable_anchor_quaternion: list[float] | None = None
    previous_position, previous_quaternion = pose(stage, args.glass_prim)
    previous_time = time.monotonic()
    peak_linear = 0.0
    peak_angular = 0.0
    while running and simulation_app.is_running():
        simulation_app.update()
        now = time.monotonic()
        position, quaternion = pose(stage, args.glass_prim)
        elapsed = max(now - previous_time, 1.0e-6)
        linear = vector_norm(
            [current - old for current, old in zip(position, previous_position)]
        ) / elapsed
        angular = quaternion_distance_deg(quaternion, previous_quaternion) / elapsed
        peak_linear = max(peak_linear, linear)
        peak_angular = max(peak_angular, angular)
        speed_is_stable = (
            linear <= args.stable_linear_speed_m_s
            and angular <= args.stable_angular_speed_deg_s
        )
        if speed_is_stable and stable_started is None:
            stable_started = now
            stable_anchor_position = position
            stable_anchor_quaternion = quaternion
        elif speed_is_stable:
            assert stable_anchor_position is not None
            assert stable_anchor_quaternion is not None
            stable_translation_mm = 1000.0 * vector_norm(
                [
                    current - anchor
                    for current, anchor in zip(position, stable_anchor_position)
                ]
            )
            stable_rotation_deg = quaternion_distance_deg(
                quaternion, stable_anchor_quaternion
            )
            if (
                stable_translation_mm > args.stable_translation_mm
                or stable_rotation_deg > args.stable_rotation_deg
            ):
                stable_started = now
                stable_anchor_position = position
                stable_anchor_quaternion = quaternion
        else:
            stable_started = None
            stable_anchor_position = None
            stable_anchor_quaternion = None
        stable_translation_mm = (
            0.0
            if stable_anchor_position is None
            else 1000.0
            * vector_norm(
                [
                    current - anchor
                    for current, anchor in zip(position, stable_anchor_position)
                ]
            )
        )
        stable_rotation_deg = (
            0.0
            if stable_anchor_quaternion is None
            else quaternion_distance_deg(quaternion, stable_anchor_quaternion)
        )
        atomic_json(
            args.status,
            {
                "schema_version": 1,
                "state": "SETTLING",
                "settled": False,
                "kinematic": False,
                "physics_sleeping": False,
                "violation": False,
                "linear_speed_m_s": linear,
                "angular_speed_deg_s": angular,
                "stable_for_s": 0.0 if stable_started is None else now - stable_started,
                "translation_in_stable_window_mm": stable_translation_mm,
                "rotation_in_stable_window_deg": stable_rotation_deg,
                "updated_wall_time": time.time(),
            },
        )
        if stable_started is not None and now - stable_started >= args.stable_duration_s:
            break
        if now - settle_started > args.settle_timeout_s:
            raise RuntimeError(
                "glasses did not settle before timeout: "
                f"linear={linear:.6g} m/s, angular={angular:.6g} deg/s"
            )
        previous_position, previous_quaternion = position, quaternion
        previous_time = now
        remaining = period - (time.monotonic() - now)
        if remaining > 0.0:
            time.sleep(remaining)
    if not running:
        raise RuntimeError("stopped before glasses became stable")

    # Keep one continuous Timeline Play session.  Pausing, authoring velocity
    # attributes, and playing again can reinitialise the rigid body from its
    # authored USD transform, invalidating the just-captured settled pose.
    # Put the dynamic body to sleep in PhysX without making it kinematic.
    # Contact from the robot will wake it and the monitor below will trip.
    physx_simulation = omni.physx.get_physx_simulation_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    glass_body_path = PhysicsSchemaTools.sdfPathToInt(glass.GetPath())
    physx_simulation.put_to_sleep(stage_id, glass_body_path)
    for _ in range(10):
        simulation_app.update()
    if not physx_simulation.is_sleeping(stage_id, glass_body_path):
        raise RuntimeError("glasses did not remain asleep after dynamic settling")
    baseline_position, baseline_quaternion = pose(stage, args.glass_prim)
    mesh_matrix = world_matrix(stage, args.glass_mesh_prim)
    captured_wall_time = time.time()
    snapshot = {
        "schema_version": 1,
        "status": "SETTLED_DYNAMIC",
        "stable": True,
        "kinematic": False,
        "physics_sleeping": True,
        "captured_wall_time": captured_wall_time,
        "stage": str(args.stage),
        "stage_mtime_ns": args.stage.stat().st_mtime_ns,
        "glass_prim": args.glass_prim,
        "glass_mesh_prim": args.glass_mesh_prim,
        "glass_world_position_m": baseline_position,
        "glass_world_quaternion_xyzw": baseline_quaternion,
        "mesh_local_mm_to_world_m": matrix_column_convention(mesh_matrix),
        "note": (
            "Captured during continuous Play after velocity and cumulative-pose "
            "settling; glasses remain dynamic for collision-error detection."
        ),
    }
    atomic_json(args.snapshot, snapshot)
    diagnostics["settle"] = {
        "duration_s": time.monotonic() - settle_started,
        "peak_observed_linear_speed_m_s": peak_linear,
        "peak_observed_angular_speed_deg_s": peak_angular,
        "stable_duration_s": float(args.stable_duration_s),
        "stable_translation_limit_mm": float(args.stable_translation_mm),
        "stable_rotation_limit_deg": float(args.stable_rotation_deg),
        "physics_sleeping_at_capture": True,
        "baseline_position_m": baseline_position,
        "baseline_quaternion_xyzw": baseline_quaternion,
    }
    atomic_json(args.diagnostics, diagnostics)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text(f"ready {captured_wall_time:.9f}\n", encoding="utf-8")

    maximum_translation = 0.0
    maximum_rotation = 0.0
    wake_event_count = 0
    previous_physics_sleeping = True
    violation = False
    print(
        "[mizb] GLASS_SETTLED_READY physics_sleeping=true "
        "wake_policy=diagnostic_only",
        flush=True,
    )
    started = time.monotonic()
    next_tick = started
    while running and simulation_app.is_running() and time.monotonic() - started < args.max_seconds:
        simulation_app.update()
        position, quaternion = pose(stage, args.glass_prim)
        physics_sleeping = bool(
            physx_simulation.is_sleeping(stage_id, glass_body_path)
        )
        if previous_physics_sleeping and not physics_sleeping:
            wake_event_count += 1
        previous_physics_sleeping = physics_sleeping
        translation_mm = 1000.0 * vector_norm(
            [current - reference for current, reference in zip(position, baseline_position)]
        )
        rotation_deg = quaternion_distance_deg(quaternion, baseline_quaternion)
        maximum_translation = max(maximum_translation, translation_mm)
        maximum_rotation = max(maximum_rotation, rotation_deg)
        command = execution_command()
        if (
            translation_mm > args.max_translation_mm
            or rotation_deg > args.max_rotation_deg
        ):
            violation = True
            classification = (
                "ROBOT_CONTROL_OR_COLLISION_FAULT"
                if bool(command.get("execution_active"))
                else "ENVIRONMENT_OR_FREEZE_FAULT"
            )
        else:
            classification = "NONE"
        status = {
            "schema_version": 1,
            "state": (
                "VIOLATION"
                if violation
                else (
                    "SETTLED_DYNAMIC_SLEEP_MONITORING"
                    if physics_sleeping
                    else "SETTLED_DYNAMIC_AWAKE_MONITORING"
                )
            ),
            "settled": True,
            "kinematic": False,
            "physics_sleeping": physics_sleeping,
            "physics_wake_event_count": wake_event_count,
            "violation": violation,
            "classification": classification,
            "execution_active": bool(command.get("execution_active")),
            "execution_phase": command.get("phase", "IDLE"),
            "translation_from_snapshot_mm": translation_mm,
            "rotation_from_snapshot_deg": rotation_deg,
            "maximum_translation_mm": maximum_translation,
            "maximum_rotation_deg": maximum_rotation,
            "snapshot_captured_wall_time": captured_wall_time,
            "glass_world_position_m": position,
            "glass_world_quaternion_xyzw": quaternion,
            "updated_wall_time": time.time(),
        }
        atomic_json(args.status, status)
        if violation:
            timeline.pause()
            diagnostics["violation"] = status
            diagnostics["joint_state_at_violation"] = hold_robot_at_current_state(stage)
            diagnostics["nozzle_tcp_world_matrix"] = matrix_column_convention(
                world_matrix(stage, "/World/rv4fl/link6/nozzle_tcp")
            )
            atomic_json(args.diagnostics, diagnostics)
            print(
                "[mizb] EMERGENCY_PAUSE: glasses moved during "
                f"{status['execution_phase']}; classification={classification}",
                flush=True,
            )
            while running and simulation_app.is_running():
                simulation_app.update()
                status["updated_wall_time"] = time.time()
                atomic_json(args.status, status)
                time.sleep(period)
            break
        next_tick += period
        remaining = next_tick - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        elif remaining < -0.5:
            next_tick = time.monotonic()
except BaseException:
    failure = traceback.format_exc()
    print(failure, flush=True)
    try:
        atomic_json(
            args.status,
            {
                "schema_version": 1,
                "state": "FAILED",
                "settled": False,
                "kinematic": False,
                "violation": True,
                "classification": "ISAAC_RUNNER_FAILURE",
                "error": failure,
                "updated_wall_time": time.time(),
            },
        )
        diagnostics["failure"] = failure
        atomic_json(args.diagnostics, diagnostics)
    except Exception:
        pass
    raise
finally:
    try:
        if timeline is not None:
            timeline.stop()
    except Exception:
        pass
    simulation_app.close()
