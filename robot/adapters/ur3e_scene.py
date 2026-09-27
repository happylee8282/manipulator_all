#!/usr/bin/env python3
"""Run the existing UR3e stage headlessly or visibly without saving changes.

The original USD is opened read-only in practice: graph repair and optional
drive/physics overrides are authored only in the in-memory stage and are never
saved.  The loop is wall-clock paced so the ROS trajectory server and PhysX do
not advance on incompatible clocks during tracking experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


DEFAULT_STAGE = Path(
    __file__
).resolve().parents[3] / "old_step2/glass_robotarm/ur_e_0813_nozzle.usda"
DEFAULT_REPAIR = Path(__file__).with_name("ur3e_graph_repair.py")
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--repair-script", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument("--command-topic", default="/isaac_joint_command")
    parser.add_argument("--state-topic", default="/isaac_joint_states")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="show the Isaac Sim window instead of running headlessly",
    )
    parser.add_argument("--update-hz", type=float, default=120.0)
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=0.0,
        help="0 keeps the USD value; positive values override it in memory",
    )
    parser.add_argument("--stiffness-scale", type=float, default=1.0)
    parser.add_argument("--damping-scale", type=float, default=1.0)
    parser.add_argument("--max-force-scale", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    return parser.parse_args()


args = parse_args()
if min(
    args.update_hz,
    args.stiffness_scale,
    args.damping_scale,
    args.max_force_scale,
    args.max_seconds,
) <= 0.0:
    raise ValueError("rates, scales, and max-seconds must be positive")

# Isaac 5's ROS bridge selects its bundled Humble libraries from this value.
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
    # glibc reads LD_LIBRARY_PATH when the process starts.  Updating os.environ
    # alone is too late for the bridge's native libraries, so restart this same
    # interpreter once with the Isaac Humble directory present from startup.
    restarted_environment = dict(os.environ)
    restarted_environment["LD_LIBRARY_PATH"] = ":".join(
        [*library_paths, str(isaac_ros_lib)]
    )
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

import carb  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402


def number(value):
    return None if value is None else float(value)


def configure_stage(stage) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "stage": str(args.stage.resolve()),
        "update_hz": float(args.update_hz),
        "physics_hz_requested": float(args.physics_hz),
        "stiffness_scale": float(args.stiffness_scale),
        "damping_scale": float(args.damping_scale),
        "max_force_scale": float(args.max_force_scale),
        "joints": {},
        "physics_scenes": [],
    }
    for prim in stage.Traverse():
        if prim.GetName() in JOINT_NAMES:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            stiffness_attr = drive.GetStiffnessAttr()
            damping_attr = drive.GetDampingAttr()
            force_attr = drive.GetMaxForceAttr()
            before = {
                "stiffness": number(stiffness_attr.Get()),
                "damping": number(damping_attr.Get()),
                "max_force": number(force_attr.Get()),
            }
            if before["stiffness"] is not None:
                stiffness_attr.Set(before["stiffness"] * args.stiffness_scale)
            if before["damping"] is not None:
                damping_attr.Set(before["damping"] * args.damping_scale)
            if before["max_force"] is not None:
                force_attr.Set(before["max_force"] * args.max_force_scale)
            diagnostics["joints"][prim.GetName()] = {
                "path": str(prim.GetPath()),
                "before": before,
                "applied": {
                    "stiffness": number(stiffness_attr.Get()),
                    "damping": number(damping_attr.Get()),
                    "max_force": number(force_attr.Get()),
                },
            }
        if prim.IsA(UsdPhysics.Scene):
            physx = PhysxSchema.PhysxSceneAPI.Apply(prim)
            rate_attr = physx.GetTimeStepsPerSecondAttr()
            before_rate = number(rate_attr.Get())
            if args.physics_hz > 0.0:
                rate_attr.Set(float(args.physics_hz))
            diagnostics["physics_scenes"].append(
                {
                    "path": str(prim.GetPath()),
                    "before_hz": before_rate,
                    "applied_hz": number(rate_attr.Get()),
                }
            )
    return diagnostics


running = True


def request_stop(_signum, _frame) -> None:
    global running
    running = False


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

try:
    enable_extension("isaacsim.ros2.bridge")
    for _ in range(10):
        simulation_app.update()

    context = omni.usd.get_context()
    # Isaac Sim 5 may return None while scheduling an otherwise successful
    # open, so reject only an explicit False and use the loading state below.
    opened = context.open_stage(str(args.stage.expanduser().resolve()))
    if opened is False:
        raise RuntimeError(f"failed to open Isaac stage: {args.stage}")
    while context.get_stage_loading_status()[2] > 0:
        simulation_app.update()
    for _ in range(10):
        simulation_app.update()

    repair_code = args.repair_script.read_text(encoding="utf-8")
    exec(compile(repair_code, str(args.repair_script), "exec"), {})
    stage = context.get_stage()
    for node_name, topic in (
        ("SubscriberJointState", args.command_topic),
        ("PublisherJointState", args.state_topic),
    ):
        topic_attr = stage.GetPrimAtPath(
            f"/Graph/ROS_JointStates/{node_name}"
        ).GetAttribute("inputs:topicName")
        if not topic_attr.IsValid():
            raise RuntimeError(f"Missing ROS topic attribute: {node_name}")
        topic_attr.Set(topic)
    diagnostics = configure_stage(stage)
    diagnostics["meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(stage))
    diagnostics["time_codes_per_second"] = float(stage.GetTimeCodesPerSecond())
    print(json.dumps(diagnostics, indent=2), flush=True)

    if args.diagnostics_json is not None:
        args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_json.write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(20):
        simulation_app.update()

    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(f"ready {time.time():.9f}\n", encoding="utf-8")
    mode = "GUI" if args.gui else "HEADLESS"
    print(f"[experiment] ISAAC_{mode}_READY", flush=True)

    period = 1.0 / args.update_hz
    started = time.monotonic()
    deadline = started + args.max_seconds
    next_tick = started
    while running and simulation_app.is_running() and time.monotonic() < deadline:
        simulation_app.update()
        next_tick += period
        remaining = next_tick - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        elif remaining < -0.5:
            next_tick = time.monotonic()
except BaseException:
    failure = traceback.format_exc()
    print(failure, flush=True)
    if args.diagnostics_json is not None:
        failure_path = args.diagnostics_json.with_name("isaac_failure.txt")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(failure, encoding="utf-8")
    raise
finally:
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass
    simulation_app.close()
