import os
from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from lee_manipulator_part.configuration import PROJECT_ROOT


def _setup(context):
    command = [sys.executable, "-m", "lee_manipulator_part.source_pipeline",
               "--input-pcd", LaunchConfiguration("input_pcd").perform(context),
               "--output-dir", LaunchConfiguration("output_dir").perform(context)]
    if LaunchConfiguration("figures").perform(context).lower() == "true":
        command.append("--figures")
    process = ExecuteProcess(cmd=command, output="screen")

    def completed(event, launch_context):
        if event.returncode and not launch_context.is_shutdown:
            raise RuntimeError(f"Path generation failed: {event.returncode}")
        return [EmitEvent(event=Shutdown(reason="Source path generation completed"))]

    return [RegisterEventHandler(OnProcessExit(target_action=process, on_exit=completed)), process]


def generate_launch_description():
    workspace = Path(os.environ.get("LEE_MANIPULATOR_ASSET_WORKSPACE", str(PROJECT_ROOT.parent)))
    return LaunchDescription([
        DeclareLaunchArgument("input_pcd", default_value=os.environ.get(
            "LEE_MANIPULATOR_SOURCE_PCD", str(workspace / "step1_modelsolution/old_step2/point_cloud/AI_Glass_Front_0.010mm.pcd"))),
        DeclareLaunchArgument("output_dir", default_value=str(PROJECT_ROOT / "runs/path_generation")),
        DeclareLaunchArgument("figures", default_value="false"),
        OpaqueFunction(function=_setup),
    ])
