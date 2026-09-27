from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    if LaunchConfiguration("mode").perform(context) != "mock":
        raise ValueError("No gripper hardware driver is configured. Only mode:=mock is available.")
    if LaunchConfiguration("allow_mock").perform(context).lower() != "true":
        raise ValueError("Explicit allow_mock:=true is required for the gripper action mock.")
    return [Node(package="lee_manipulator_part", executable="gripper_mock", output="screen",
                 parameters=[{"allow_mock": True,
                              "action_name": LaunchConfiguration("action_name").perform(context)}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="mock"),
        DeclareLaunchArgument("allow_mock", default_value="false"),
        DeclareLaunchArgument("action_name", default_value="/mock_gripper/gripper_cmd"),
        OpaqueFunction(function=_setup),
    ])
