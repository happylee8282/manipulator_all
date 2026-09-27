#!/usr/bin/env python3
"""Move safely to the verified start, execute it, and measure TCP tracking."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

import rclpy
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from ..configuration import DEFAULT_CONFIG, load_config, output_path
from ..control.common.geometry import quaternion_error_deg
from ..control.common.base_controller import duration_message
from ..control.common.trajectory import TimedTrajectory, sample_timed_trajectory


class SolvedPathExecutor(Node):
    def __init__(self, config_path: Path, archive_path: Path | None = None) -> None:
        super().__init__("step3_solved_path_executor")
        self.config, self.root = load_config(config_path)
        self.joint_names = tuple(self.config["project"]["joint_names"])
        path = archive_path or output_path(self.config, self.root, "joint_archive")
        archive = np.load(path)
        self.timed = TimedTrajectory(
            positions=archive["positions"].astype(float),
            velocities=archive["velocities"].astype(float),
            accelerations=archive["accelerations"].astype(float),
            time_s=archive["time_s"].astype(float),
            maximum_velocity=np.zeros(len(self.joint_names)),
            maximum_acceleration=np.zeros(len(self.joint_names)),
            maximum_jerk=np.zeros(len(self.joint_names)),
            global_time_scale=1.0,
        )
        self.process_positions = archive["cartesian_positions_m"].astype(float)
        self.command_positions = archive["command_cartesian_positions_m"].astype(float)
        self.quaternions = archive["cartesian_quaternions_xyzw"].astype(float)
        self.phases = archive["phase"].astype(str)
        if self.timed.positions.ndim != 2 or self.timed.positions.shape[1] != len(self.joint_names):
            raise ValueError("joint archive dimensions do not match the selected robot")
        if "joint_names" not in archive or tuple(archive["joint_names"].astype(str)) != self.joint_names:
            raise ValueError("joint archive names/order do not match the selected robot")
        self.current: np.ndarray | None = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            JointState,
            str(self.config["controller"]["state_topic"]),
            self._state_callback,
            qos,
        )
        self.controller = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.config["controller"]["action_name"]),
        )
        self.move_group = ActionClient(
            self, MoveGroup, str(self.config["execution"]["move_group_action"])
        )
        self.tf_buffer = Buffer(cache_time=RclpyDuration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tracking_active = False
        self.tracking_started = 0.0
        self.controller_elapsed: float | None = None
        self.tracking_rows: list[dict[str, Any]] = []
        self.create_timer(1.0 / float(self.config["controller"]["publish_rate_hz"]), self._track)

    def _state_callback(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        if all(name in values for name in self.joint_names):
            self.current = np.asarray([values[name] for name in self.joint_names], float)

    def _wait_for_state(self) -> np.ndarray:
        deadline = time.monotonic() + 10.0
        while self.current is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.current is None:
            raise RuntimeError(f"no joint state received on {self.config['controller']['state_topic']}")
        return self.current.copy()

    def _move_to_start(self) -> None:
        settings = self.config["execution"]
        target = self.timed.positions[0]
        if not bool(settings["move_to_start_with_move_group"]):
            return
        if not self.move_group.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("MoveGroup action is unavailable")
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = str(self.config["project"]["planning_group"])
        request.pipeline_id = "ompl"
        request.num_planning_attempts = 8
        request.allowed_planning_time = 8.0
        request.max_velocity_scaling_factor = float(settings["start_velocity_scaling"])
        request.max_acceleration_scaling_factor = float(settings["start_acceleration_scaling"])
        constraints = Constraints()
        for name, value in zip(self.joint_names, target):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(value)
            joint.tolerance_above = 0.001
            joint.tolerance_below = 0.001
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            raise RuntimeError("MoveGroup rejected the transition to the verified start")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=float(settings["move_group_timeout_s"])
        )
        result = result_future.result() if result_future.done() else None
        code = result.result.error_code.val if result is not None else None
        if code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveGroup start transition failed with code {code}")

    def _verify_start(self) -> None:
        tolerance = float(self.config["execution"]["require_start_tolerance_rad"])
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            actual = self._wait_for_state()
            error = float(np.max(np.abs(actual - self.timed.positions[0])))
            if error <= tolerance:
                self.get_logger().info(f"Start verified: max joint error={error:.6f} rad")
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(f"robot did not reach the start within {tolerance:g} rad")

    def _goal(self) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        for index in range(len(self.timed.positions)):
            point = JointTrajectoryPoint()
            point.positions = self.timed.positions[index].tolist()
            point.velocities = self.timed.velocities[index].tolist()
            point.accelerations = self.timed.accelerations[index].tolist()
            point.time_from_start = duration_message(float(self.timed.time_s[index]))
            goal.trajectory.points.append(point)
        return goal

    def _desired_cartesian(self, elapsed: float) -> tuple[np.ndarray, np.ndarray, int, float]:
        if elapsed <= self.timed.time_s[0]:
            return self.process_positions[0], self.quaternions[0], 0, 0.0
        if elapsed >= self.timed.time_s[-1]:
            last = len(self.timed.time_s) - 1
            return self.process_positions[last], self.quaternions[last], last, 0.0
        right = int(np.searchsorted(self.timed.time_s, elapsed, side="right"))
        left = right - 1
        alpha = (elapsed - self.timed.time_s[left]) / (
            self.timed.time_s[right] - self.timed.time_s[left]
        )
        position = (1.0 - alpha) * self.process_positions[left] + alpha * self.process_positions[right]
        right_q = self.quaternions[right].copy()
        if np.dot(self.quaternions[left], right_q) < 0.0:
            right_q *= -1.0
        quaternion = (1.0 - alpha) * self.quaternions[left] + alpha * right_q
        quaternion /= np.linalg.norm(quaternion)
        return position, quaternion, left, float(alpha)

    def _track(self) -> None:
        if not self.tracking_active or self.current is None or self.controller_elapsed is None:
            return
        # Use the controller's FollowJointTrajectory feedback clock.  Starting
        # a separate local stopwatch after goal acceptance introduces a few ms
        # of false TCP error even when the robot tracks perfectly.
        elapsed = float(self.controller_elapsed)
        try:
            transform = self.tf_buffer.lookup_transform(
                str(self.config["project"]["planning_frame"]),
                str(self.config["project"]["end_effector_link"]),
                rclpy.time.Time(),
            )
        except TransformException:
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        actual_position = np.array([translation.x, translation.y, translation.z])
        actual_quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
        actual_quaternion /= np.linalg.norm(actual_quaternion)
        desired_position, desired_quaternion, index, alpha = self._desired_cartesian(elapsed)
        desired_joint, _, _, _ = sample_timed_trajectory(self.timed, elapsed)
        position_error = actual_position - desired_position
        joint_error = self.current - desired_joint
        row: dict[str, Any] = {
            "elapsed_s": elapsed,
            "left_index": index,
            "alpha": alpha,
            "phase": self.phases[min(index, len(self.phases) - 1)],
            "desired_x_m": desired_position[0],
            "desired_y_m": desired_position[1],
            "desired_z_m": desired_position[2],
            "actual_x_m": actual_position[0],
            "actual_y_m": actual_position[1],
            "actual_z_m": actual_position[2],
            "error_x_mm": position_error[0] * 1000.0,
            "error_y_mm": position_error[1] * 1000.0,
            "error_z_mm": position_error[2] * 1000.0,
            "position_error_mm": np.linalg.norm(position_error) * 1000.0,
            "orientation_error_deg": quaternion_error_deg(
                actual_quaternion, desired_quaternion
            )[0],
            "max_abs_joint_error_rad": np.max(np.abs(joint_error)),
        }
        self.tracking_rows.append(row)

    def _controller_feedback(self, message) -> None:
        self.controller_elapsed = (
            float(message.feedback.desired.time_from_start.sec)
            + float(message.feedback.desired.time_from_start.nanosec) * 1.0e-9
        )

    def _save_tracking(self) -> dict[str, Any]:
        if not self.tracking_rows:
            raise RuntimeError("no TCP tracking samples were recorded")
        csv_path = output_path(self.config, self.root, "tracking_csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.tracking_rows[0]))
            writer.writeheader()
            writer.writerows(self.tracking_rows)
        bonding = [row for row in self.tracking_rows if row["phase"] == "bonding"]
        selected = bonding or self.tracking_rows
        metrics = {
            "status": "MEASURED",
            "sample_count": len(self.tracking_rows),
            "bonding_sample_count": len(bonding),
            "tcp_position_error_mm": stats([row["position_error_mm"] for row in selected]),
            "tcp_orientation_error_deg": stats(
                [row["orientation_error_deg"] for row in selected]
            ),
            "max_abs_joint_error_rad": stats(
                [row["max_abs_joint_error_rad"] for row in selected]
            ),
            "tracking_csv": str(csv_path),
        }
        metrics_path = output_path(self.config, self.root, "tracking_metrics")
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics

    def run(self) -> dict[str, Any]:
        self._wait_for_state()
        self._move_to_start()
        self._verify_start()
        if not self.controller.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("quintic FollowJointTrajectory server is unavailable")
        self.controller_elapsed = None
        future = self.controller.send_goal_async(
            self._goal(), feedback_callback=self._controller_feedback
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            raise RuntimeError("quintic controller rejected the verified trajectory")
        self.tracking_rows.clear()
        self.tracking_started = time.monotonic()
        self.tracking_active = True
        result_future = handle.get_result_async()
        timeout = float(self.timed.time_s[-1]) + float(
            self.config["controller"]["goal_time_tolerance_s"]
        ) + 10.0
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        self.tracking_active = False
        wrapped = result_future.result() if result_future.done() else None
        if wrapped is None:
            raise RuntimeError("controller action timed out")
        if wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"trajectory execution failed: {wrapped.result.error_string}"
            )
        metrics = self._save_tracking()
        self.get_logger().info(
            "Execution PASS: TCP p95="
            f"{metrics['tcp_position_error_mm']['p95']:.6f} mm"
        )
        return metrics


def stats(values) -> dict[str, float]:
    array = np.asarray(values, float)
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = SolvedPathExecutor(args.config, args.archive)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
