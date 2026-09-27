"""Profile-driven quintic server with persistent goal/result diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from ..common.base_controller import (
    QuinticTrajectoryServer,
    duration_seconds,
)

from .control_math import corrected_joint_command, filtered_feedback_error


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class DiagnosticQuinticTrajectoryServer(QuinticTrajectoryServer):
    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("diagnostics_file", "")
        self.declare_parameter("position_feedback_gain", 0.15)
        self.declare_parameter("maximum_feedback_correction_rad", 0.004)
        self.declare_parameter("feedback_filter_time_constant_s", 0.10)
        self.declare_parameter("feedback_deadband_rad", 0.0005)
        self.declare_parameter("velocity_feedforward_scale", 1.0)
        value = str(self.get_parameter("diagnostics_file").value)
        self.diagnostics_path = Path(value).expanduser().resolve() if value else None
        self.position_feedback_gain = float(
            self.get_parameter("position_feedback_gain").value
        )
        self.maximum_feedback_correction = float(
            self.get_parameter("maximum_feedback_correction_rad").value
        )
        self.velocity_feedforward_scale = float(
            self.get_parameter("velocity_feedforward_scale").value
        )
        self.feedback_filter_time_constant = float(
            self.get_parameter("feedback_filter_time_constant_s").value
        )
        self.feedback_deadband = float(self.get_parameter("feedback_deadband_rad").value)
        if (
            self.position_feedback_gain < 0.0
            or self.maximum_feedback_correction <= 0.0
            or self.velocity_feedforward_scale < 0.0
            or self.feedback_filter_time_constant <= 0.0
            or self.feedback_deadband < 0.0
        ):
            raise ValueError("invalid feedforward/feedback controller parameters")
        self.feedback_alpha = 1.0 - math.exp(
            -1.0 / (self.rate_hz * self.feedback_filter_time_constant)
        )
        self.filtered_feedback_error = np.zeros(len(self.joint_names))
        self.maximum_applied_feedback_correction = 0.0
        self.maximum_feedback_correction_step = 0.0
        self.previous_feedback_correction = np.zeros(len(self.joint_names))
        self.get_logger().info(
            "Outer-loop control: "
            f"velocity feedforward={self.velocity_feedforward_scale:g}, "
            f"position feedback gain={self.position_feedback_gain:g}, "
            f"correction limit={self.maximum_feedback_correction:g} rad, "
            f"filter tau={self.feedback_filter_time_constant:g} s, "
            f"deadband={self.feedback_deadband:g} rad"
        )

    def _publish_command(self, position: np.ndarray, velocity: np.ndarray) -> None:
        actual, age = self._actual_values()
        if actual is None or age > self.state_timeout:
            super()._publish_command(position, velocity)
            return
        self.filtered_feedback_error, active_feedback_error = filtered_feedback_error(
            self.filtered_feedback_error,
            position - actual,
            smoothing_alpha=self.feedback_alpha,
            deadband_rad=self.feedback_deadband,
        )
        command_position, command_velocity, correction = corrected_joint_command(
            position,
            velocity,
            actual,
            position_feedback_gain=self.position_feedback_gain,
            maximum_correction_rad=self.maximum_feedback_correction,
            velocity_feedforward_scale=self.velocity_feedforward_scale,
            feedback_error=active_feedback_error,
        )
        self.maximum_applied_feedback_correction = max(
            self.maximum_applied_feedback_correction,
            float(np.max(np.abs(correction))),
        )
        self.maximum_feedback_correction_step = max(
            self.maximum_feedback_correction_step,
            float(np.max(np.abs(correction - self.previous_feedback_correction))),
        )
        self.previous_feedback_correction = correction.copy()
        super()._publish_command(command_position, command_velocity)

    def _save_diagnostic(self, value: dict[str, Any]) -> None:
        if self.diagnostics_path is not None:
            atomic_json(self.diagnostics_path, value)

    async def _execute(self, goal_handle):
        self.filtered_feedback_error = np.zeros(len(self.joint_names))
        self.maximum_applied_feedback_correction = 0.0
        self.maximum_feedback_correction_step = 0.0
        self.previous_feedback_correction = np.zeros(len(self.joint_names))
        goal = goal_handle.request
        actual, age = self._actual_values()
        incoming_names = list(goal.trajectory.joint_names)
        first = None
        if goal.trajectory.points and all(name in incoming_names for name in self.joint_names):
            order = [incoming_names.index(name) for name in self.joint_names]
            first = np.asarray(
                [goal.trajectory.points[0].positions[index] for index in order], float
            )
        diagnostic: dict[str, Any] = {
            "status": "RUNNING",
            "received_wall_time": time.time(),
            "joint_names": incoming_names,
            "point_count": len(goal.trajectory.points),
            "first_point_time_s": (
                duration_seconds(goal.trajectory.points[0].time_from_start)
                if goal.trajectory.points
                else None
            ),
            "duration_s": (
                duration_seconds(goal.trajectory.points[-1].time_from_start)
                if goal.trajectory.points
                else None
            ),
            "actual_state_age_s": age,
            "actual_start_rad": None if actual is None else actual.tolist(),
            "trajectory_first_rad": None if first is None else first.tolist(),
            "initial_max_error_rad": (
                None
                if actual is None or first is None
                else float(np.max(np.abs(actual - first)))
            ),
            "configured_path_tolerance_rad": self.path_tolerance,
            "configured_path_grace_s": self.path_grace,
            "controller_mode": "QUINTIC_VELOCITY_FEEDFORWARD_PLUS_BOUNDED_P_FEEDBACK",
            "publish_velocity_commands": self.publish_velocity,
            "velocity_feedforward_scale": self.velocity_feedforward_scale,
            "position_feedback_gain": self.position_feedback_gain,
            "maximum_feedback_correction_rad": self.maximum_feedback_correction,
            "feedback_filter_time_constant_s": self.feedback_filter_time_constant,
            "feedback_filter_alpha": self.feedback_alpha,
            "feedback_deadband_rad": self.feedback_deadband,
        }
        self._save_diagnostic(diagnostic)
        try:
            result = await super()._execute(goal_handle)
            diagnostic.update(
                {
                    "status": "SUCCESS" if result.error_code == 0 else "FAILED",
                    "result_error_code": int(result.error_code),
                    "result_error_string": str(result.error_string),
                    "completed_wall_time": time.time(),
                    "maximum_applied_feedback_correction_rad": (
                        self.maximum_applied_feedback_correction
                    ),
                    "maximum_feedback_correction_step_rad": (
                        self.maximum_feedback_correction_step
                    ),
                }
            )
            self._save_diagnostic(diagnostic)
            return result
        except BaseException:
            diagnostic.update(
                {
                    "status": "EXCEPTION",
                    "exception": traceback.format_exc(),
                    "completed_wall_time": time.time(),
                }
            )
            self._save_diagnostic(diagnostic)
            raise


def main() -> None:
    rclpy.init()
    node = DiagnosticQuinticTrajectoryServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
