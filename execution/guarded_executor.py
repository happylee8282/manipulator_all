"""Execute a robot-profile path while guarding dynamically settled glasses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from . import base_executor as implementation

from ..configuration import (
    DEFAULT_CONFIG,
    output_path,
    resolve_path,
    load_config,
)
from ..safety.non_contact_gate import validate_non_contact_artifacts
from ..safety.glass_guard import read_guard_status, write_guard_command


class GuardedSolvedPathExecutor(implementation.SolvedPathExecutor):
    """Cancel motion if Isaac reports displacement of the settled glasses."""

    def __init__(self, config_path: Path, archive_path: Path | None = None) -> None:
        self.config_path = config_path.expanduser().resolve()
        config, _ = load_config(self.config_path)
        if bool(config.get("robot", {}).get("dry_run_only", False)):
            raise RuntimeError("selected robot is planning-only; guarded execution requires a commissioned scene/guard adapter")
        if config.get("backend", {}).get("supported") is False:
            raise RuntimeError("selected backend is not implemented")
        if not config.get("frame_transform", {}).get("require_glass_snapshot", True):
            raise RuntimeError("guarded execution requires a settled dynamic-glass snapshot")
        super().__init__(self.config_path, archive_path)
        settings = self.config["glass_guard"]
        self.guard_status_path = resolve_path(self.root, settings["status_file"])
        self.guard_command_path = resolve_path(self.root, settings["command_file"])
        self.controller_diagnostics_path = resolve_path(
            self.root, self.config["controller"]["diagnostics_file"]
        )
        self.guard_timeout = float(settings["status_timeout_s"])
        self.abort_on_violation = bool(settings["abort_on_violation"])
        self.guard_failure: str | None = None
        self.guard_monitoring = False
        self.guard_phase = "IDLE"
        self.last_guard_command_write = 0.0
        self.active_goal_handle = None
        self.cancel_requested = False
        self.expected_snapshot_time = self.config.get("runtime_snapshot", {}).get(
            "captured_wall_time"
        )
        self.create_timer(1.0 / float(settings["poll_rate_hz"]), self._poll_guard)

    def _read_guard(self) -> dict[str, Any]:
        return read_guard_status(
            self.guard_status_path,
            timeout_s=self.guard_timeout,
            expected_snapshot_time=self.expected_snapshot_time,
        )

    def _set_guard_command(self, active: bool, phase: str) -> None:
        self.guard_phase = phase
        self.last_guard_command_write = time.monotonic()
        write_guard_command(
            self.guard_command_path,
            active=active,
            phase=phase,
        )

    def _poll_guard(self) -> None:
        if not self.guard_monitoring or self.guard_failure is not None:
            return
        # Keep the Isaac-side execution classification alive during paths
        # that run for several minutes.
        if time.monotonic() - self.last_guard_command_write >= 0.5:
            self._set_guard_command(True, self.guard_phase)
        try:
            self._read_guard()
        except RuntimeError as error:
            self.guard_failure = str(error)
            self.get_logger().error(f"GLASS INTERLOCK: {self.guard_failure}")
            if (
                self.abort_on_violation
                and self.active_goal_handle is not None
                and not self.cancel_requested
            ):
                self.cancel_requested = True
                self.active_goal_handle.cancel_goal_async()

    def _assert_guard(self) -> dict[str, Any]:
        if self.guard_failure is not None:
            raise RuntimeError(self.guard_failure)
        return self._read_guard()

    def _controller_failure_detail(self) -> str:
        try:
            diagnostic = json.loads(
                self.controller_diagnostics_path.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return ""
        return (
            f"; controller={diagnostic.get('result_error_string')}"
            f"; initial_error={diagnostic.get('initial_max_error_rad')} rad"
        )

    def _verify_joint_target(
        self, target: np.ndarray, tolerance: float, label: str
    ) -> None:
        deadline = time.monotonic() + 10.0
        error = float("inf")
        while time.monotonic() < deadline:
            actual = self._wait_for_state()
            error = float(np.max(np.abs(actual - target)))
            if error <= tolerance:
                self.get_logger().info(
                    f"{label} verified: max joint error={error:.6f} rad"
                )
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            f"robot did not reach {label} within {tolerance:g} rad; "
            f"last max error={error:.6f} rad"
        )

    def _move_to_joint_target_guarded(
        self,
        target: np.ndarray,
        label: str,
        velocity_scaling: float,
        acceleration_scaling: float,
        already_there_tolerance: float,
    ) -> None:
        settings = self.config["execution"]
        current = self._wait_for_state()
        initial_error = float(np.max(np.abs(current - target)))
        if initial_error <= already_there_tolerance:
            self.get_logger().info(
                f"{label} already active: max joint error={initial_error:.6f} rad"
            )
            return
        if not self.move_group.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("MoveGroup action is unavailable")
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = str(self.config["project"]["planning_group"])
        request.pipeline_id = "ompl"
        request.num_planning_attempts = 8
        request.allowed_planning_time = 8.0
        request.max_velocity_scaling_factor = float(velocity_scaling)
        request.max_acceleration_scaling_factor = float(acceleration_scaling)
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
            raise RuntimeError(f"MoveGroup rejected the transition to {label}")
        self.active_goal_handle = handle
        self.cancel_requested = False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=float(settings["move_group_timeout_s"])
        )
        self.active_goal_handle = None
        self._assert_guard()
        result = result_future.result() if result_future.done() else None
        code = result.result.error_code.val if result is not None else None
        if code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f"MoveGroup transition to {label} failed with code {code}"
                f"{self._controller_failure_detail()}"
            )

    def _return_to_initial_pose_guarded(self) -> None:
        settings = self.config["execution"]
        if not bool(settings["return_to_initial_pose"]):
            return
        target = np.asarray(settings["initial_pose_rad"], dtype=float)
        if target.shape != (len(self.joint_names),) or not np.all(np.isfinite(target)):
            raise RuntimeError("execution.initial_pose_rad must match the robot's joint count with finite values")
        tolerance = float(settings["require_initial_pose_tolerance_rad"])
        self._move_to_joint_target_guarded(
            target,
            "INITIAL_POSE",
            float(settings["initial_pose_velocity_scaling"]),
            float(settings["initial_pose_acceleration_scaling"]),
            tolerance,
        )
        self._verify_joint_target(target, tolerance, "INITIAL_POSE")

    def _move_to_start_guarded(self) -> None:
        settings = self.config["execution"]
        if not bool(settings["move_to_start_with_move_group"]):
            return
        self._move_to_joint_target_guarded(
            self.timed.positions[0],
            "PATH_START",
            float(settings["start_velocity_scaling"]),
            float(settings["start_acceleration_scaling"]),
            float(settings["require_start_tolerance_rad"]),
        )

    def run(self) -> dict[str, Any]:
        if not bool(self.config["execution"].get("execute", False)):
            raise RuntimeError("motion requires the explicit --execute option")
        proof = validate_non_contact_artifacts(
            self.config, self.root, self.config_path
        )
        if bool(proof.get("required", False)):
            self.get_logger().info(
                "NON_CONTACT_GATE PASS: clearance="
                f"{proof['generated_clearance_min_mm']:.3f}.."
                f"{proof['generated_clearance_max_mm']:.3f} mm, "
                "nozzle-glass collision disabled, "
                f"Isaac SDF>={proof['isaac_sdf']['minimum_resolution']}"
            )
        self._assert_guard()
        self.guard_monitoring = True
        try:
            self._wait_for_state()
            self._set_guard_command(True, "RETURN_TO_INITIAL_POSE")
            self._return_to_initial_pose_guarded()
            self._assert_guard()
            self._set_guard_command(True, "MOVE_TO_PATH_START")
            self._move_to_start_guarded()
            self._verify_start()
            self._assert_guard()
            if not self.controller.wait_for_server(timeout_sec=15.0):
                raise RuntimeError("quintic FollowJointTrajectory server is unavailable")
            self._set_guard_command(True, "PROCESS_PATH")
            self.controller_elapsed = None
            future = self.controller.send_goal_async(
                self._goal(), feedback_callback=self._controller_feedback
            )
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            handle = future.result() if future.done() else None
            if handle is None or not handle.accepted:
                raise RuntimeError("quintic controller rejected the verified trajectory")
            self.active_goal_handle = handle
            self.cancel_requested = False
            self.tracking_rows.clear()
            self.tracking_started = time.monotonic()
            self.tracking_active = True
            result_future = handle.get_result_async()
            timeout = float(self.timed.time_s[-1]) + float(
                self.config["controller"]["goal_time_tolerance_s"]
            ) + 10.0
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
            self.tracking_active = False
            self.active_goal_handle = None
            self._assert_guard()
            wrapped = result_future.result() if result_future.done() else None
            if wrapped is None:
                raise RuntimeError("controller action timed out")
            if wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                raise RuntimeError(
                    f"trajectory execution failed: {wrapped.result.error_string}"
                )
            metrics = self._save_tracking()
            status = self._read_guard()
            metrics["glass_guard"] = {
                "status": "PASS_STATIONARY",
                "max_translation_mm": status.get("maximum_translation_mm", 0.0),
                "max_rotation_deg": status.get("maximum_rotation_deg", 0.0),
            }
            output_path(self.config, self.root, "tracking_metrics").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            self.get_logger().info(
                "Execution PASS with stationary glasses: TCP p95="
                f"{metrics['tcp_position_error_mm']['p95']:.6f} mm"
            )
            return metrics
        finally:
            self.tracking_active = False
            self.active_goal_handle = None
            self.guard_monitoring = False
            self._set_guard_command(False, "IDLE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize the guarded trajectory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit("motion is disabled; supply --execute after planning validation")
    rclpy.init()
    node = GuardedSolvedPathExecutor(args.config, args.archive)
    node.config["execution"]["execute"] = True
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
