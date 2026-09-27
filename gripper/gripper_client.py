"""Bounded ROS action calls, separate from task event validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class GripperCommandRequest:
    position_m: float
    max_effort_n: float
    timeout_s: float = 3.0
    allow_stall: bool = False

    def __post_init__(self):
        if not math.isfinite(self.position_m) or self.position_m < 0.0:
            raise ValueError("gripper opening must be finite and non-negative (m)")
        if not math.isfinite(self.max_effort_n) or self.max_effort_n <= 0.0:
            raise ValueError("maximum gripper effort must be finite and positive (N)")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("gripper timeout must be finite and positive")
        if not isinstance(self.allow_stall, bool):
            raise ValueError("allow_stall must be boolean")


@dataclass(frozen=True)
class GripperCommandResult:
    succeeded: bool
    position_m: float
    effort_n: float
    reached_goal: bool
    stalled: bool

    def accepted_for(self, request: GripperCommandRequest) -> bool:
        return bool(
            self.succeeded
            and math.isfinite(self.position_m)
            and math.isfinite(self.effort_n)
            and self.position_m >= 0.0
            and (self.reached_goal or (request.allow_stall and self.stalled))
        )


class GripperClient:
    """Blocking client for a dedicated node, never call within a ROS callback.

    The caller must provide a node not already attached to an executor. A command
    only returns after its action result is received. A timeout requests cancel
    and raises; the task must stay stopped even if cancellation is unconfirmed.
    """

    def __init__(self, node, action_name: str):
        from control_msgs.action import GripperCommand
        from rclpy.action import ActionClient

        if not action_name:
            raise ValueError("a gripper action name is required")
        self.node = node
        self.client = ActionClient(node, GripperCommand, action_name)
        self.action_type = GripperCommand
        self.recovery_required = False

    def _wait(self, future, deadline: float, label: str):
        import rclpy

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"gripper {label} timed out")
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=remaining)
        if not future.done():
            raise TimeoutError(f"gripper {label} timed out")
        return future.result()

    def command(self, request: GripperCommandRequest) -> GripperCommandResult:
        from action_msgs.msg import GoalStatus

        if self.recovery_required:
            raise RuntimeError("previous gripper timeout requires device-state recovery")
        if self.node.executor is not None:
            raise RuntimeError("GripperClient needs a dedicated node outside an executor")
        deadline = time.monotonic() + request.timeout_s
        if not self.client.wait_for_server(timeout_sec=request.timeout_s):
            raise TimeoutError("gripper action server is unavailable")
        if time.monotonic() >= deadline:
            raise TimeoutError("gripper deadline expired before sending a goal")
        goal = self.action_type.Goal()
        goal.command.position = request.position_m
        goal.command.max_effort = request.max_effort_n
        goal_handle = None
        goal_future = self.client.send_goal_async(goal)
        try:
            goal_handle = self._wait(goal_future, deadline, "goal acknowledgement")
            if not goal_handle.accepted:
                raise RuntimeError("gripper goal was rejected")
            response = self._wait(goal_handle.get_result_async(), deadline, "result")
        except TimeoutError:
            self.recovery_required = True
            if goal_handle is not None and goal_handle.accepted:
                self._cancel(goal_handle)
            else:
                goal_future.add_done_callback(self._cancel_late_goal)
                self.node.get_logger().error(
                    "Gripper acknowledgement missing; actuation/cancellation is unconfirmed. "
                    "Keep the task stopped and check device state."
                )
            raise
        result = response.result
        return GripperCommandResult(
            succeeded=response.status == GoalStatus.STATUS_SUCCEEDED,
            position_m=float(result.position),
            effort_n=float(result.effort),
            reached_goal=bool(result.reached_goal),
            stalled=bool(result.stalled),
        )

    def _cancel(self, goal_handle):
        try:
            self._wait(goal_handle.cancel_goal_async(), time.monotonic() + 1.0, "cancel")
        except (TimeoutError, RuntimeError) as error:
            self.node.get_logger().error(f"Gripper cancellation unconfirmed: {error}")

    def _cancel_late_goal(self, future):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
        except Exception as error:
            self.node.get_logger().error(f"Late gripper cancellation failed: {error}")

    def close(self):
        self.client.destroy()
