"""Explicit opt-in action mock; does not actuate a simulated or real device."""

from __future__ import annotations

import math
from threading import Lock
import time


def main(args=None):
    import rclpy
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    class MockGripperNode(Node):
        def __init__(self):
            super().__init__("mock_gripper")
            self.declare_parameter("allow_mock", False)
            self.declare_parameter("action_name", "/mock_gripper/gripper_cmd")
            self.declare_parameter("maximum_opening_m", 0.08)
            self.declare_parameter("mock_speed_m_s", 0.04)
            if self.get_parameter("allow_mock").value is not True:
                raise RuntimeError("Action mock disabled; explicitly set allow_mock:=true for protocol testing")
            self.maximum_opening = float(self.get_parameter("maximum_opening_m").value)
            self.speed = float(self.get_parameter("mock_speed_m_s").value)
            if not math.isfinite(self.maximum_opening) or self.maximum_opening <= 0:
                raise ValueError("maximum_opening_m must be finite and positive")
            if not math.isfinite(self.speed) or self.speed <= 0:
                raise ValueError("mock_speed_m_s must be finite and positive")
            self.position = self.maximum_opening
            self.active = False
            self.lock = Lock()
            self.server = ActionServer(
                self,
                GripperCommand,
                str(self.get_parameter("action_name").value),
                execute_callback=self.execute,
                goal_callback=self.goal,
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=ReentrantCallbackGroup(),
            )
            self.get_logger().warning("ACTION MOCK ONLY: no physics, device actuation, or force measurement")

        def goal(self, request):
            opening = request.command.position
            effort = request.command.max_effort
            if not math.isfinite(opening) or not 0 <= opening <= self.maximum_opening:
                return GoalResponse.REJECT
            if not math.isfinite(effort) or effort <= 0:
                return GoalResponse.REJECT
            with self.lock:
                if self.active:
                    return GoalResponse.REJECT
                self.active = True
            return GoalResponse.ACCEPT

        def execute(self, goal_handle):
            target = goal_handle.request.command.position
            result = GripperCommand.Result()
            previous_time = time.monotonic()
            try:
                while rclpy.ok():
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        break
                    current_time = time.monotonic()
                    delta = self.speed * (current_time - previous_time)
                    previous_time = current_time
                    self.position += max(-delta, min(delta, target - self.position))
                    if abs(target - self.position) <= 1e-6:
                        result.reached_goal = True
                        goal_handle.succeed()
                        break
                    feedback = GripperCommand.Feedback()
                    feedback.position = self.position
                    feedback.effort = 0.0
                    goal_handle.publish_feedback(feedback)
                    time.sleep(0.02)
                else:
                    goal_handle.abort()
                result.position = self.position
                result.effort = 0.0
                result.stalled = False
                return result
            finally:
                with self.lock:
                    self.active = False

    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = MockGripperNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
