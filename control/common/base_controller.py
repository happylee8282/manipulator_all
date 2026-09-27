#!/usr/bin/env python3
# ROS2 FollowJointTrajectory로 받은 로봇팔의 경유점들을 5차 Hermite 보간하여 부드러운
# 시간 기반 관절 궤적으로 만들고, 120 Hz로 Isaac에 전달하면서 실제 관절 위치가 목표 궤적을 
# 제대로 추종하고 있는지 감시하는 궤적 실행 서버입니다.
# 사용 방식: 5차 Hermite 다항식 보간(Quintic Hermite Interpolation) 기반 관절 궤적 실행.
# 입력: 각 경유점의 시간 t, 관절 위치 q, 속도 qdot, 가속도 qddot(프로필의 관절 수).
# 계산: 두 경유점의 위치·속도·가속도 조건 6개를 만족하는 5차 다항식을 관절별로 구성한다.
#       정규화 시간 u=(t-t_i)/(t_(i+1)-t_i)에 대해 q(u)=c0+c1*u+...+c5*u^5.
#       내부 구간 연결부에서 위치·속도·가속도가 연속인 C² 궤적을 만든다.
# 출력: 기본 120 Hz로 목표 위치와 선택적으로 목표 속도를 Isaac에 전송한다.
# 감시: 최대 관절 위치 오차 max_j|q_desired,j-q_actual,j|로 중단 및 도달 여부를 판정한다.
# 역할: ROS 2 FollowJointTrajectory 액션 서버이며, PID/토크 제어는 이 파일에서 수행하지 않는다.
# 실제 다항식 계수 계산과 평가는 control/common/trajectory.py의 함수를 호출한다.

"""FollowJointTrajectory server that preserves q, qdot, and qddot.

The legacy Isaac bridge used linear interpolation and ignored velocities and
accelerations supplied by the planner.  This server evaluates the C2 quintic
Hermite segment implied by every two trajectory points and aborts on sustained
path-tolerance violations.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Optional, Sequence

import numpy as np

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from .trajectory import TimedTrajectory, sample_timed_trajectory


DEFAULT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


# ROS Duration을 초 단위 실수로 변환한다.
# t = sec + nanosec × 10^-9.
def duration_seconds(value: Duration) -> float:
    return float(value.sec) + float(value.nanosec) * 1.0e-9


# 초 단위 실수를 ROS Duration으로 변환한다.
# N = max(0, round(t × 10^9)), sec = N // 10^9, nanosec = N % 10^9.
# 음수 시간은 0으로 제한하고 나노초 단위로 반올림한다.
def duration_message(seconds: float) -> Duration:
    nanoseconds = max(0, int(round(seconds * 1.0e9)))
    return Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


class QuinticTrajectoryServer(Node):
    # 프로필 관절의 목표 궤적을 발행하고 실제 위치를 감시하는 ROS 서버를 초기화한다.
    # q ∈ R^J이며 위치/속도/가속도의 단위는 rad, rad/s, rad/s^2이다.
    # 기본 주파수 f=120 Hz → 발행 주기 Δt=1/f≈8.33 ms.
    # 기본 경로 오차 한계 ε_path=0.020 rad, 연속 초과 유예 시간=0.25 s.
    # 기본 종점 오차 한계 ε_goal=0.008 rad, 종점 대기 시간=5 s.
    # 상태 수신 경과 시간 한계는 기본 1 s이며, 통신 채널과 동시 접근 잠금을 설정한다.
    # 이 서버는 목표 위치/속도를 생성하며 PID 보정이나 토크 계산은 하지 않는다.
    def __init__(self) -> None:
        super().__init__("quintic_isaac_joint_trajectory_server")
        self.declare_parameter("joint_names", list(DEFAULT_JOINT_NAMES))
        self.declare_parameter("command_topic", "/isaac_joint_command")
        self.declare_parameter("state_topic", "/isaac_joint_states")
        self.declare_parameter(
            "action_name", "/isaac_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.declare_parameter("publish_rate_hz", 120.0)
        self.declare_parameter("publish_velocity_commands", True)
        self.declare_parameter("state_timeout_s", 1.0)
        self.declare_parameter("path_tolerance_rad", 0.020)
        self.declare_parameter("path_tolerance_grace_s", 0.25)
        self.declare_parameter("goal_tolerance_rad", 0.008)
        self.declare_parameter("goal_time_tolerance_s", 5.0)

        self.joint_names = tuple(map(str, self.get_parameter("joint_names").value))
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must contain nonempty unique joints")
        if not all(self.joint_names):
            raise ValueError("joint_names must not contain empty names")
        self.rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.publish_velocity = bool(self.get_parameter("publish_velocity_commands").value)
        self.state_timeout = float(self.get_parameter("state_timeout_s").value)
        self.path_tolerance = float(self.get_parameter("path_tolerance_rad").value)
        self.path_grace = float(self.get_parameter("path_tolerance_grace_s").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance_rad").value)
        self.goal_time_tolerance = float(self.get_parameter("goal_time_tolerance_s").value)
        if min(self.rate_hz, self.state_timeout, self.path_tolerance, self.goal_tolerance) <= 0.0:
            raise ValueError("controller rates, timeouts, and tolerances must be positive")

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.callback_group = ReentrantCallbackGroup()
        self.command_publisher = self.create_publisher(
            JointState, str(self.get_parameter("command_topic").value), reliable
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("state_topic").value),
            self._state_callback,
            best_effort,
            callback_group=self.callback_group,
        )
        self.state_lock = threading.Lock()
        self.actual: Optional[Dict[str, float]] = None
        self.actual_received = 0.0
        self.goal_lock = threading.Lock()
        self.goal_active = False
        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("action_name").value),
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            f"Quintic controller ready at {self.rate_hz:g} Hz; "
            f"path tolerance={self.path_tolerance:g} rad"
        )

    # 수신한 관절 상태를 검증하고 현재 위치 q_actual을 저장한다.
    # 관절 이름과 위치 개수가 일치하고, 필요한 6개 관절의 값이 모두 유한해야 한다.
    # 잠금 안에서 위치와 수신 시각 t_received를 함께 갱신한다.
    def _state_callback(self, message: JointState) -> None:
        if len(message.name) != len(message.position):
            return
        incoming = dict(zip(message.name, message.position))
        if not all(name in incoming for name in self.joint_names):
            return
        values = {name: float(incoming[name]) for name in self.joint_names}
        if not all(math.isfinite(value) for value in values.values()):
            return
        with self.state_lock:
            self.actual = values
            self.actual_received = time.monotonic()

    # 내부 관절 순서로 정렬한 q_actual ∈ R^6와 상태 수신 경과 시간을 반환한다.
    # age = t_monotonic_now - t_received이며 센서 측정 시각 기준의 지연은 아니다.
    # 수신 이력이 없으면 (None, ∞)를 반환한다.
    def _actual_values(self) -> tuple[np.ndarray | None, float]:
        with self.state_lock:
            if self.actual is None:
                return None, math.inf
            values = np.asarray([self.actual[name] for name in self.joint_names], float)
            age = time.monotonic() - self.actual_received
        return values, age

    # 입력 경유점 (t_i, q_i, v_i, a_i)의 형식과 시간 순서를 검사한다.
    # 경유점은 2개 이상이며 t_i≥0, t_(i+1)>t_i → 구간 길이 h_i>0이어야 한다.
    # q_i는 관절 수만큼 필요하고 v_i, a_i는 각각 생략하거나 같은 수를 제공해야 한다.
    # 관절 이름 집합과 값의 유한성을 검사하며, 오류가 없으면 None을 반환한다.
    # 속도/가속도의 전체 경유점 간 제공 여부 일관성은 _trajectory에서 검사한다.
    def _validate_goal(self, goal: FollowJointTrajectory.Goal) -> str | None:
        trajectory = goal.trajectory
        if len(trajectory.joint_names) != len(self.joint_names) or set(trajectory.joint_names) != set(self.joint_names):
            return "trajectory joint names do not match the configured controller joints"
        if len(trajectory.points) < 2:
            return "trajectory needs at least two points"
        previous = -1.0
        count = len(self.joint_names)
        for index, point in enumerate(trajectory.points):
            point_time = duration_seconds(point.time_from_start)
            if point_time < 0.0 or point_time <= previous:
                if not (index == 0 and point_time == 0.0):
                    return f"point {index} has non-increasing time"
            if len(point.positions) != count:
                return f"point {index} has the wrong position count"
            if len(point.velocities) not in (0, count):
                return f"point {index} has an incomplete velocity array"
            if len(point.accelerations) not in (0, count):
                return f"point {index} has an incomplete acceleration array"
            values = list(point.positions) + list(point.velocities) + list(point.accelerations)
            if not all(math.isfinite(value) for value in values):
                return f"point {index} contains a non-finite value"
            previous = point_time
        return None

    # 입력 검증을 통과하고 실행 중인 목표가 없을 때만 새 궤적을 수락한다.
    # goal_active를 잠금 안에서 검사·설정하여 동시에 실행되는 목표 수를 최대 1로 제한한다.
    def _goal_callback(self, goal: FollowJointTrajectory.Goal) -> GoalResponse:
        error = self._validate_goal(goal)
        if error:
            self.get_logger().error(f"Rejecting trajectory: {error}")
            return GoalResponse.REJECT
        with self.goal_lock:
            if self.goal_active:
                return GoalResponse.REJECT
            self.goal_active = True
        return GoalResponse.ACCEPT

    # 취소 요청을 수락한다. 이 콜백 자체는 정지 명령을 발행하지 않는다.
    # _execute의 궤적 실행 루프가 취소를 확인하면 q_cmd=q_actual, v_cmd=0을 요청한다.
    @staticmethod
    def _cancel_callback(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    # ROS 경유점을 시간 배열과 위치·속도·가속도 행렬 Q,V,A ∈ R^(N×J)로 구성한다.
    # 각 행은 하나의 경유점이며 모든 벡터를 내부 관절 순서에 맞춰 재배열한다.
    # 첫 시간이 10^-12 s보다 크면 (t=0, q=actual_start, v=0, a=0)을 앞에 추가한다.
    # 속도/가속도는 각각 모든 경유점에서 제공하거나 모두 생략해야 한다.
    # 제공된 미분값은 보존하고, 생략된 값은 np.gradient로 v≈dq/dt, a≈dv/dt를 구한다.
    # 등간격 내부점: v_i=(q_(i+1)-q_(i-1))/(2Δt). 비등간격에는 간격별 가중치를 쓴다.
    # 추정한 속도/가속도는 각각 양 끝 값을 0으로 덮어써 정지 경계 조건을 부여한다.
    # 반환 객체의 최대 속도/가속도/jerk=0은 자리표시자이며 시간 배율은 1이다.
    # 이 함수는 운동 한계를 검증하거나 궤적 시간을 재조정하지 않는다.
    def _trajectory(
        self, goal: FollowJointTrajectory.Goal, actual_start: np.ndarray
    ) -> TimedTrajectory:
        incoming_names = list(goal.trajectory.joint_names)
        order = [incoming_names.index(name) for name in self.joint_names]

        # 관절 순열 P를 적용하여 입력 벡터 x를 내부 순서의 P x로 재배열한다.
        # 위치·속도·가속도에 동일한 순열을 적용하여 각 열의 관절을 일치시킨다.
        def ordered(values: Sequence[float]) -> list[float]:
            return [float(values[index]) for index in order]

        positions = np.asarray([ordered(point.positions) for point in goal.trajectory.points])
        time_s = np.asarray(
            [duration_seconds(point.time_from_start) for point in goal.trajectory.points]
        )
        velocity_supplied = all(len(point.velocities) == len(self.joint_names) for point in goal.trajectory.points)
        acceleration_supplied = all(
            len(point.accelerations) == len(self.joint_names)
            for point in goal.trajectory.points
        )
        if any(len(point.velocities) for point in goal.trajectory.points) and not velocity_supplied:
            raise ValueError("velocity arrays must be present at every point or omitted everywhere")
        if any(len(point.accelerations) for point in goal.trajectory.points) and not acceleration_supplied:
            raise ValueError("acceleration arrays must be present at every point or omitted everywhere")
        if time_s[0] > 1.0e-12:
            positions = np.vstack((actual_start, positions))
            time_s = np.r_[0.0, time_s]
        if velocity_supplied:
            velocities = np.asarray(
                [ordered(point.velocities) for point in goal.trajectory.points]
            )
            if len(velocities) < len(positions):
                velocities = np.vstack((np.zeros(len(self.joint_names)), velocities))
        else:
            edge_order = 2 if len(positions) > 2 else 1
            velocities = np.gradient(positions, time_s, axis=0, edge_order=edge_order)
            velocities[0] = 0.0
            velocities[-1] = 0.0
        if acceleration_supplied:
            accelerations = np.asarray(
                [ordered(point.accelerations) for point in goal.trajectory.points]
            )
            if len(accelerations) < len(positions):
                accelerations = np.vstack((np.zeros(len(self.joint_names)), accelerations))
        else:
            edge_order = 2 if len(velocities) > 2 else 1
            accelerations = np.gradient(velocities, time_s, axis=0, edge_order=edge_order)
            accelerations[0] = 0.0
            accelerations[-1] = 0.0
        zeros = np.zeros(len(self.joint_names))
        return TimedTrajectory(
            positions,
            velocities,
            accelerations,
            time_s,
            zeros,
            zeros,
            zeros,
            1.0,
        )

    # 목표 관절각 q_cmd=position을 JointState로 발행한다.
    # 옵션이 켜져 있으면 목표 속도 v_cmd=velocity도 발행하며 가속도는 전송하지 않는다.
    # 메시지 타임스탬프에는 ROS 시계를 사용한다.
    def _publish_command(self, position: np.ndarray, velocity: np.ndarray) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.joint_names)
        message.position = position.tolist()
        if self.publish_velocity:
            message.velocity = velocity.tolist()
        self.command_publisher.publish(message)

    # 목표 위치/속도와 실제 위치, 관절별 부호 있는 오차를 피드백으로 보낸다.
    # e(t)=q_d(t)-q_actual(t) ∈ R^6이며 실제 속도/가속도는 이 함수에서 보고하지 않는다.
    # 각 time_from_start에는 전달받은 궤적 경과 시간을 기록한다.
    def _feedback(
        self,
        goal_handle,
        elapsed: float,
        desired_position: np.ndarray,
        desired_velocity: np.ndarray,
        actual: np.ndarray,
    ) -> None:
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(self.joint_names)
        feedback.desired.positions = desired_position.tolist()
        feedback.desired.velocities = desired_velocity.tolist()
        feedback.desired.time_from_start = duration_message(elapsed)
        feedback.actual.positions = actual.tolist()
        feedback.actual.time_from_start = duration_message(elapsed)
        feedback.error.positions = (desired_position - actual).tolist()
        feedback.error.time_from_start = duration_message(elapsed)
        goal_handle.publish_feedback(feedback)

    # 궤적을 시간에 따라 보간·발행하고 추종 오차 및 최종 도달 여부를 판정한다.
    # ① 시작 검사: 신선한 상태와 ||q_actual-q_start||_∞≤ε_path를 요구한다.
    #    무한대 노름 ||e||_∞=max_j|e_j|이며 시작 오차에는 유예 시간을 적용하지 않는다.
    # ② 보간: trajectory.py의 sample_timed_trajectory로 각 구간을 5차 Hermite 보간한다.
    #    h=t_(i+1)-t_i, u=(t-t_i)/h ∈ [0,1], p(u)=c0+c1*u+...+c5*u^5.
    #    p(0)=q_i, p(1)=q_(i+1), p′(0)=h*v_i, p′(1)=h*v_(i+1),
    #    p″(0)=h²*a_i, p″(1)=h²*a_(i+1)의 6개 조건으로 계수 6개를 결정한다.
    #    c0=q_i, c1=h*v_i, c2=h²*a_i/2이며 나머지는 종점 조건으로 결정된다.
    #    실제 시간 미분은 v_d=p′(u)/h, a_d=p″(u)/h², jerk=p‴(u)/h³이다.
    #    인접 구간은 같은 q,v,a를 공유하므로 내부 접합점에서 C² 연속이나 jerk 연속은 보장하지 않는다.
    #    반환된 가속도/jerk는 사용하지 않고 목표 위치와 속도만 발행한다.
    # ③ 시간 진행: monotonic 시계로 t=min(now-started, t_final)을 계산한다.
    #    ROS 시뮬레이션 시간과 별개이며 예정 발행 시각을 Δt=1/f씩 증가시킨다.
    # ④ 경로 감시: E=||q_d-q_actual||_∞가 ε_path를 연속으로 grace 이상 초과하면 중단한다.
    #    허용 범위로 돌아오는 샘플이 있으면 초과 시간 타이머를 초기화한다.
    #    지속 오차 또는 실행 루프에서 확인한 취소에는 q_cmd=q_actual, v_cmd=0을 요청한다.
    #    이는 위치 유지 요청이며 물리적인 순간 정지를 보장하지 않는다.
    #    실행 중 상태가 오래되면 중단한다. 해당 분기에서는 별도 유지 명령을 발행하지 않는다.
    # ⑤ 종점 대기: q_cmd=q_final, v_cmd=0을 유지하며 goal_time_tolerance 동안 기다린다.
    #    종점 입력 속도가 0이 아니면 이 단계로 넘어갈 때 속도 명령이 불연속일 수 있다.
    #    신선한 상태에서 ||q_final-q_actual||_∞≤ε_goal이면 즉시 성공한다.
    #    실제 속도/가속도 수렴이나 허용 범위 유지 시간은 검사하지 않는다.
    #    종점 대기 루프에는 경로 오차 유예 검사와 취소 요청 검사가 없다.
    #    기한 내 도달하지 못하면 실패하며, 종료 시 항상 실행 중 표시를 해제한다.
    async def _execute(self, goal_handle):
        result = FollowJointTrajectory.Result()
        try:
            actual, age = self._actual_values()
            if actual is None or age > self.state_timeout:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "fresh Isaac joint state is unavailable"
                goal_handle.abort()
                return result
            trajectory = self._trajectory(goal_handle.request, actual)
            start_error = float(np.max(np.abs(actual - trajectory.positions[0])))
            if start_error > self.path_tolerance:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = (
                    f"actual start differs from trajectory by {start_error:.6f} rad"
                )
                goal_handle.abort()
                return result

            period = 1.0 / self.rate_hz
            started = time.monotonic()
            next_publish = started
            violation_started: float | None = None
            while True:
                now = time.monotonic()
                elapsed = min(now - started, float(trajectory.time_s[-1]))
                position, velocity, _, _ = sample_timed_trajectory(trajectory, elapsed)
                self._publish_command(position, velocity)
                actual, age = self._actual_values()
                if actual is None or age > self.state_timeout:
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    result.error_string = "Isaac joint state became stale"
                    goal_handle.abort()
                    return result
                self._feedback(goal_handle, elapsed, position, velocity, actual)
                error = float(np.max(np.abs(position - actual)))
                if error > self.path_tolerance:
                    violation_started = now if violation_started is None else violation_started
                    if now - violation_started >= self.path_grace:
                        self._publish_command(actual, np.zeros_like(actual))
                        result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                        result.error_string = (
                            f"sustained path error {error:.6f} rad exceeds "
                            f"{self.path_tolerance:.6f} rad"
                        )
                        goal_handle.abort()
                        return result
                else:
                    violation_started = None
                if goal_handle.is_cancel_requested:
                    self._publish_command(actual, np.zeros_like(actual))
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    result.error_string = "trajectory canceled and held"
                    return result
                if elapsed >= trajectory.time_s[-1]:
                    break
                next_publish += period
                delay = next_publish - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)

            final = trajectory.positions[-1]
            deadline = time.monotonic() + self.goal_time_tolerance
            while time.monotonic() < deadline:
                self._publish_command(final, np.zeros_like(final))
                actual, age = self._actual_values()
                if actual is not None and age <= self.state_timeout:
                    error = float(np.max(np.abs(final - actual)))
                    if error <= self.goal_tolerance:
                        goal_handle.succeed()
                        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                        result.error_string = f"goal reached; max joint error={error:.6f} rad"
                        return result
                time.sleep(period)
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = "final joint state did not settle inside goal tolerance"
            return result
        finally:
            with self.goal_lock:
                self.goal_active = False


# ROS를 초기화하고 서버를 4개 스레드의 실행기에 등록하여 콜백을 처리한다.
# 종료 시 실행기, 노드, ROS 자원을 정리한다. 별도의 수치 계산은 수행하지 않는다.
def main() -> None:
    rclpy.init()
    node = QuinticTrajectoryServer()
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
