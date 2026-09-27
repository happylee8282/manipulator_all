"""관절 feedforward + feedback 제어에 사용하는 순수 수치 계산 함수.

이 파일은 ROS message를 직접 발행하지 않는다. Quintic controller가 전달한 목표
관절 위치/속도와 Isaac Sim에서 받은 실제 관절 위치를 NumPy 배열로 받아, 최종
관절 위치 명령과 속도 명령을 계산한다.
"""

from __future__ import annotations

import numpy as np


def filtered_feedback_error(
    previous_error: np.ndarray,
    raw_error: np.ndarray,
    *,
    smoothing_alpha: float,
    deadband_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """관절 위치 오차를 저역통과 필터링하고 작은 센서 잡음을 제거한다.

    Args:
        previous_error: 직전 제어 주기의 필터링된 관절 오차.
        raw_error: 현재 목표 위치 - 현재 실제 위치.
        smoothing_alpha: 새 오차의 반영 비율. 1이면 필터링하지 않는다.
        deadband_rad: 이 크기 이하의 작은 오차를 제어 입력에서 제거하는 범위.

    Returns:
        ``filtered``는 다음 주기에 다시 사용할 필터 상태이고, ``active``는
        deadband까지 적용되어 실제 P feedback 계산에 사용할 오차다.
    """

    # list 등으로 전달되더라도 동일한 float NumPy 배열 연산을 사용하도록 변환한다.
    previous = np.asarray(previous_error, dtype=float)
    raw = np.asarray(raw_error, dtype=float)

    # 이전 오차와 현재 오차는 동일한 관절 개수와 배열 구조를 가져야 한다.
    if previous.shape != raw.shape:
        raise ValueError("previous and raw feedback errors must have equal shapes")

    # alpha=0이면 필터 상태가 영원히 갱신되지 않으므로 (0, 1]만 허용한다.
    if not 0.0 < smoothing_alpha <= 1.0:
        raise ValueError("smoothing alpha must be in (0, 1]")
    if deadband_rad < 0.0:
        raise ValueError("feedback deadband must be non-negative")

    # 1차 지수 이동 평균(EMA):
    # filtered[k] = filtered[k-1] + alpha * (raw[k] - filtered[k-1])
    # alpha가 작을수록 오차가 부드럽게 변해 관절 명령의 떨림을 줄인다.
    filtered = previous + smoothing_alpha * (raw - previous)

    # 대칭 deadband를 적용한다. deadband 안은 0이고, 바깥 영역은 경계에서
    # 불연속 점프가 발생하지 않도록 deadband 크기를 뺀 값만 사용한다.
    # 예: deadband=0.001, filtered=0.003이면 active=0.002 rad.
    active = np.sign(filtered) * np.maximum(np.abs(filtered) - deadband_rad, 0.0)
    return filtered, active


def corrected_joint_command(
    desired_position: np.ndarray,
    desired_velocity: np.ndarray,
    actual_position: np.ndarray,
    *,
    position_feedback_gain: float,
    maximum_correction_rad: float,
    velocity_feedforward_scale: float,
    feedback_error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """제한된 외부 P feedback과 관절속도 feedforward를 적용한다.

    위치 명령은 ``목표 위치 + 제한된 P 보정값``이고, 속도 명령은 quintic
    trajectory의 목표속도에 feedforward scale을 곱한 값이다.

    Args:
        desired_position: 현재 시각의 quintic 목표 관절 위치, rad.
        desired_velocity: 현재 시각의 quintic 목표 관절 속도, rad/s.
        actual_position: Isaac에서 측정한 실제 관절 위치, rad.
        position_feedback_gain: 위치 오차에 적용하는 P gain.
        maximum_correction_rad: 한 관절에 허용하는 feedback 보정 절댓값 상한.
        velocity_feedforward_scale: 목표 관절속도에 적용하는 배율.
        feedback_error: 외부에서 필터/deadband를 적용한 오차. 생략하면 여기서
            ``desired_position - actual_position``을 사용한다.

    Returns:
        보정된 위치 명령, feedforward 속도 명령, 실제 적용된 위치 보정값.
    """

    # 모든 입력을 float NumPy 배열로 통일한다.
    desired = np.asarray(desired_position, dtype=float)
    velocity = np.asarray(desired_velocity, dtype=float)
    actual = np.asarray(actual_position, dtype=float)

    # 위치·속도·실제 상태는 같은 관절 순서와 개수를 가져야 한다.
    if desired.shape != velocity.shape or desired.shape != actual.shape:
        raise ValueError("desired, velocity, and actual joint arrays must have equal shapes")

    # 음수 gain은 오차 반대 방향으로 로봇을 밀고, 0 이하 제한값은 유효한
    # saturation 범위를 만들 수 없으므로 허용하지 않는다.
    if position_feedback_gain < 0.0 or maximum_correction_rad <= 0.0:
        raise ValueError("feedback gain must be non-negative and correction limit positive")
    if velocity_feedforward_scale < 0.0:
        raise ValueError("velocity feedforward scale must be non-negative")

    # 일반 호출에서는 앞 함수에서 필터/deadband 처리한 feedback_error가 들어온다.
    # 단독 사용 시에는 목표 위치와 실제 위치의 차이를 직접 계산한다.
    error = desired - actual if feedback_error is None else np.asarray(feedback_error, float)
    if error.shape != desired.shape:
        raise ValueError("feedback error must have the same shape as desired position")

    # P 보정이 너무 커져 급격한 관절 명령이나 진동을 만들지 않도록 관절별로
    # ±maximum_correction_rad 범위에 제한한다.
    correction = np.clip(
        position_feedback_gain * error,
        -maximum_correction_rad,
        maximum_correction_rad,
    )

    # position: quintic 목표 위치 + bounded feedback
    # velocity: quintic 목표 속도 feedforward
    # correction: 진단 파일에 기록할 실제 적용 보정값
    return (
        desired + correction,
        velocity_feedforward_scale * velocity,
        correction,
    )
