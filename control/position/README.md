# 위치 궤적 제어

`controller.py`의 `DiagnosticQuinticTrajectoryServer`는 공통 5차 궤적 서버에 위치 오차 보정과 진단 기록을 추가합니다. `control_math.py`는 ROS 의존성이 없는 수치 계산입니다.

```text
관절 궤적 → 5차 보간 → 목표 q, dq
                         ↓ 실제 관절 상태
                  EMA → deadband → 제한된 P 보정
                         ↓
                  위치 q+보정, 속도 scale*dq
```

설정은 `config/controllers/position.yaml`을 읽어 launch에서 ROS 파라미터로 전달합니다. `position_feedback_gain`은 위치 오차 비례 이득, `maximum_feedback_correction_rad`는 보정량 한계, `feedback_filter_time_constant_s`와 `feedback_deadband_rad`는 작은 떨림 억제, `velocity_feedforward_scale`은 목표 속도 전달 비율입니다.

입력 action은 로봇 프로필의 `controller.action_name`, 입력 관절 상태와 출력 명령은 각각 `state_topic`, `command_topic`입니다. 진단 JSON에는 초기 오차, 결과, 보정량 최대값을 기록합니다. 목표 도달/경로 오차/상태 수신 시간 제한은 공통 서버에서 검사합니다.

이 제어기는 위치 중심의 외부 루프입니다. 힘 또는 토크를 직접 명령하지 않으며, 실제 로봇 적용에는 제조사 인터페이스 검증이 필요합니다.
