# 공통 계산과 ROS 궤적 서버

UR3e와 RV-4FL이 같은 코드 한 벌을 사용합니다. 로봇 관절 이름, 개수, 제한은 합성된 설정에서 가져옵니다.

| 파일 | 입력 | 출력/역할 |
|---|---|---|
| `geometry.py` | 모델 좌표점, 변환행렬, 면 방향 | world 좌표점, 연속 quaternion, 표면 프레임 |
| `ik_core.py` | 후보 관절각, 이전 관절각, 관절 제한 | 연속 IK 분기 비용, 결정적인 seed 변형 |
| `trajectory.py` | 관절 경유점, 경로 위치, 속도/가속도/jerk 제한 | 시간·위치·속도·가속도를 갖는 `TimedTrajectory` |
| `moveit_solver.py` | Cartesian NPZ와 합성 설정 | IK/FK/충돌 검증된 관절 NPZ와 결과 JSON |
| `base_controller.py` | `FollowJointTrajectory` 목표, 관절 상태 | 5차 보간 위치·속도를 `JointState`로 발행 |

기존 여섯 축의 IK seed 방향과 관절 비용 가중치는 유지했습니다. 다른 관절 수는 배열 크기에 맞춰 계산하며, `ik.cost.joint_weights`, `ik.cost.wrist_joint_index`, `ik.seed_perturbation_direction`으로 로봇 특성을 지정할 수 있습니다. `wrist_joint_index: null`이면 특정 손목 관절의 특이점 비용을 사용하지 않습니다. 이 경험적 비용은 Jacobian 기반 특이점 검사를 대체하지 않습니다.

단위는 위치 m, 관절각 rad, 시간 s입니다. 현재 실행 엔진의 관절은 회전 관절이며 prismatic 혼합 로봇은 별도 단위/제한 설계가 필요합니다. 관절 수 일반화가 임의 로봇의 동작 검증 완료를 의미하지는 않습니다.

`moveit_solver`는 `/compute_ik`, FK, 상태 유효성, planning scene 서비스를 사용합니다. `base_controller`는 시뮬레이션 위치 명령용 서버이며, 실제 제조사 드라이버에 임의로 연결하는 토크 제어기가 아닙니다.
