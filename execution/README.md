# 검증된 경로 실행

`base_executor.py`는 관절 NPZ를 읽고 ROS trajectory goal, MoveIt 시작점 전이, TCP tracking 기록을 준비합니다. 외부 실행 진입점은 `guarded_executor.py`이며 반드시 `--execute`를 명시해야 합니다. launch는 사용자가 실행을 선택했을 때만 이 인자를 전달합니다.

```text
최신 Cartesian/IK/충돌 proof 확인
 → 안경 runtime guard 확인
 → initial pose → 경로 시작점
 → FollowJointTrajectory 실행과 guard 감시
 → tracking CSV/JSON 저장
```

입력: 합성 설정 `--config`, 관절 NPZ(필요 시 `--archive`), 관절 상태, TCP TF, 안경 monitor JSON. 출력: action 목표, guard phase 명령, tracking 결과. NPZ의 관절 이름·순서·차원은 선택한 로봇과 일치해야 합니다.

`compare_tracking.py`는 추종 결과를 분석하는 보조 CLI이며 `--old`로 기준 metrics JSON을 명시해야 합니다. 관절/Cartesian 목표 배열과 실제 TF를 비교하며 실제 힘 센서의 측정값을 대신하지 않습니다.

UR3e planning-only 프로필, 미지원 backend 또는 snapshot이 없는 프로필은 실행 진입점에서 거절합니다. RV-4FL의 안경 접촉 guard와 non-contact proof 검증은 유지됩니다. 그리퍼 이벤트는 별도 `gripper/` 스케줄러를 통해 정지 도착점에서 실행하며, 현재 연속 경로 중 임의 index에서 자동으로 궤적을 분할하는 기능은 없습니다.
