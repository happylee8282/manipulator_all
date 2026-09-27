# Impedance control 확장 지점

상태: **실행 제어기 미지원**. `controller.main()`은 하드웨어에 명령을 보내기 전에
명시적 오류를 발생시킵니다.

| 파일 | 입력 → 출력 |
|---|---|
| `dynamics.py` | Cartesian 자세/속도 오차, 축별 K/D → spring-damper wrench |
| `safety_limit.py` | force 폴더의 공통 wrench 검사 재사용 |
| `controller.py` | 실행 요청 → 미구현 오류 |

`spring_damper_wrench`는 `K * pose_error + D * velocity_error`만 계산합니다.
위치 오차는 m, 회전 오차는 rotation vector rad, 속도는 m/s·rad/s이며 모두 같은
좌표계여야 합니다. Euler 각 차이를 회전 오차로 직접 넣지 않습니다.
결과는 힘 N와 토크 Nm의 6축 wrench이며 관절 토크 명령이 아닙니다.

실제 관절 토크 제어에는 Jacobian, 중력·동역학 보상, 센서/상태 시간 일치,
제어 주기 보장, 토크·속도 제한, 접촉 안정성 검증이 추가로 필요합니다.
위치 전용 드라이버에서는 별도 admittance 제어로 설계해야 합니다.
이 폴더의 함수는 오프라인 계산용이며 기존 위치 제어 루프에 자동 연결되지 않습니다.
