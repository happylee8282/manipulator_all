# 모델·경로·실행 검증

| 파일 | 검사 대상 |
|---|---|
| `preflight.py` | 입력 파일, 로봇 관절 이름/차원, URDF 제한, base–TCP 연결, runtime 상태 |
| `model_contract.py` | tool/base 메타데이터와 조립 URDF, SRDF group/TCP, MoveIt controller action 일치 |
| `snapshot.py` | 정착한 dynamic 안경 snapshot, stage 경로/수정 시각, 유효한 좌표 변환 |
| `non_contact_gate.py` | 작업 간격, 노즐 접촉 금지, 최신 Cartesian/IK proof, Isaac SDF 설정 |
| `glass_guard.py` | monitor freshness, dynamic/settled 상태, snapshot 일치, 이동 위반 |
| `diagnose_guard_failure.py` | guard와 제어 진단·tracking을 이용한 실패 원인 분류 |

`preflight.run(config_path, require_runtime=False)`는 ROS 없이 모델과 입력을 확인합니다. USD는 파일 존재와 프로필 정보까지 정적으로 보고하며, composition 검증 완료로 표시하지 않습니다. 실제 articulation, TCP, 물리 충돌 shape와 정착 상태는 실행 중 Isaac 어댑터의 진단으로 확인합니다.

RV-4FL은 최신 snapshot과 monitor가 없으면 `STRUCTURE_PASS_RUNTIME_PENDING`입니다. `--require-runtime`은 이 상태를 실패로 처리합니다. UR3e 정적 프로필은 `PLANNING_ONLY`를 반환하고 실제 guarded 실행은 허용하지 않습니다.

이 폴더의 gate는 기존 시뮬레이션 검증 계약입니다. 실제 로봇용 certified safety 기능이나 하드웨어 비상정지를 구현한 것이 아닙니다.

현재 URDF는 검증된 조립 모델입니다. tool의 `mount_xyz_m`, `mount_rpy_rad`, `tcp_xyz_m`를 바꾸어도 모델을 자동 재조립하지 않습니다. 모델 계약 검사는 이 값이 URDF와 다르면 명시적으로 거절합니다. 장착 위치를 바꾸려면 URDF와 tool 프로필을 함께 변경해야 합니다. UR3e의 world 고정 관절 위치도 같은 원칙을 따릅니다.
