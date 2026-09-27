# Lee Manipulator Part

UR3e와 Mitsubishi RV-4FL의 경로 생성·MoveIt IK·관절 궤적·위치 제어를 한 ROS 2
패키지로 정리한 작업 폴더다. ROS 패키지와 Python import 이름은
`lee_manipulator_part`다. 실행은 `ros2 launch`를 사용한다.

기존 `step1_modelsolution`과 `lee_ws`의 실행 코드를 import하지 않는다.
URDF·MoveIt 설정·필요한 로봇 메시·기준 경로 CSV/PCD는 이 폴더에 포함했다.
Isaac 환경 USD와 원본 대용량 PCD는 기존 작업 폴더의 자산을 사용한다.
다른 PC로 옮길 때 필요한 자산 목록은 [사용 매뉴얼](docs/USER_MANUAL.md)에 있다.

## 현재 제공 범위

| 선택 | 현재 상태 |
| --- | --- |
| `robot:=rv4fl` 또는 `mizb` | 기존 동적 안경 감시·비접촉 검사·quintic 위치 제어 이관 |
| `robot:=ur3e` 또는 `ur` | 기존 UR3e/노즐 모델·Isaac 장면·경로 생성·MoveIt 계획. 동적 안경 감시 이식 전까지 실행 차단 |
| `controller:=position` | 관절 위치 + 속도 feedforward + 제한된 필터링 P 피드백 |
| 그리퍼 | action client, 정지 waypoint 이벤트 API, 명시적 통신 mock. 현재 노즐은 고정 도구 |
| `backend:=real` | 드라이버 연결 위치와 매뉴얼 제공. 실장비 실행은 미구현으로 차단 |
| `controller:=force/impedance` | 수치 보조 함수와 폴더 제공. 제어 루프 활성화는 차단 |

구조·단위 테스트·ROS launch 검증과 Isaac 전체 경로 완주는 별개의 검증이다.
이 새 패키지로 시뮬레이션을 완주했다는 의미는 아니다.

## 최초 빌드

ROS Humble용 시스템 Python으로 빌드한다. Conda 환경이 활성화돼 있으면 해제한다.

```bash
cd /home/happy/modelsoultion/lee_manipulator_part
source /opt/ros/humble/setup.bash
colcon build --packages-select lee_manipulator_part
source install/setup.bash
```

## RV-4FL 실행

터미널 1: Isaac 장면·MoveIt·위치 controller 시작. 작업 궤적 실행은 기본 꺼져 있다.
Isaac runner를 시작하면 기존 동작과 같이 Play 및 초기 자세 설정·안경 안정화를 수행한다.

```bash
ros2 launch lee_manipulator_part system.launch.py \
  robot:=rv4fl start_isaac:=true
```

터미널 2에서도 ROS와 이 패키지의 `install/setup.bash`를 source한 뒤 실행한다.

```bash
# 최신 안경 snapshot → TCP 경로 → 전체 IK/FK/충돌 검사
ros2 launch lee_manipulator_part task.launch.py robot:=rv4fl

# 준비 과정을 다시 수행하고 통과하면 로봇 궤적 실행
ros2 launch lee_manipulator_part task.launch.py robot:=rv4fl execute:=true
```

기존 Step 1 솔버와 이 패키지를 같은 ROS domain에서 동시에 켜지 않는다.
RV-4FL 기본 domain은 31, UR3e는 32다. 변경 시 모든 터미널에 같은
`ros_domain_id:=...` 값을 지정한다. 동일 run에 서로 다른 설정을 적용하려면 먼저
그 run의 launch를 종료한다.

## UR3e 실행

```bash
ros2 launch lee_manipulator_part system.launch.py robot:=ur3e start_isaac:=true
ros2 launch lee_manipulator_part task.launch.py robot:=ur3e
```

UR3e는 계획 검토용이다. 고정 좌표변환을 쓰므로 실제 안경 배치가 같은지 확인해야 한다.
이 프로필에서는 위치 명령 서버와 MoveIt 실행을 활성화하지 않으며 `execute:=true`도
거절한다. RV-4FL의 동적 안경 guard를 UR3e에 연결·검증한 뒤 실행 범위를 확장한다.

## 폴더 안내

| 폴더 | 역할 |
| --- | --- |
| [docs](docs/README.md) | 사용자·로봇 교체·그리퍼·실장비·힘 제어 매뉴얼 |
| [config](config/README.md) | robot/tool/backend/controller/task 프로필 |
| [robot](robot/README.md) | URDF·메시·MoveIt·Isaac/real adapter |
| [control](control/README.md) | 공통 계산 및 위치·force·impedance 제어 |
| [gripper](gripper/README.md) | 그리퍼 action 및 정지 지점 이벤트 |
| [path](path/README.md) | Step 1·2 및 TCP 경로 생성, 기준 경로 데이터 |
| [execution](execution/README.md) | 검증된 관절 경로 실행·기록 |
| [safety](safety/README.md) | 모델·snapshot·충돌·안경 이동 검사 |
| [launch](launch/README.md) | 공식 ROS 2 실행 진입점 |
| [scripts](scripts/README.md) | 빌드·점검 안내; 실행용 shell wrapper는 불필요 |
| [test](test/README.md) | 수치 계산·프로필·계약 검증 |

로봇별 실행 산출물은 기본 `runs/rv4fl_isaac/`, `runs/ur3e_isaac/`에 분리된다.
주요 변경점과 검증 범위는 [마이그레이션 기록](docs/MIGRATION.md)을 참고한다.
