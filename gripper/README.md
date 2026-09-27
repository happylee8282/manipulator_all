# Gripper: 선택적 작업 이벤트

현재 UR3e/RV-4FL 작업 도구는 노즐입니다. 이 폴더는 실제 그리퍼 드라이버가
연결될 때 사용할 공통 action client와 정지 지점 이벤트를 제공합니다.
기존 유리 경로 실행기에 그리퍼 동작이 자동으로 삽입되는 상태는 아닙니다.

| 파일 | 입력 | 출력 / 역할 |
|---|---|---|
| `gripper_client.py` | 벌림 거리 m, 최대 effort N, timeout | ROS `control_msgs/action/GripperCommand` 결과 |
| `event_scheduler.py` | waypoint 도착 확인, 실제 관절 속도, 상태 나이 | 해당 지점 명령 완료 후 이벤트 이름 |
| `gripper_node.py` | 명시적 `allow_mock:=true` | 통신 시험용 action 서버; 장치/Isaac 그리퍼를 움직이지 않음 |

그리퍼 서버는 계속 실행하고 작업 실행기가 action goal을 보냅니다.
지점마다 프로세스를 새로 실행할 필요가 없습니다.

## 이벤트 연결 예

```python
from lee_manipulator_part.gripper import (
    GripperCommandRequest, GripperEvent, WaypointGripperScheduler,
)

scheduler = WaypointGripperScheduler([
    GripperEvent("grasp", 120, GripperCommandRequest(
        position_m=0.01, max_effort_n=20.0, timeout_s=3.0,
    )),
])

# 실행기가 stop_waypoints에서 궤적을 분할하고 정지하도록 계획해야 합니다.
# 아래 호출은 arm action의 도착 결과와 최신 실제 속도를 확인한 다음 수행합니다.
scheduler.execute_at_stop(
    120,
    reached=arm_reached,
    joint_velocities_rad_s=measured_velocities,
    state_age_s=measured_state_age,
    command=gripper_client.command,
)
```

궤적을 자른 뒤에는 속도·가속도 경계조건을 다시 계산하여 해당 지점에서 실제로
멈춰야 합니다. 기존 연속 궤적에 이벤트 호출만 추가하면 정지하지 않습니다.
위 코드는 통합 예이며 변수와 궤적 분할은 작업 실행기가 제공해야 합니다.

`GripperClient`는 executor에 등록되지 않은 전용 ROS node를 받아 호출 동안
그 node를 spin합니다. ROS callback 안에서 호출하지 말고 별도 작업 스레드에서
사용합니다. scheduler도 한 작업 스레드에서 순차 호출합니다.

이벤트는 목표 도착 확인과 최신 저속/정지 상태가 모두 있어야 실행됩니다.
완료된 이벤트는 중복 실행하지 않으며, 지나친 필수 지점은 오류로 처리합니다.
timeout·실패 결과는 작업 중단 상태로 남깁니다. 취소 요청 성공이 실제 정지를
보장하지 않으므로 장치 상태를 확인한 뒤에만 작업을 복구합니다.
goal 전송 후 timeout이면 client도 추가 명령을 거부합니다. acknowledgement까지
누락된 경우 취소 여부를 확인할 수 없습니다. 늦게 도착한 goal에 대한 취소
callback은 해당 node가 다시 spin되어야 처리되며, 장치의 별도 정지·상태 확인을
대신하지 않습니다. 복구 후 새 client와 scheduler를 만듭니다.

`allow_stall`은 기본 `false`입니다. 물체를 잡아서 목표 벌림까지 못 가는 상황을
성공으로 허용하려면 그리퍼 드라이버의 `stalled` 의미를 확인한 후 명시적으로
켜야 합니다. action이 실패/취소된 경우에는 이 옵션으로 성공 처리하지 않습니다.

## 파지 강도

`position_m`은 개구 거리, `max_effort_n`은 ROS 인터페이스에 요청하는 최대 effort입니다.
실제 파지력이 정확히 그 값이라는 뜻은 아닙니다. 드라이버가 이 필드를 지원하는지,
어떻게 환산하는지와 단위·캘리브레이션·허용 범위를 먼저 확인해야 합니다.
양손가락 각각의 변위인지 전체 개구 거리인지도 장치 어댑터에서 맞춥니다.
공압 그리퍼는 별도 밸브/압력 어댑터가 필요합니다.

## 통신 mock

Python entrypoint는 `lee_manipulator_part.gripper.gripper_node:main`입니다.
launch에서 이 entrypoint를 등록한 실행 파일에 다음 ROS parameter를 전달합니다.

```yaml
allow_mock: true
action_name: /mock_gripper/gripper_cmd
maximum_opening_m: 0.08
mock_speed_m_s: 0.04
```

이 서버는 메모리 속 개구 값만 이동시킵니다. effort 결과는 0이며, 접촉·파지·강체
연결·Isaac articulation 제어를 구현하지 않습니다. 실기 드라이버로 사용하지 않습니다.

테스트: 저장소 상위 경로에서
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q lee_manipulator_part/test/test_gripper_events.py`.
