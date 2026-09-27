# ROS 2 launch

`system.launch.py`, `moveit.launch.py`, `isaac.launch.py`, `task.launch.py`는 공통
`launch_support.py`를 사용한다. 동일 설정 로딩·인자·모델 선택 코드를 재사용한다.

```bash
ros2 launch lee_manipulator_part system.launch.py --show-args
ros2 launch lee_manipulator_part system.launch.py robot:=rv4fl start_isaac:=true
ros2 launch lee_manipulator_part task.launch.py robot:=rv4fl execute:=true
```

공정 실행은 기본 꺼져 있다. UR3e planning-only 프로필은 controller 명령 서버를 만들지
않으며 MoveIt 실행도 비활성화한다. 서로 다른 로봇은 기본 ROS domain 31/32로 분리된다.

`path_generation.launch.py`는 원본 PCD→Step 1·2 순차 처리,
`gripper.launch.py`는 명시적인 protocol mock이다. 자세한 단계와 인자는
[USER_MANUAL](../docs/USER_MANUAL.md)을 참고한다.
