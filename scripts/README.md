# 실행 명령

공식 사용자 인터페이스는 `ros2 launch`다. 같은 동작을 감싸는 `run.sh`는 만들지 않았다.

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select lee_manipulator_part
source install/setup.bash
ros2 launch lee_manipulator_part system.launch.py robot:=rv4fl start_isaac:=true
```

빌드는 프로젝트 루트에서 실행한다. 터미널별 사용 순서는
[사용 매뉴얼](../docs/USER_MANUAL.md)에 정리되어 있다.
