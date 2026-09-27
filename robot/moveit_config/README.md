# MoveIt configuration

`rv4fl/`, `ur3e/`의 기존 설정을 별도 복사했습니다. 상위 launch가 선택한 로봇 프로필의 파일 경로를 읽어 `move_group`과 `robot_state_publisher`를 구성합니다.

- `*.srdf`: planning group, 초기 자세, 자기충돌 예외.
- `kinematics.yaml`: IK plugin.
- `joint_limits.yaml`: MoveIt 관절 제한 override.
- `ompl_planning.yaml`: OMPL planning 설정.
- `moveit_controllers.yaml`: FollowJointTrajectory action 연결.
- `moveit.rviz`: 기존 화면 설정.

제어기의 `action_name` 및 관절 순서가 각 파일과 일치해야 합니다. `config/robots`의 궤적 시간 제한은 별도 더 낮은 작업용 제한이며 제조사 최대값과 구분합니다.
