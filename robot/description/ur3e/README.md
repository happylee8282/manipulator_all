# Universal Robots UR3e

원본 조립 Xacro: `old_step2/ur_ws/src/ur3e_isaac_moveit_config/urdf/ur3e_nozzle.urdf.xacro`.
로봇 형상/관절 모델: workspace의 `Universal_Robots_ROS2_Description` (`ur_description`). 해당 upstream BSD license를 `LICENSE`에 보존했습니다.

기본 launch는 Xacro를 미리 전개한 `ur3e.urdf`를 사용하므로 이전 UR workspace의 ament package 등록이 필요하지 않습니다. 로봇 메시도 여기에 복사했습니다.

- TCP는 `nozzle_tip`; `tool0`에서 0.072 m.
- STL 자체는 mm이며 장착 메시 origin이 +0.007 m입니다.
- world→base_link의 z=0.05 m는 URDF에 이미 들어 있어 별도 static TF를 중복 발행하지 않습니다.
- wrist_3_joint는 원본 macro에서 continuous joint입니다. 이 작업의 IK 탐색은 프로필의 ±2π 범위로 제한합니다.

`ur3e_nozzle.urdf.xacro`는 원본 재생성을 위한 참고 파일입니다. 재생성하려면 `ur_description`을 설치/source해야 하고, 생성 결과의 `package://ur_description/meshes/ur3e/`를 이 패키지의 `robot/description/ur3e/meshes/`로 연결해야 합니다. 실제 개체의 보정된 kinematics를 자동으로 포함한 모델은 아닙니다.

새 패키지의 UR3e는 장면 열기, Cartesian 생성과 MoveIt dry-run을 제공하며 `robot.dry_run_only: true`입니다. 기존 UR 물리 runner에는 RV-4FL의 동적 안경 정착/접촉 감시가 없어 guarded 실행을 바로 허용하지 않습니다.
