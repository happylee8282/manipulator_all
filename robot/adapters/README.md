# Backend adapters

`isaac_adapter.build_command(config, root, python_executable, gui)`는 실행할 argv를 만들고 launch가 별도 Isaac Python 프로세스로 시작합니다. ROS Python에 Isaac을 import하지 않습니다.

- `rv4fl_scene.py`: 이전 RV-4FL runner 이식본. 첫 Play 전에 초기 관절 자세/drive를 맞추고 안경을 dynamic rigid body로 정착시킨 뒤 snapshot/status/diagnostics를 기록합니다.
- `ur3e_scene.py`: 기존 UR3e 실험 runner 이식본. `ur3e_graph_repair.py`로 articulation target/system timestamp를 맞추고 장면을 실행합니다. 동적 안경 정착 감시 기능은 없습니다.
- `real_robot_adapter.py`: 실물 드라이버 연결 전에는 명확한 `NotImplementedError`를 반환합니다.

Isaac interpreter 기본값은 `config/backends/isaac.yaml`에 있습니다. 설치 위치가 다른 PC에서는 launch의 Isaac Python 경로를 변경합니다.
stage, runtime 파일, 초기 자세, RV command/state topic은 launch에서 최종 설정을 전달합니다. RV joint prim/link6 구조와 물리 drive 기본값은 아직 `rv4fl_scene.py`의 로봇 전용 부분입니다.

USD 의존성:

- RV: workspace `step1_modelsolution/rv-4fl_rev_B_step/rv4fl_ur_e_0813_nozzle.usda` 및 해당 USD가 참조하는 리소스.
- UR: workspace `old_step2/glass_robotarm/ur_e_0813_nozzle.usda` 및 해당 USD가 참조하는 리소스.

원본 USD를 저장하거나 수정하지 않고 실행 세션에서만 drive/graph/physics를 조정합니다. UR 프로필의 `dry_run_only`를 바꾸기만 해서는 누락된 안경 감시가 구현되지 않습니다.

일반 install 위치에서 실행하거나 폴더를 이동한 경우 launch의 `asset_workspace` 또는 `LEE_MANIPULATOR_ASSET_WORKSPACE` 환경변수로 `step1_modelsolution/`, `old_step2/`가 들어 있는 원본 workspace 위치를 지정합니다.
