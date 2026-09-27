# Tool profiles

`nozzle.yaml`은 선택한 로봇의 조립 URDF에 이미 들어 있는 노즐을 사용하는 공통 이름입니다.
`rv4fl_nozzle.yaml`과 `ur3e_nozzle.yaml`은 서로 다른 65/72 mm TCP와 STL 장착 정보를 기록합니다. 서로 호환되는 공구가 아니므로 compatible_robots로 구분합니다.

장착 변환/TCP는 현재 설명용 메타데이터이고 URDF가 실제 기하 원본입니다. YAML 값만 수정해 URDF가 자동 변경되는 것으로 해석하지 마세요.
launch는 장착 변환·TCP·mesh 정보와 URDF를 대조하며 서로 다르면 오류를 냅니다.
`gripper_template.yaml`은 실제 모델/driver/힘 제한을 채워야 하는 미지원 template입니다. 그리퍼의 실제 동작 명령 로직은 `gripper/`가 담당합니다.
