# Robot profiles

`rv4fl.yaml`과 `ur3e.yaml`은 모델 파일, planning group/TCP, 관절 이름·제한·초기값, action/topic, 궤적 제한을 선택합니다.
`profile`은 종류/식별자, `robot`은 모델 및 backend 메타데이터, `settings`는 공통 엔진에 전달할 설정입니다.

UR은 UR3e만 구현되어 있습니다. `new_robot_template.yaml`은 필요한 필드를 안내하는 미완성 template입니다.
새 로봇은 모델/MoveIt 파일과 관절 배열 길이를 함께 등록해야 하며 이름만 바꾸는 것으로 사용할 수 없습니다.

`robot.dry_run_only`는 UR3e 동적 안경 guard 미구현 상태를 나타냅니다. 값만 false로 바꾸어 검증을 생략하지 마세요. RV의 기존 궤적도 새 run에서 Cartesian/IK를 다시 만들어야 합니다.
