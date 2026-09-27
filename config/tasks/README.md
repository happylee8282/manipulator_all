# Task profiles

`glass_following.yaml`은 Step 2 경로 입력, 0.5 mm TCP clearance, 접근/후퇴, Cartesian 자세 평활화, IK 공통 설정 및 안경 감시 규칙을 정의합니다.
`settings`의 경로는 package asset root 기준입니다. 입력은 `path/data/step2_1_accurate_global_path.*`; 실행별 runtime/output 위치는 상위 config loader가 지정합니다.

로봇별 관절 제한, 초깃값, action/topic과 UR 전용 25° 기울기는 `config/robots`가 덮어씁니다.
기존 UR의 측정 편향 보정 `[-0.513, 0.277, 1.641] mm`는 새 모델 검증을 위해 자동 적용하지 않습니다. RV와 동일하게 노즐/안경 접촉 허용도 false입니다. UR의 이전 실험 수치를 새 프로필 성능으로 주장하지 않습니다.

`task.events`는 기본적으로 비어 있습니다. 실제 그리퍼 작업은 action driver, 정지/도착 조건과 함께 `gripper/` 가이드의 event schema를 사용합니다.
