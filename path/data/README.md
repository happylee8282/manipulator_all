# 기준 초록 경로

`step2_1_accurate_global_path.csv`와 `.pcd`는 기존 완성본의 Step 2-1 보정 완료 경로입니다. 두 파일은 원본과 바이트 단위로 동일하게 포함되어 있습니다. CSV에는 중심선과 양쪽 경계 정보가 있고, binary PCD에는 검증용 304개 중심선 점이 있습니다. 좌표는 안경 모델 로컬 mm입니다.

원본은 `step1_modelsolution/path_generation/step2_pcd_path/check_output/`입니다. 이 파일은 robot joint trajectory가 아니며, 선택한 로봇과 현재 안경 자세로 Cartesian/IK 단계를 다시 계산합니다.

약 1.1 GB 원본 `AI_Glass_Front_0.010mm.pcd`는 복제하지 않았습니다. 원본부터 재생성할 때는 path-generation launch의 입력 PCD 인자 또는 `LEE_MANIPULATOR_SOURCE_PCD`를 지정합니다. 출력 경로는 `runs/`를 사용하며 기준 입력 파일을 자동 덮어쓰지 않습니다.
