# 원본 PCD에서 Step 1·2 재생성

입력은 안경 로컬 좌표(mm)의 원본 binary PCD입니다. 큰 PCD는 NumPy memory map과 chunk 단위 순회로 처리합니다. 기본 수치 파라미터와 보정 알고리즘은 기존 완성본에서 가져왔습니다.

순서는 다음과 같습니다.

1. `step1.extract_step1_thin_face`: 원본 → 얇은 면·앞/뒤 경계·검토 그림
2. `step2.generate_global_path`: 같은 원본 → outer/inner 경계·원시 초록 중심선
3. `step2.smooth_global_path`: 원시 중심선과 thin-face PCD → 보정된 CSV/PCD/NPZ
4. `step2.make_step2_result_figures`: 결과 검토용 그림

Step 2는 Step 1의 출력만 읽는 방식이 아니라 같은 원본을 별도로 분석합니다. 원본 PCD는 이 패키지에 복제하지 않으며 실행 인자로 지정합니다. Python 모듈은 상대 import를 사용하므로 `python3 -m lee_manipulator_part.path.source_generation.step1.extract_step1_thin_face --help`처럼 호출할 수 있습니다. 공식 운영 진입점은 프로젝트 launch 안내를 따릅니다.

기본 출력은 `runs/source_generation/step1/`, `runs/source_generation/step2/`입니다. 새 경로를 로봇 작업에 적용하려면 task 프로필의 `input.step2_csv`와 `input.step2_pcd`를 재생성 결과로 지정합니다. `numpy`, `scipy`, `matplotlib`, Step 1의 `opencv-python`이 필요합니다.
