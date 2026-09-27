# Step 2: 초록 경로 생성과 보정

`generate_global_path.py`는 원본에서 바깥/안쪽 경계를 검출하여 그 중간 경로를 생성합니다. 입력은 `--input-pcd`, 출력은 `--output-dir`입니다. 결과는 `step2_global_path.csv`와 `step2_thin_bonding_face.pcd` 등입니다.

`smooth_global_path.py`는 `--input step2_global_path.csv --thin-face-pcd step2_thin_bonding_face.pcd --output-dir ...`를 받아 짧은 진동을 줄이고 측정 면 중심으로 보정합니다. 최종 로봇 경로 생성 입력은 `step2_1_accurate_global_path.csv/.pcd`입니다. `make_step2_result_figures.py`는 `--input-dir`과 `--output-dir`로 검토 그림을 생성합니다.

이 단계의 좌표는 mm입니다. 노즐의 작업 간격은 `path/cartesian_generator.py` 단계에서 적용하며 원본 중심선을 높이 파라미터 때문에 변경하지 않습니다.
