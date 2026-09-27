# Step 1: 얇은 면과 경계

`extract_step1_thin_face.py`가 전체 굴곡을 포함하는 생산용 추출기입니다. `extract_step1_pcd_path.py`는 공통 binary PCD reader와 chunk 순회, 초기 중앙 구간 추출을 제공합니다.

입력 옵션은 `--pcd /absolute/source.pcd`, `--output-dir /absolute/output`입니다. 로컬 X/Y/Z 단위는 mm이며 world Z 높이와 혼동하면 안 됩니다. 출력은 `step1_complete_surface.npz`, 얇은 면 PCD, 앞/뒤 경계 CSV 및 검토 그림입니다.

ROI·raster·경계 corridor·spacing 옵션은 `--help`로 확인합니다. 원본 기본값은 `LEE_MANIPULATOR_SOURCE_PCD`로 지정할 수 있습니다. 같은 기본 파라미터를 사용하면 기존 구현의 계산 순서를 유지합니다.
