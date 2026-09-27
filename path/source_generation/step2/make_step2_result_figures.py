#!/usr/bin/env python3
"""저장된 Step2/Step2-1 결과로 발표·검토용 PNG를 각각 다시 생성한다.

원본 9천만 점 PCD를 다시 분석하지 않고 기존 NPZ 결과를 읽기 때문에 빠르게 실행된다.
생성 파일
* step2_result_before_accuracy.png: Step2 기본 검출 결과
* step2_1_accuracy_result.png: Step2-1 정밀 보정 결과
* step2_before_after_accuracy_comparison.png: 보정 전후 진단 비교
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from .generate_global_path import write_thin_face_preview
from .smooth_global_path import write_final_path_preview, write_preview


from ..defaults import SOURCE_OUTPUT_ROOT

# =============================================================================
# USER INPUT / OUTPUT SETTINGS
# 이미 생성된 Step 2/2-1 NPZ를 읽어 PNG를 다시 만들 때 사용하는 위치와 이름이다.
# =============================================================================
INPUT_DIR = SOURCE_OUTPUT_ROOT / "step2"
OUTPUT_DIR = INPUT_DIR
INPUT_STEP2_NPZ_NAME = "step2_global_path.npz"
INPUT_STEP2_1_NPZ_NAME = "step2_1_accurate_global_path.npz"
OUTPUT_STEP2_PNG_NAME = "step2_result_before_accuracy.png"
OUTPUT_STEP2_1_PNG_NAME = "step2_1_accuracy_result.png"
OUTPUT_COMPARISON_PNG_NAME = "step2_before_after_accuracy_comparison.png"


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """NPZ 배열을 파일이 닫힌 뒤에도 사용할 수 있도록 메모리에 복사한다."""

    if not path.exists():
        raise FileNotFoundError(f"required Step2 result does not exist: {path}")
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def parse_args() -> argparse.Namespace:
    """Step 2 NPZ 입력 폴더와 PNG 출력 폴더를 명령행에서 선택한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="directory containing Step 2 and Step 2-1 NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="PNG destination",
    )
    return parser.parse_args()


def main() -> None:
    """Step2 기본/정밀 보정 결과와 비교 그림을 서로 다른 파일로 출력한다."""

    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_npz(input_dir / INPUT_STEP2_NPZ_NAME)
    corrected = load_npz(input_dir / INPUT_STEP2_1_NPZ_NAME)
    if "thin_face_points_mm" not in raw:
        raise ValueError("step2_global_path.npz has no thin_face_points_mm")
    thin_face = np.asarray(raw["thin_face_points_mm"], dtype=float)

    step2_path = output_dir / OUTPUT_STEP2_PNG_NAME
    accuracy_path = output_dir / OUTPUT_STEP2_1_PNG_NAME
    comparison_path = output_dir / OUTPUT_COMPARISON_PNG_NAME

    # Step2: 최초 검출한 얇은 면, outer/inner, 중앙선 및 XZ 경사각.
    write_thin_face_preview(step2_path, raw, thin_face)

    # Step2-1: robust 경계 밖 점을 회색으로 표시하고 최종 흰색 경로를 겹친다.
    write_final_path_preview(accuracy_path, corrected, thin_face)

    # 보정 전후의 이동량과 폭 변화를 수치 검토할 때 사용하는 4분할 진단 그림.
    write_preview(comparison_path, corrected, thin_face)

    print("Step2 result figures generated:")
    for path in (step2_path, accuracy_path, comparison_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
