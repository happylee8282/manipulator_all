#!/usr/bin/env python3
"""Step 1의 전체 얇은 본딩 면과 끝단 포함 경계를 원본 PCD에서 추출한다.

좌표계
------
입력 PCD는 안경 모델 로컬 좌표이며 단위는 mm이다.

* X: 정면에서 좌우
* Y: 정면에서 위아래
* Z: 정면 기준 앞뒤 깊이

주의할 점은 이 파일의 local Z가 로봇이 위에서 접근할 때 사용하는 Isaac world Z와
같은 축이 아니라는 것이다. 이 단계는 아직 로봇 높이 offset을 적용하지 않고, PCD
자체에서 얇은 면과 그 앞/뒤 깊이 경계만 분리한다.

전체 처리 순서
--------------
1. 92.7M 포인트 binary PCD를 ``chunk_points`` 단위로 나누어 읽는다.
2. X/Y 사각 ROI를 0.02 mm 격자의 정면 XY occupancy map으로 투영한다.
3. OpenCV 외곽 contour 중 가장 큰 contour를 선택한다.
4. 좌우 최대 X 끝점 사이의 두 contour 방향 중 평균 Y가 높은 상단 arc를 선택한다.
   이 방식은 Y=f(X)를 가정하지 않으므로 양끝의 거의 수직인 굴곡도 보존한다.
5. 상단 arc 주변 ``face_half_width_mm`` corridor에 들어가는 원본 3D PCD 점만 다시
   수집하여 실제 얇은 면을 crop한다.
6. 각 arc 위치 주변의 local 최소/최대 Z를 앞쪽/뒤쪽 경계 후보로 만든다.
7. 후보를 KD-tree로 실제 crop PCD 점에 snapping한다.
8. 경계를 호길이 기준으로 Savitzky-Golay smoothing하고 0.5 mm 간격으로
   재샘플링한 뒤, 다시 측정 PCD 경계점으로 snapping한다.
9. PCD, CSV, NPZ, JSON metadata 및 검증용 PNG를 출력한다.

함수 호출 구조
--------------
``main``
  -> ``parse_args``
  -> ``extract_complete_surface``
       -> ``validate_config`` / ``make_raster_geometry``
       -> ``rasterise_xy_projection``
       -> ``find_upper_outer_arc``
       -> ``build_face_corridor_mask`` / ``extract_face_points``
       -> ``neighbourhood_extrema_on_arc``
       -> ``refine_boundary_targets``
       -> ``smooth_and_resample_edge``
       -> 각종 ``write_*`` 함수

이전 ``extract_step1_pcd_path.py``는 X마다 점 하나를 고르는 빠른 중앙 구간 확인용이다.
끝단 굴곡과 얇은 면 전체가 필요한 최종 결과에는 이 파일을 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from .extract_step1_pcd_path import (
    DEFAULT_PCD,
    DEFAULT_OUTPUT_DIR as STEP1_OUTPUT_DIR,
    PcdHeader,
    cumulative_distance,
    iter_chunks,
    open_binary_pcd,
    sample_preview_cloud,
)


# =============================================================================
# USER INPUT / OUTPUT SETTINGS
# 아래 값만 바꾸면 전체 얇은 면 추출의 입력 PCD와 결과 위치/이름이 바뀐다.
# 명령행의 --pcd, --output-dir 옵션을 주면 실행할 때만 이 값을 덮어쓴다.
# =============================================================================
INPUT_PCD = DEFAULT_PCD
OUTPUT_DIR = STEP1_OUTPUT_DIR

OUTPUT_THIN_FACE_PCD_NAME = "step1_thin_face.pcd"
OUTPUT_FRONT_EDGE_CSV_NAME = "step1_front_edge_path.csv"
OUTPUT_BACK_EDGE_CSV_NAME = "step1_back_edge_path.csv"
OUTPUT_FRONT_RAW_CSV_NAME = "step1_front_edge_raw.csv"
OUTPUT_BACK_RAW_CSV_NAME = "step1_back_edge_raw.csv"
OUTPUT_ARCHIVE_NPZ_NAME = "step1_complete_surface.npz"
OUTPUT_METADATA_JSON_NAME = "step1_complete_surface_metadata.json"
OUTPUT_PREVIEW_PNG_NAME = "step1_complete_surface_preview.png"

DEFAULT_OUTPUT_DIR = OUTPUT_DIR


@dataclass(frozen=True)
class SurfaceConfig:
    """Step 1 crop과 경계 생성에 필요한 설정값.

    모든 길이 파라미터는 모델 로컬 mm 단위이다. ``x/y_min/max``는 1차 사각 ROI,
    ``raster_mm``는 XY 영상 해상도, ``face_half_width_mm``는 contour 주변 실제 면을
    다시 선택할 때의 반폭이다. ``edge_neighbourhood_mm``는 동일 XY 부근에서 앞/뒤 Z
    경계를 검색할 반경이며, ``output_spacing_mm``는 최종 경로점 간격이다.
    """

    x_min_mm: float = -74.40
    x_max_mm: float = 74.40
    y_min_mm: float = 3.0
    y_max_mm: float = 21.0
    raster_mm: float = 0.02
    face_half_width_mm: float = 0.06
    face_voxel_mm: float = 0.02
    edge_neighbourhood_mm: float = 0.06
    smooth_window_mm: float = 0.40
    output_spacing_mm: float = 0.50
    chunk_points: int = 5_000_000
    preview_max_points: int = 100_000


@dataclass(frozen=True)
class RasterGeometry:
    """모델 로컬 XY 좌표와 raster 행/열 사이의 변환 정보."""

    x_min_mm: float
    y_min_mm: float
    resolution_mm: float
    rows: int
    cols: int

    def xy_to_pixels(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """연속 좌표 X/Y(mm)를 0부터 시작하는 raster row/column으로 변환한다.

        ``floor``를 사용하므로 같은 격자 안의 여러 PCD 점은 동일 픽셀로 투영된다.
        반환 순서는 NumPy 영상 인덱싱에 맞춘 ``(rows, cols)``이다.
        """
        cols = np.floor((x - self.x_min_mm) / self.resolution_mm).astype(np.int32)
        rows = np.floor((y - self.y_min_mm) / self.resolution_mm).astype(np.int32)
        return rows, cols

    def pixels_to_xy(self, pixels_xy: np.ndarray) -> np.ndarray:
        """OpenCV contour의 ``(column, row)``를 각 픽셀 중심의 X/Y(mm)로 복원한다."""
        pixels_xy = np.asarray(pixels_xy)
        return np.column_stack(
            (
                self.x_min_mm + (pixels_xy[:, 0] + 0.5) * self.resolution_mm,
                self.y_min_mm + (pixels_xy[:, 1] + 0.5) * self.resolution_mm,
            )
        )


def validate_config(config: SurfaceConfig) -> None:
    """잘못된 범위나 0 이하의 해상도로 실행되는 것을 시작 전에 차단한다.

    이 함수는 데이터를 바꾸지 않는다. 범위의 대소 관계, 길이 파라미터의 양수 여부,
    chunk/preview 점 개수만 검증하고 문제가 있으면 ``ValueError``를 발생시킨다.
    """
    if not config.x_min_mm < config.x_max_mm:
        raise ValueError("x_min_mm must be smaller than x_max_mm")
    if not config.y_min_mm < config.y_max_mm:
        raise ValueError("y_min_mm must be smaller than y_max_mm")
    for name in (
        "raster_mm",
        "face_half_width_mm",
        "face_voxel_mm",
        "edge_neighbourhood_mm",
        "smooth_window_mm",
        "output_spacing_mm",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.chunk_points <= 0 or config.preview_max_points <= 0:
        raise ValueError("chunk_points and preview_max_points must be positive")


def make_raster_geometry(config: SurfaceConfig) -> RasterGeometry:
    """설정된 ROI 크기와 해상도에서 필요한 raster 행/열 개수를 계산한다.

    기본 설정은 X 폭 148.8 mm / 0.02 mm = 7,440열,
    Y 폭 18 mm / 0.02 mm = 900행이다.
    """
    cols = int(math.ceil((config.x_max_mm - config.x_min_mm) / config.raster_mm))
    rows = int(math.ceil((config.y_max_mm - config.y_min_mm) / config.raster_mm))
    return RasterGeometry(
        x_min_mm=config.x_min_mm,
        y_min_mm=config.y_min_mm,
        resolution_mm=config.raster_mm,
        rows=rows,
        cols=cols,
    )


def print_progress(label: str, processed: int, total: int, started: float) -> None:
    """대용량 PCD chunk 처리율과 초당 처리 포인트 수를 터미널에 출력한다."""
    elapsed = max(time.monotonic() - started, 1.0e-6)
    print(
        f"  {label}: {100.0 * processed / total:5.1f}% "
        f"({processed / elapsed / 1.0e6:5.1f} M points/s)",
        flush=True,
    )


def rasterise_xy_projection(
    records: np.ndarray,
    geometry: RasterGeometry,
    config: SurfaceConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """1차 XY ROI를 occupancy 및 픽셀별 Z 최솟값/최댓값 map으로 투영한다.

    처리 과정
    1. 전체 PCD를 ``chunk_points``씩 읽는다.
    2. 유한한 점 중 X/Y ROI에 들어오는 점만 고른다. Z는 자르지 않는다.
    3. 각 점을 raster row/column으로 변환한다.
    4. 해당 픽셀의 occupancy를 1로 표시하고 local 최소 Z와 최대 Z를 누적한다.

    Z를 ROI 조건에서 제외하는 이유는 같은 XY에서 얇은 면의 앞쪽과 뒤쪽 깊이 경계를
    모두 남기기 위해서다.

    Returns
    -------
    occupied:
        점이 하나 이상 투영된 픽셀의 bool 성격 2D map.
    min_z, max_z:
        각 XY 픽셀에 들어온 원본 점의 최소/최대 local Z.
    selected_count:
        사각 ROI에 들어와 raster에 누적된 원본 점 수.
    """

    pixel_count = geometry.rows * geometry.cols
    occupied = np.zeros(pixel_count, dtype=np.uint8)
    min_z = np.full(pixel_count, np.inf, dtype=np.float32)
    max_z = np.full(pixel_count, -np.inf, dtype=np.float32)
    selected_count = 0
    started = time.monotonic()

    for number, (start, chunk) in enumerate(
        iter_chunks(records, config.chunk_points), start=1
    ):
        x = np.asarray(chunk["x"])
        y = np.asarray(chunk["y"])
        z = np.asarray(chunk["z"])
        mask = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & (x >= config.x_min_mm)
            & (x <= config.x_max_mm)
            & (y >= config.y_min_mm)
            & (y <= config.y_max_mm)
        )
        if np.any(mask):
            rows, cols = geometry.xy_to_pixels(x[mask], y[mask])
            valid = (
                (rows >= 0)
                & (rows < geometry.rows)
                & (cols >= 0)
                & (cols < geometry.cols)
            )
            rows = rows[valid]
            cols = cols[valid]
            values_z = z[mask][valid]
            flat = rows.astype(np.int64) * geometry.cols + cols
            occupied[flat] = 1
            np.minimum.at(min_z, flat, values_z)
            np.maximum.at(max_z, flat, values_z)
            selected_count += len(flat)
        if number % 4 == 0 or start + len(chunk) == len(records):
            print_progress("XY raster", start + len(chunk), len(records), started)

    return (
        occupied.reshape(geometry.rows, geometry.cols),
        min_z.reshape(geometry.rows, geometry.cols),
        max_z.reshape(geometry.rows, geometry.cols),
        selected_count,
    )


def contour_arc_indices(size: int, start: int, end: int, step: int) -> np.ndarray:
    """닫힌 contour 배열에서 start부터 end까지 한 방향의 인덱스를 만든다.

    OpenCV contour는 마지막 점 다음이 첫 점으로 이어지는 순환 배열이다. ``step=+1``과
    ``step=-1``을 각각 호출하면 좌측 끝에서 우측 끝으로 가는 두 후보 arc를 얻는다.
    """
    values = [start]
    current = start
    while current != end:
        current = (current + step) % size
        values.append(current)
        if len(values) > size + 1:
            raise RuntimeError("failed to traverse closed contour")
    return np.asarray(values, dtype=np.int64)


def find_upper_outer_arc(occupied: np.ndarray) -> Tuple[np.ndarray, Mapping[str, int]]:
    """가장 큰 XY 외곽 contour에서 양끝 굴곡을 포함한 상단 arc를 찾는다.

    1. 3x3 morphological closing으로 1~2픽셀의 작은 PCD 누락을 연결한다.
    2. ``RETR_EXTERNAL``로 내부 렌즈 구멍이 아닌 외부 contour만 찾는다.
    3. 면적이 가장 큰 contour를 안경 프레임 실루엣으로 선택한다.
    4. 최소/최대 X 열에서 가장 높은 Y점을 좌우 상단 끝점으로 선택한다.
    5. 끝점 사이의 정방향/역방향 arc 중 평균 Y가 높은 쪽을 상단 경로로 선택한다.

    단순히 X별 최대 Y를 고르지 않고 순서가 있는 contour를 따라가기 때문에 X가 거의
    고정된 양끝 수직 굴곡도 경로에 포함된다.

    Returns
    -------
    arc:
        OpenCV 형식 ``(column, row)``의 순서가 있는 상단 contour 픽셀.
    stats:
        전체 contour/선택 arc 길이와 좌우 열 인덱스.
    """

    kernel = np.ones((3, 3), dtype=np.uint8)
    clean = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(
        clean * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("no external XY contour found in the configured ROI")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
    if len(contour) < 10:
        raise ValueError("largest XY contour is unexpectedly short")

    min_col = int(np.min(contour[:, 0]))
    max_col = int(np.max(contour[:, 0]))
    left_candidates = np.flatnonzero(contour[:, 0] <= min_col + 1)
    right_candidates = np.flatnonzero(contour[:, 0] >= max_col - 1)
    # At the extreme X there can be a vertical side.  Its highest occupied
    # point is the end of the requested top thin face and includes the curved
    # downturn leading into that extreme.
    left_index = int(left_candidates[np.argmax(contour[left_candidates, 1])])
    right_index = int(right_candidates[np.argmax(contour[right_candidates, 1])])

    forward_indices = contour_arc_indices(len(contour), left_index, right_index, +1)
    backward_indices = contour_arc_indices(len(contour), left_index, right_index, -1)
    forward = contour[forward_indices]
    backward = contour[backward_indices]
    # Raster row increases with model-local Y in this program.  The arc with
    # the larger mean row is the upper route; the other goes around the lower
    # frame/cropped bottom boundary.
    arc = forward if np.mean(forward[:, 1]) >= np.mean(backward[:, 1]) else backward
    if arc[0, 0] > arc[-1, 0]:
        arc = arc[::-1]

    consecutive_change = np.r_[True, np.any(arc[1:] != arc[:-1], axis=1)]
    arc = arc[consecutive_change]
    return arc, {
        "external_contour_pixels": int(len(contour)),
        "upper_arc_pixels": int(len(arc)),
        "left_col": min_col,
        "right_col": max_col,
    }


def build_face_corridor_mask(
    arc_pixels: np.ndarray,
    geometry: RasterGeometry,
    half_width_mm: float,
) -> np.ndarray:
    """상단 contour 주위에 실제 얇은 면을 crop할 좁은 XY corridor mask를 만든다.

    ``half_width_mm``를 raster 픽셀 반경으로 바꾼 뒤 OpenCV polyline을 그린다.
    기본값 0.06 mm와 0.02 mm raster에서는 반경 3픽셀, 두께 7픽셀이다. 반환 mask가
    2차 crop 조건이며, 이후 원본 PCD를 다시 읽어 투영 픽셀이 이 mask에 든 점만 남긴다.
    """
    mask = np.zeros((geometry.rows, geometry.cols), dtype=np.uint8)
    radius = max(1, int(math.ceil(half_width_mm / geometry.resolution_mm)))
    cv2.polylines(
        mask,
        [arc_pixels.reshape(-1, 1, 2)],
        isClosed=False,
        color=1,
        thickness=2 * radius + 1,
        lineType=cv2.LINE_8,
    )
    return mask.astype(bool)


def voxel_unique_indices(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    """3D 공간을 voxel로 양자화하고 voxel마다 대표점 하나의 인덱스를 반환한다.

    ``floor(points / voxel_mm)``로 정수 voxel 주소를 만들고 중복 주소를 제거한다.
    원본 형상을 평균으로 이동시키지 않기 위해 centroid 대신 실제 입력점 하나를 남긴다.
    """
    voxels = np.floor(np.asarray(points) / voxel_mm).astype(np.int32)
    packed = np.ascontiguousarray(voxels).view(
        np.dtype((np.void, voxels.dtype.itemsize * voxels.shape[1]))
    )
    return np.unique(packed.ravel(), return_index=True)[1]


def extract_face_points(
    records: np.ndarray,
    corridor: np.ndarray,
    geometry: RasterGeometry,
    config: SurfaceConfig,
) -> Tuple[np.ndarray, int]:
    """원본 PCD에서 상단 contour corridor에 투영되는 실제 3D 점만 crop한다.

    전체 PCD를 다시 chunk 단위로 순회하면서 각 점의 X/Y 픽셀을 구하고,
    ``corridor[row, col]``가 참인 점의 X/Y/Z를 모두 보존한다. Z 범위를 자르지 않으므로
    동일한 XY corridor 안에 존재하는 얇은 면의 앞쪽/뒤쪽 깊이가 함께 남는다.

    메모리와 중복을 줄이기 위해 각 chunk에서 한 번, 모든 chunk를 합친 뒤 다시 한 번
    ``face_voxel_mm`` 3D voxel 중복 제거를 수행한다.

    Returns
    -------
    face_points:
        voxel 중복 제거 후에도 원본 좌표를 유지하는 실제 얇은 면 PCD 점.
    raw_count:
        voxel 처리 전에 corridor가 선택한 원본 점 수.
    """

    pieces: List[np.ndarray] = []
    raw_count = 0
    started = time.monotonic()
    for number, (start, chunk) in enumerate(
        iter_chunks(records, config.chunk_points), start=1
    ):
        x = np.asarray(chunk["x"])
        y = np.asarray(chunk["y"])
        z = np.asarray(chunk["z"])
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        indices = np.flatnonzero(finite)
        if len(indices):
            rows, cols = geometry.xy_to_pixels(x[indices], y[indices])
            inside = (
                (rows >= 0)
                & (rows < geometry.rows)
                & (cols >= 0)
                & (cols < geometry.cols)
            )
            indices = indices[inside]
            rows = rows[inside]
            cols = cols[inside]
            on_face = corridor[rows, cols]
            indices = indices[on_face]
            if len(indices):
                points = np.column_stack((x[indices], y[indices], z[indices])).astype(
                    np.float32, copy=False
                )
                raw_count += len(points)
                local_unique = voxel_unique_indices(points, config.face_voxel_mm)
                pieces.append(points[local_unique])
        if number % 4 == 0 or start + len(chunk) == len(records):
            print_progress("thin face", start + len(chunk), len(records), started)

    if not pieces:
        raise ValueError("thin-face corridor did not select any source PCD points")
    combined = np.concatenate(pieces, axis=0)
    global_unique = voxel_unique_indices(combined, config.face_voxel_mm)
    return np.asarray(combined[global_unique], dtype=np.float32), raw_count


def neighbourhood_extrema_on_arc(
    min_z: np.ndarray,
    max_z: np.ndarray,
    arc_pixels: np.ndarray,
    geometry: RasterGeometry,
    radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """각 상단 arc 픽셀 주변에서 얇은 면의 앞/뒤 local Z 경계를 구한다.

    ``radius_mm``를 픽셀 반경으로 변환한 후 minimum/maximum filter를 적용한다.
    기본 반경 0.06 mm에서는 arc 픽셀마다 약 7x7 이웃을 본다.

    * front: 이웃에 존재하는 최소 local Z, 즉 XZ 그림의 빨간 경계
    * back: 이웃에 존재하는 최대 local Z, 즉 XZ 그림의 파란 경계

    contour의 98% 이상에서 두 경계를 얻지 못하면 잘못된 ROI/crop으로 보고 실패시킨다.
    일부 누락만 있을 경우에는 contour 순서상 앞뒤 값을 선형 보간한다.
    """
    radius = max(1, int(math.ceil(radius_mm / geometry.resolution_mm)))
    size = 2 * radius + 1
    local_min = minimum_filter(min_z, size=size, mode="constant", cval=np.inf)
    local_max = maximum_filter(max_z, size=size, mode="constant", cval=-np.inf)
    rows = arc_pixels[:, 1]
    cols = arc_pixels[:, 0]
    front = local_min[rows, cols]
    back = local_max[rows, cols]
    valid = np.isfinite(front) & np.isfinite(back)
    if np.mean(valid) < 0.98:
        raise ValueError(
            f"only {np.mean(valid):.1%} of contour pixels have local Z support"
        )
    if not np.all(valid):
        positions = np.arange(len(front))
        front = np.interp(positions, positions[valid], front[valid])
        back = np.interp(positions, positions[valid], back[valid])
    return front, back


def remove_consecutive_duplicates(points: np.ndarray, tolerance_mm: float) -> np.ndarray:
    """연속한 3D 점 사이 거리가 허용값 이하이면 뒤의 중복점을 제거한다.

    KD-tree snapping이나 raster contour 때문에 같은 PCD 점이 여러 번 선택될 수 있다.
    순서는 유지하면서 연속 중복만 제거하여 호길이와 미분 계산의 0 나눗셈을 방지한다.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance_mm]
    return points[keep]


def refine_boundary_targets(
    targets: np.ndarray,
    face_points: np.ndarray,
    raster_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Raster에서 계산한 경계 후보를 실제 crop PCD의 최근접 3D 점으로 snapping한다.

    raster 픽셀 중심과 min/max Z로 만든 target은 수치적으로 생성한 좌표이다. 얇은 면
    ``face_points``의 KD-tree에서 최근접 점을 찾음으로써 경계를 실제 측정 PCD 좌표로
    되돌린다. 반환 거리 배열은 raster 후보와 PCD 사이의 snapping 오차 검증에 사용한다.
    """

    tree = cKDTree(np.asarray(face_points, dtype=np.float64))
    distances, indices = tree.query(targets, workers=-1)
    refined = np.asarray(face_points[indices], dtype=np.float64)
    refined = remove_consecutive_duplicates(refined, tolerance_mm=0.25 * raster_mm)
    return refined, np.asarray(distances, dtype=np.float64)


def smooth_and_resample_edge(
    raw_points: np.ndarray,
    smooth_window_mm: float,
    raster_mm: float,
    spacing_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """순서가 있는 경계를 부드럽게 하고 일정 호길이 간격으로 재샘플링한다.

    처리 순서
    1. 연속 중복점을 제거한다.
    2. raw 3D 거리가 아니라 XY contour 호길이로 parameter를 만든다. 뒤쪽 경계의 작은
       Z sampling jitter가 가짜 경로 길이를 만드는 것을 막기 위해서다.
    3. raster 해상도 간격의 균일한 점으로 보간한다.
    4. X/Y/Z 각각에 3차 Savitzky-Golay smoothing을 적용한다.
    5. smoothing 때문에 실제 양끝점이 움직이지 않도록 첫/마지막 점을 원본에 고정한다.
    6. 부드러운 곡선을 ``spacing_mm``(기본 0.5 mm) 호길이 간격으로 재샘플링한다.
    7. 이 곡선은 guide로만 쓰고, 최종 waypoint를 다시 raw PCD 경계의 최근접 점으로
       snapping한다. 따라서 결과가 임의 spline 좌표로 PCD 밖에 뜨지 않는다.
    8. 최종 측정점의 누적 거리와 단위 tangent를 계산한다.

    Returns
    -------
    measured:
        PCD 경계에 snapping된 최종 waypoint.
    measured_distance:
        첫 점부터의 3D 누적 호길이(mm).
    tangents:
        각 waypoint의 정규화된 진행 방향.
    ideal:
        마지막 PCD snapping 직전의 부드러운 이상적 guide 곡선.
    """

    raw_points = remove_consecutive_duplicates(raw_points, tolerance_mm=0.25 * raster_mm)
    # Parameterise by ordered XY contour distance first.  Using raw 3D length
    # here would turn small Z sampling jitter on the rear edge into artificial
    # extra path length before the smoothing step.
    xy_steps = np.linalg.norm(np.diff(raw_points[:, :2], axis=0), axis=1)
    xy_keep = np.r_[True, xy_steps > 0.25 * raster_mm]
    raw_points = raw_points[xy_keep]
    raw_parameter = cumulative_distance(raw_points[:, :2])
    uniform_s = np.arange(0.0, raw_parameter[-1], raster_mm)
    if not np.isclose(uniform_s[-1], raw_parameter[-1]):
        uniform_s = np.r_[uniform_s, raw_parameter[-1]]
    uniform = np.column_stack(
        [np.interp(uniform_s, raw_parameter, raw_points[:, axis]) for axis in range(3)]
    )

    window = max(5, int(round(smooth_window_mm / raster_mm)))
    if window % 2 == 0:
        window += 1
    maximum = len(uniform) if len(uniform) % 2 == 1 else len(uniform) - 1
    window = min(window, maximum)
    smooth = uniform.copy()
    if window >= 5:
        for axis in range(3):
            smooth[:, axis] = savgol_filter(
                uniform[:, axis], window, polyorder=3, mode="interp"
            )
    smooth[[0, -1]] = uniform[[0, -1]]

    smooth_distance = cumulative_distance(smooth)
    targets_s = np.arange(0.0, smooth_distance[-1], spacing_mm)
    if not np.isclose(targets_s[-1], smooth_distance[-1]):
        targets_s = np.r_[targets_s, smooth_distance[-1]]
    ideal = np.column_stack(
        [np.interp(targets_s, smooth_distance, smooth[:, axis]) for axis in range(3)]
    )

    # The ideal spline is only a guide.  The delivered path is snapped back to
    # the measured PCD-derived boundary so every final waypoint is on the map.
    raw_tree = cKDTree(raw_points)
    _, nearest = raw_tree.query(ideal, workers=-1)
    measured = remove_consecutive_duplicates(
        raw_points[nearest], tolerance_mm=0.25 * raster_mm
    )
    measured_distance = cumulative_distance(measured)
    derivatives = np.gradient(measured, measured_distance, axis=0, edge_order=1)
    tangent_norm = np.linalg.norm(derivatives, axis=1, keepdims=True)
    tangents = derivatives / np.maximum(tangent_norm, 1.0e-12)
    return measured, measured_distance, tangents, ideal


def write_binary_pcd(path: Path, points: np.ndarray) -> None:
    """XYZ float32 점 배열을 PCD v0.7 binary 형식으로 저장한다.

    색상이나 normal 필드는 추가하지 않으며, ``step1_thin_face.pcd``가 원본 PCD와 같은
    모델 로컬 mm 좌표를 유지하도록 X/Y/Z만 기록한다.
    """
    points = np.asarray(points, dtype="<f4")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        points.tofile(stream)


def write_edge_csv(
    path: Path,
    points: np.ndarray,
    distances: np.ndarray,
    tangents: np.ndarray,
) -> None:
    """최종 경로의 좌표, 누적 거리 및 단위 tangent를 CSV로 저장한다."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "index",
                "path_distance_mm",
                "x_mm",
                "y_mm",
                "z_mm",
                "tangent_x",
                "tangent_y",
                "tangent_z",
            )
        )
        for index, (point, distance, tangent) in enumerate(
            zip(points, distances, tangents)
        ):
            writer.writerow(
                (
                    index,
                    f"{distance:.9f}",
                    *[f"{value:.9f}" for value in point],
                    *[f"{value:.9f}" for value in tangent],
                )
            )


def write_raw_edge_csv(path: Path, points: np.ndarray) -> None:
    """smoothing/0.5 mm 재샘플링 전의 PCD-snapped raw 경계를 CSV로 저장한다.

    최종 결과와 원본 측정 경계를 비교하거나 필터 파라미터를 다시 조정할 때 사용한다.
    """
    distance = cumulative_distance(points)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "path_distance_mm", "x_mm", "y_mm", "z_mm"))
        for index, (point, value_s) in enumerate(zip(points, distance)):
            writer.writerow((index, f"{value_s:.9f}", *[f"{v:.9f}" for v in point]))


def write_preview(
    path: Path,
    cloud_sample: np.ndarray,
    face_points: np.ndarray,
    front_raw: np.ndarray,
    back_raw: np.ndarray,
    final_path: np.ndarray,
) -> None:
    """Step 1 crop과 경계를 사람이 검토할 2x2 PNG를 생성한다.

    * 좌상: 전체 PCD 정면 XY, 청록 얇은 면, 빨간 최종 경로와 시작/끝
    * 우상: XZ에서 앞쪽 빨간 경계와 뒤쪽 파란 경계, 청록 면
    * 좌하/우하: 양끝 수직 굴곡 확대

    대용량 점을 모두 그리지 않고 입력 preview sample과 최대 약 10만 개의 면 점만
    표시한다. 이 그림은 계산 입력이 아니라 결과 검사용이다.
    """
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step1_pcd_path_matplotlib")
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unable to import Axes3D.*", category=UserWarning
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    face_stride = max(1, int(math.ceil(len(face_points) / 100_000)))
    face_sample = face_points[::face_stride]
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    front, depth, left_zoom, right_zoom = axes.ravel()

    front.scatter(cloud_sample[:, 0], cloud_sample[:, 1], s=0.2, c="0.78", alpha=0.25)
    front.scatter(face_sample[:, 0], face_sample[:, 1], s=0.4, c="cyan", alpha=0.25)
    front.plot(final_path[:, 0], final_path[:, 1], color="red", linewidth=2.0)
    front.scatter(
        final_path[[0, -1], 0], final_path[[0, -1], 1], c=["lime", "magenta"], s=45
    )
    front.set(title="End-to-end upper outer contour (XY)", xlabel="X [mm]", ylabel="Y [mm]")
    front.set_aspect("equal", adjustable="box")
    front.grid(alpha=0.25)

    depth.scatter(cloud_sample[:, 0], cloud_sample[:, 2], s=0.2, c="0.8", alpha=0.2)
    depth.scatter(face_sample[:, 0], face_sample[:, 2], s=0.4, c="cyan", alpha=0.25)
    depth.plot(front_raw[:, 0], front_raw[:, 2], color="red", linewidth=1.5, label="front / lower Z")
    depth.plot(back_raw[:, 0], back_raw[:, 2], color="blue", linewidth=1.5, label="back / upper Z")
    depth.set(title="Two boundaries of the extracted thin face (XZ)", xlabel="X [mm]", ylabel="Z [mm]")
    depth.grid(alpha=0.25)
    depth.legend()

    for axis, side, x_limits, title in (
        (left_zoom, final_path[:, 0] < -64.0, (-75.5, -63.0), "Left downturned end"),
        (right_zoom, final_path[:, 0] > 64.0, (63.0, 75.5), "Right downturned end"),
    ):
        cloud_side = (cloud_sample[:, 0] >= x_limits[0]) & (cloud_sample[:, 0] <= x_limits[1])
        face_side = (face_sample[:, 0] >= x_limits[0]) & (face_sample[:, 0] <= x_limits[1])
        axis.scatter(
            cloud_sample[cloud_side, 0],
            cloud_sample[cloud_side, 1],
            s=1.0,
            c="0.75",
            alpha=0.35,
        )
        axis.scatter(
            face_sample[face_side, 0],
            face_sample[face_side, 1],
            s=1.2,
            c="cyan",
            alpha=0.3,
        )
        axis.plot(final_path[side, 0], final_path[side, 1], color="red", linewidth=2.2)
        axis.set(title=title, xlabel="X [mm]", ylabel="Y [mm]")
        axis.set_xlim(*x_limits)
        axis.set_ylim(12.5, 19.0)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")

    figure.suptitle("Step 1 complete thin face and PCD-snapped boundary path")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_metadata(
    pcd_path: Path,
    header: PcdHeader,
    config: SurfaceConfig,
    geometry: RasterGeometry,
    raster_source_count: int,
    contour_stats: Mapping[str, int],
    face_raw_count: int,
    face_points: np.ndarray,
    front_snap_distance: np.ndarray,
    back_snap_distance: np.ndarray,
    final_path: np.ndarray,
    final_distance: np.ndarray,
) -> Mapping[str, object]:
    """재현성과 자동 검증에 필요한 설정/통계를 JSON 직렬화 가능한 dict로 만든다.

    입력 점 수, raster 크기, contour 길이, crop 전후 점 수, PCD snapping 오차,
    최종 경로 길이·범위·시작/끝을 기록한다. 알고리즘 결과를 발표하거나 다른 PCD와
    비교할 때 화면 그림이 아닌 이 수치를 기준으로 판단한다.
    """
    return {
        "input_pcd": str(pcd_path),
        "input_points": header.points,
        "method": "XY outer-contour graph with original-PCD 3D boundary snapping",
        "axis_convention": {"horizontal": "X", "front_view_up": "Y", "depth": "Z"},
        "config_mm": asdict(config),
        "raster": {
            "rows": geometry.rows,
            "cols": geometry.cols,
            "source_point_count": raster_source_count,
            **dict(contour_stats),
        },
        "thin_face": {
            "raw_selected_point_count": face_raw_count,
            "voxelised_actual_point_count": int(len(face_points)),
            "bounds_mm": {
                "min": np.min(face_points, axis=0).astype(float).tolist(),
                "max": np.max(face_points, axis=0).astype(float).tolist(),
            },
        },
        "boundary_snap_error_mm": {
            "front_mean": float(np.mean(front_snap_distance)),
            "front_max": float(np.max(front_snap_distance)),
            "back_mean": float(np.mean(back_snap_distance)),
            "back_max": float(np.max(back_snap_distance)),
        },
        "final_front_path": {
            "waypoint_count": int(len(final_path)),
            "length_mm": float(final_distance[-1]),
            "bounds_mm": {
                "min": np.min(final_path, axis=0).tolist(),
                "max": np.max(final_path, axis=0).tolist(),
            },
            "start_mm": final_path[0].tolist(),
            "end_mm": final_path[-1].tolist(),
        },
    }


def extract_complete_surface(
    pcd_path: Path,
    output_dir: Path,
    config: SurfaceConfig,
    *,
    make_preview: bool = True,
) -> Mapping[str, Path]:
    """Step 1 전체 파이프라인을 순서대로 실행하는 최상위 처리 함수.

    Parameters
    ----------
    pcd_path:
        입력 ``AI_Glass_Front_0.010mm.pcd`` 경로.
    output_dir:
        PCD/CSV/NPZ/JSON/PNG를 저장할 디렉터리.
    config:
        ROI, raster, corridor, smoothing 및 최종 간격 설정.
    make_preview:
        False이면 계산 결과는 저장하되 무거운 Matplotlib preview만 생략한다.

    이 함수는 각 세부 함수를 아래 순서로 연결한다.

    ``PCD 열기 -> XY raster -> 상단 contour -> corridor crop -> Z 양 경계 ->``
    ``실제 PCD snapping -> smoothing/resampling -> 파일 저장``

    반환값은 생성된 결과 이름과 절대 파일 경로의 mapping이다.
    """
    validate_config(config)
    pcd_path = pcd_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {pcd_path}")
    header, records = open_binary_pcd(pcd_path)
    print(f"PCD: {header.points:,} points, record={records.dtype.itemsize} bytes")
    geometry = make_raster_geometry(config)
    print(
        f"XY raster: {geometry.cols:,} x {geometry.rows:,} at "
        f"{geometry.resolution_mm:g} mm"
    )

    # 1차 crop: 넓은 XY ROI를 정면 실루엣과 픽셀별 깊이 범위로 변환한다.
    occupied, min_z, max_z, raster_source_count = rasterise_xy_projection(
        records, geometry, config
    )
    # 순서가 있는 외곽 contour에서 좌우 끝을 잇는 상단 arc만 선택한다.
    arc_pixels, contour_stats = find_upper_outer_arc(occupied)
    arc_xy = geometry.pixels_to_xy(arc_pixels)
    print(
        f"Upper contour: {len(arc_pixels):,} pixels, "
        f"start={arc_xy[0].round(3).tolist()}, end={arc_xy[-1].round(3).tolist()}"
    )

    # 2차 핵심 crop: 상단 arc 주변의 좁은 corridor를 만들고 원본 3D 점을 재수집한다.
    corridor = build_face_corridor_mask(
        arc_pixels, geometry, config.face_half_width_mm
    )
    face_points, face_raw_count = extract_face_points(
        records, corridor, geometry, config
    )
    print(
        f"Thin face: {face_raw_count:,} raw source points -> "
        f"{len(face_points):,} actual voxel representatives"
    )

    # crop된 하나의 얇은 면에서 최소/최대 local Z 양쪽 깊이 경계를 분리한다.
    front_z, back_z = neighbourhood_extrema_on_arc(
        min_z,
        max_z,
        arc_pixels,
        geometry,
        config.edge_neighbourhood_mm,
    )
    front_targets = np.column_stack((arc_xy, front_z))
    back_targets = np.column_stack((arc_xy, back_z))
    # raster 후보를 실제 crop PCD 점으로 되돌려 최종 경계가 측정 map 위에 있게 한다.
    front_raw, front_snap = refine_boundary_targets(
        front_targets, face_points, config.raster_mm
    )
    back_raw, back_snap = refine_boundary_targets(
        back_targets, face_points, config.raster_mm
    )

    # PCD 요철을 완화하고 공정용 0.5 mm 간격 waypoint/tangent를 생성한다.
    final_front, front_distance, front_tangent, front_ideal = smooth_and_resample_edge(
        front_raw,
        config.smooth_window_mm,
        config.raster_mm,
        config.output_spacing_mm,
    )
    final_back, back_distance, back_tangent, back_ideal = smooth_and_resample_edge(
        back_raw,
        config.smooth_window_mm,
        config.raster_mm,
        config.output_spacing_mm,
    )

    face_pcd = output_dir / OUTPUT_THIN_FACE_PCD_NAME
    front_csv = output_dir / OUTPUT_FRONT_EDGE_CSV_NAME
    back_csv = output_dir / OUTPUT_BACK_EDGE_CSV_NAME
    front_raw_csv = output_dir / OUTPUT_FRONT_RAW_CSV_NAME
    back_raw_csv = output_dir / OUTPUT_BACK_RAW_CSV_NAME
    archive_npz = output_dir / OUTPUT_ARCHIVE_NPZ_NAME
    metadata_json = output_dir / OUTPUT_METADATA_JSON_NAME
    preview_png = output_dir / OUTPUT_PREVIEW_PNG_NAME

    write_binary_pcd(face_pcd, face_points)
    write_edge_csv(front_csv, final_front, front_distance, front_tangent)
    write_edge_csv(back_csv, final_back, back_distance, back_tangent)
    write_raw_edge_csv(front_raw_csv, front_raw)
    write_raw_edge_csv(back_raw_csv, back_raw)
    np.savez_compressed(
        archive_npz,
        thin_face_points_mm=face_points,
        contour_xy_mm=arc_xy,
        front_edge_raw_mm=front_raw,
        back_edge_raw_mm=back_raw,
        front_path_mm=final_front,
        back_path_mm=final_back,
        front_distance_mm=front_distance,
        back_distance_mm=back_distance,
        front_tangents=front_tangent,
        back_tangents=back_tangent,
        front_ideal_before_pcd_snap_mm=front_ideal,
        back_ideal_before_pcd_snap_mm=back_ideal,
    )
    metadata = build_metadata(
        pcd_path,
        header,
        config,
        geometry,
        raster_source_count,
        contour_stats,
        face_raw_count,
        face_points,
        front_snap,
        back_snap,
        final_front,
        front_distance,
    )
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    outputs = {
        "thin_face_pcd": face_pcd,
        "front_edge_csv": front_csv,
        "back_edge_csv": back_csv,
        "front_raw_csv": front_raw_csv,
        "back_raw_csv": back_raw_csv,
        "archive_npz": archive_npz,
        "metadata_json": metadata_json,
    }
    if make_preview:
        cloud_sample = sample_preview_cloud(records, config.preview_max_points)
        write_preview(
            preview_png,
            cloud_sample,
            face_points,
            final_front,
            final_back,
            final_front,
        )
        outputs["preview_png"] = preview_png

    print(
        f"Front path: {len(final_front)} points, {front_distance[-1]:.3f} mm, "
        f"start={final_front[0].round(4).tolist()}, "
        f"end={final_front[-1].round(4).tolist()}"
    )
    print(
        f"Front path bounds: min={np.min(final_front, axis=0).round(4).tolist()}, "
        f"max={np.max(final_front, axis=0).round(4).tolist()}"
    )
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs


def parse_args() -> argparse.Namespace:
    """명령행 옵션을 정의하고 사용자가 지정한 값을 ``argparse.Namespace``로 반환한다.

    파라미터를 생략하면 ``SurfaceConfig``와 같은 검증된 기본값을 사용한다. 다른 모델을
    사용할 때는 먼저 X/Y ROI와 축 방향을 확인한 뒤 값을 변경해야 한다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, default=INPUT_PCD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=-74.40)
    parser.add_argument("--x-max", type=float, default=74.40)
    parser.add_argument("--y-min", type=float, default=3.0)
    parser.add_argument("--y-max", type=float, default=21.0)
    parser.add_argument("--raster", type=float, default=0.02)
    parser.add_argument("--face-half-width", type=float, default=0.06)
    parser.add_argument("--face-voxel", type=float, default=0.02)
    parser.add_argument("--edge-neighbourhood", type=float, default=0.06)
    parser.add_argument("--smooth-window", type=float, default=0.40)
    parser.add_argument("--spacing", type=float, default=0.50)
    parser.add_argument("--chunk-points", type=int, default=5_000_000)
    parser.add_argument("--preview-max-points", type=int, default=100_000)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI 진입점: 인자를 설정 객체로 변환하고 Step 1 추출을 한 번 실행한다."""
    args = parse_args()
    config = SurfaceConfig(
        x_min_mm=args.x_min,
        x_max_mm=args.x_max,
        y_min_mm=args.y_min,
        y_max_mm=args.y_max,
        raster_mm=args.raster,
        face_half_width_mm=args.face_half_width,
        face_voxel_mm=args.face_voxel,
        edge_neighbourhood_mm=args.edge_neighbourhood,
        smooth_window_mm=args.smooth_window,
        output_spacing_mm=args.spacing,
        chunk_points=args.chunk_points,
        preview_max_points=args.preview_max_points,
    )
    extract_complete_surface(
        args.pcd,
        args.output_dir,
        config,
        make_preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
