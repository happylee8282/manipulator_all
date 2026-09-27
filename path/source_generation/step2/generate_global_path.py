#!/usr/bin/env python3
"""Step 2: 원본 PCD에서 얇은 본딩 윗면을 찾고 그 중앙 경로를 생성한다.

전체 처리 순서
1. 원본 PCD를 X-Y 격자로 투영하고 각 격자의 물리적인 윗면 Z를 저장한다.
2. X 위치마다 안경 바깥쪽 윤곽(outer edge)을 찾는다.
3. 바깥쪽에서 안쪽으로 Z 높이 변화를 검사하여 홈 쪽 경계(inner edge)를 찾는다.
4. 두 경계의 3D 중간값을 얇은 본딩 면의 중심(surface center)으로 사용한다.
5. 경계의 작은 raster 계단과 노이즈를 median/Savitzky-Golay 필터로 줄인다.
6. 경로를 일정한 거리 간격(기본 0.5 mm)으로 다시 샘플링한다.
7. 필요하면 양 끝의 급격한 직각 전환부를 경사각 기준으로 제외한다.
8. CSV, NPZ, PCD, 메타데이터 및 확인용 그림으로 저장한다.

주의
* 여기서 찾는 면은 Step 1 그림의 굵은 표시 아래에 있던 좁은 '윗면 띠'이다.
* Step 1에서 시각화했던 min-Z/max-Z 사이의 전체 깊이 단면을 경로로 쓰지 않는다.
* 최종 XY는 outer/inner 경계의 중앙이고, Z는 그 중앙 XY에서 측정한 PCD 윗면값이다.
* Step 2의 ``z_offset_mm`` 기본값은 0이다. 실제 위쪽 안전 높이는 Step 3에서
  world-Z 방향 ``surface_offset_mm``로 함께 이동시키도록 분리되어 있다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


from ..defaults import SOURCE_PCD, SOURCE_OUTPUT_ROOT

# =============================================================================
# USER INPUT / OUTPUT SETTINGS
# Step 2의 입력 PCD, 결과 폴더와 주요 결과 파일명은 여기서 한 번에 바꾼다.
# --input-pcd와 --output-dir를 지정하면 해당 실행에서만 이 값을 덮어쓴다.
# =============================================================================
INPUT_PCD = SOURCE_PCD
OUTPUT_DIR = SOURCE_OUTPUT_ROOT / "step2"

OUTPUT_PATH_CSV_NAME = "step2_global_path.csv"
OUTPUT_PATH_NPZ_NAME = "step2_global_path.npz"
OUTPUT_PATH_PCD_NAME = "step2_global_path.pcd"
OUTPUT_THIN_FACE_PCD_NAME = "step2_thin_bonding_face.pcd"
OUTPUT_METADATA_JSON_NAME = "step2_global_path_metadata.json"
OUTPUT_PREVIEW_PNG_NAME = "step2_global_path_preview.png"
OUTPUT_HEIGHT_PNG_NAME = "step2_z_height_visualization.png"
OUTPUT_THIN_FACE_PNG_NAME = "step2_thin_bonding_face_preview.png"

DEFAULT_PCD = INPUT_PCD
DEFAULT_OUTPUT_DIR = OUTPUT_DIR

from ..step1.extract_step1_pcd_path import iter_chunks, open_binary_pcd


@dataclass(frozen=True)
class GlobalPathConfig:
    """본딩 면 검출, 평활화, 끝점 제거 및 출력 간격 설정값."""

    # 원본 PCD에서 분석할 안경 윗부분의 XY 범위 [mm]
    x_min_mm: float = -65.60
    x_max_mm: float = 65.60
    y_min_mm: float = 10.0
    y_max_mm: float = 21.0
    # PCD를 XY 이미지로 바꾸는 한 픽셀의 크기 [mm]
    raster_mm: float = 0.02
    # outer edge로부터 inner edge를 찾을 거리 범위 [mm]
    inner_search_min_mm: float = 0.35
    inner_search_max_mm: float = 2.80
    # Z 단면 미분 전에 적용하는 Gaussian 필터 및 최소 경계 강도
    profile_sigma_mm: float = 0.03
    minimum_edge_strength: float = 0.15
    # 검출한 XY 경계와 Z 높이에 적용할 필터 창 길이 [mm]
    edge_median_window_mm: float = 0.50
    edge_smooth_window_mm: float = 3.00
    height_median_window_mm: float = 0.40
    height_smooth_window_mm: float = 1.20
    # 중앙 높이 주변에서 얇은 면 PCD로 인정할 Z 허용 범위 [mm]
    top_face_height_tolerance_mm: float = 0.15
    chunk_points: int = 5_000_000
    # 두 경계를 같은 파라미터로 맞출 때의 해상도와 후처리 필터 설정
    boundary_resolution_mm: float = 0.02
    median_window_mm: float = 0.30
    smooth_window_mm: float = 0.80
    # 최종 waypoint 간 거리 및 Step 2 local-Z 오프셋 [mm]
    output_spacing_mm: float = 0.50
    z_offset_mm: float = 0.0
    # 양 끝 직각 전환부 제거 여부와 경사각/최소 폭 조건
    trim_endpoint_transitions: bool = False
    endpoint_slope_threshold_deg: float = 55.0
    minimum_trimmed_span_mm: float = 130.0


def find_default_input() -> Path:
    """기본 원본 PCD가 존재하는지 확인하고 절대 경로를 반환한다."""
    if DEFAULT_PCD.exists():
        return DEFAULT_PCD
    raise FileNotFoundError(f"source PCD not found: {DEFAULT_PCD}")


def validate_config(config: GlobalPathConfig) -> None:
    """범위, 필터 길이, 경사각 등 설정값이 유효한지 실행 전에 검사한다."""
    if not config.x_min_mm < config.x_max_mm:
        raise ValueError("x_min_mm must be smaller than x_max_mm")
    if not config.y_min_mm < config.y_max_mm:
        raise ValueError("y_min_mm must be smaller than y_max_mm")
    for name in (
        "raster_mm",
        "inner_search_min_mm",
        "inner_search_max_mm",
        "profile_sigma_mm",
        "edge_median_window_mm",
        "edge_smooth_window_mm",
        "height_median_window_mm",
        "height_smooth_window_mm",
        "top_face_height_tolerance_mm",
        "boundary_resolution_mm",
        "median_window_mm",
        "smooth_window_mm",
        "output_spacing_mm",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.z_offset_mm < 0.0:
        raise ValueError("z_offset_mm must be non-negative")
    if not 0.0 < config.endpoint_slope_threshold_deg < 90.0:
        raise ValueError("endpoint_slope_threshold_deg must be in (0, 90)")
    if config.minimum_trimmed_span_mm <= 0.0:
        raise ValueError("minimum_trimmed_span_mm must be positive")
    if config.inner_search_min_mm >= config.inner_search_max_mm:
        raise ValueError("inner_search_min_mm must be smaller than inner_search_max_mm")
    if config.chunk_points <= 0:
        raise ValueError("chunk_points must be positive")


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    """순서가 있는 3D 점들의 누적 호 길이 ``[0, s1, ..., sn]``을 계산한다."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("a path needs at least two points")
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def remove_duplicate_xy(points: np.ndarray, tolerance_mm: float = 1.0e-8) -> np.ndarray:
    """NaN/Inf와 연속 중복 XY점을 제거하여 경계의 진행 순서를 정리한다."""
    points = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if len(points) < 2:
        raise ValueError("edge contains fewer than two finite points")
    keep = np.r_[
        True,
        np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > tolerance_mm,
    ]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("edge has no XY progression")
    return points


def ensure_same_direction(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """두 경계의 시작·끝 비용을 비교해 candidate의 진행 방향을 reference에 맞춘다."""
    same = np.linalg.norm(reference[0, :2] - candidate[0, :2]) + np.linalg.norm(
        reference[-1, :2] - candidate[-1, :2]
    )
    reversed_cost = np.linalg.norm(
        reference[0, :2] - candidate[-1, :2]
    ) + np.linalg.norm(reference[-1, :2] - candidate[0, :2])
    return candidate if same <= reversed_cost else candidate[::-1]


def odd_window(window_mm: float, sample_spacing_mm: float, count: int) -> int:
    """mm 단위 필터 길이를 데이터 개수를 넘지 않는 홀수 샘플 창으로 변환한다."""
    requested = max(3, int(round(window_mm / sample_spacing_mm)))
    if requested % 2 == 0:
        requested += 1
    maximum = count if count % 2 == 1 else count - 1
    return max(1, min(requested, maximum))


def rasterise_top_envelope(
    pcd_path: Path, config: GlobalPathConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """원본 PCD를 XY occupancy와 물리적 윗면 Z map으로 변환한다.

    PCD를 chunk 단위로 읽고 지정된 XY ROI 안의 점을 ``raster_mm`` 격자에 넣는다.
    각 셀에는 점 존재 여부와 local Z의 최솟값을 저장한다. 현재 검증된 Isaac 좌표
    변환에서는 model-local Z가 뒤집히므로, local Z가 가장 작은 점이 실제로 가장 높은
    표면점이다. 반환되는 ``min_z``가 이후 경계 및 경로의 굴곡 높이 원본이다.
    """

    header, records = open_binary_pcd(pcd_path)
    columns = int(math.ceil((config.x_max_mm - config.x_min_mm) / config.raster_mm))
    rows = int(math.ceil((config.y_max_mm - config.y_min_mm) / config.raster_mm))
    occupied = np.zeros(rows * columns, dtype=np.uint8)
    min_z = np.full(rows * columns, np.inf, dtype=np.float32)
    selected = 0

    for _, chunk in iter_chunks(records, config.chunk_points):
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
        if not np.any(mask):
            continue
        px = np.floor((x[mask] - config.x_min_mm) / config.raster_mm).astype(np.int32)
        py = np.floor((y[mask] - config.y_min_mm) / config.raster_mm).astype(np.int32)
        values = z[mask]
        valid = (px >= 0) & (px < columns) & (py >= 0) & (py < rows)
        flat = py[valid].astype(np.int64) * columns + px[valid]
        occupied[flat] = 1
        np.minimum.at(min_z, flat, values[valid])
        selected += int(np.sum(valid))

    occupied = occupied.reshape(rows, columns).astype(bool)
    min_z = min_z.reshape(rows, columns)
    x_values = config.x_min_mm + (np.arange(columns) + 0.5) * config.raster_mm
    y_values = config.y_min_mm + (np.arange(rows) + 0.5) * config.raster_mm
    return occupied, min_z, x_values, y_values, {
        "source_pcd_points": int(header.points),
        "roi_source_points": selected,
        "raster_rows": rows,
        "raster_columns": columns,
        "occupied_cells": int(np.sum(occupied)),
    }


def smooth_profile(values: np.ndarray, median_mm: float, smooth_mm: float, spacing_mm: float) -> np.ndarray:
    """median filter로 돌출 노이즈를 제거한 뒤 SG 필터로 곡선을 부드럽게 만든다."""
    result = np.asarray(values, dtype=np.float64).copy()
    median_size = odd_window(median_mm, spacing_mm, len(result))
    if median_size >= 3:
        result = median_filter(result, size=median_size, mode="nearest")
    smooth_size = odd_window(smooth_mm, spacing_mm, len(result))
    if smooth_size >= 5:
        result = savgol_filter(result, smooth_size, polyorder=3, mode="interp")
    return result


def sample_height_map(
    min_z: np.ndarray,
    occupied: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> np.ndarray:
    """요청 XY에서 top-envelope Z를 읽고, 빈 셀은 주변 5x5 중앙값으로 보완한다."""

    dx = float(x_values[1] - x_values[0])
    dy = float(y_values[1] - y_values[0])
    cols = np.clip(np.rint((query_x - x_values[0]) / dx).astype(int), 0, len(x_values) - 1)
    rows = np.clip(np.rint((query_y - y_values[0]) / dy).astype(int), 0, len(y_values) - 1)
    values = min_z[rows, cols].astype(np.float64)
    missing = ~np.isfinite(values)
    for index in np.flatnonzero(missing):
        row = rows[index]
        col = cols[index]
        r0, r1 = max(0, row - 2), min(min_z.shape[0], row + 3)
        c0, c1 = max(0, col - 2), min(min_z.shape[1], col + 3)
        patch = min_z[r0:r1, c0:c1]
        valid = occupied[r0:r1, c0:c1] & np.isfinite(patch)
        if np.any(valid):
            values[index] = float(np.median(patch[valid]))
    if not np.all(np.isfinite(values)):
        raise ValueError("top-envelope height is missing at one or more centreline stations")
    return values


def detect_top_wall_edges(
    occupied: np.ndarray,
    min_z: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: GlobalPathConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """각 X 열에서 얇은 윗면의 outer edge와 groove-side inner edge를 검출한다.

    처리 과정
    1. occupancy 열의 가장 큰 Y점을 바깥쪽 윤곽 ``outer_y``로 정한다.
    2. outer에서 안쪽으로 0.35~2.80 mm 범위의 local-Z 단면을 가져온다.
    3. Gaussian 필터 후 Z 미분값이 가장 강하게 변하는 지점을 inner edge로 정한다.
    4. 폭과 경계 강도 조건을 통과하지 못한 열은 인접 유효 열로 보간한다.
    5. 두 경계에 median + Savitzky-Golay 필터를 적용한다.
    6. ``center_y=(outer_y+inner_y)/2``에서 실제 PCD top-envelope Z를 읽는다.

    현재 Isaac 변환에서는 물리 높이가 ``-local_Z``이므로 미분 부호를 해석할 때 이
    좌표 반전을 반영한다. 반환되는 outer/inner는 각각 (X,Y,Z) 3D 경계이다.
    """

    outer_y = np.full(len(x_values), np.nan, dtype=np.float64)
    inner_y = np.full(len(x_values), np.nan, dtype=np.float64)
    strengths = np.full(len(x_values), np.nan, dtype=np.float64)
    min_pixels = max(1, int(round(config.inner_search_min_mm / config.raster_mm)))
    max_pixels = max(min_pixels + 2, int(round(config.inner_search_max_mm / config.raster_mm)))
    sigma_pixels = max(0.5, config.profile_sigma_mm / config.raster_mm)

    for column in range(len(x_values)):
        support = np.flatnonzero(occupied[:, column])
        if not len(support):
            continue
        outer_row = int(support[-1])
        lo = max(int(support[0]), outer_row - max_pixels)
        hi = outer_row - min_pixels
        if hi - lo < 4:
            continue
        rows = np.arange(lo, hi + 1)
        profile = min_z[rows, column].astype(np.float64)
        finite = np.isfinite(profile)
        if np.sum(finite) < 5:
            continue
        profile = np.interp(rows, rows[finite], profile[finite])
        profile = gaussian_filter1d(profile, sigma=sigma_pixels, mode="nearest")
        derivative = np.gradient(profile, config.raster_mm)
        edge_index = int(np.argmin(derivative))
        strength = float(-derivative[edge_index])
        outer_y[column] = y_values[outer_row]
        inner_y[column] = y_values[rows[edge_index]]
        strengths[column] = strength

    width = outer_y - inner_y
    valid = (
        np.isfinite(width)
        & (width >= config.inner_search_min_mm)
        & (width <= config.inner_search_max_mm)
        & (strengths >= config.minimum_edge_strength)
    )
    if np.mean(valid) < 0.80:
        raise ValueError(f"only {np.mean(valid):.1%} of X columns contain a valid raised-wall edge pair")

    indices = np.arange(len(x_values))
    outer_valid = np.isfinite(outer_y)
    outer_y[~outer_valid] = np.interp(indices[~outer_valid], indices[outer_valid], outer_y[outer_valid])
    inner_y[~valid] = np.interp(indices[~valid], indices[valid], inner_y[valid])
    outer_y = smooth_profile(
        outer_y, config.edge_median_window_mm, config.edge_smooth_window_mm, config.raster_mm
    )
    inner_y = smooth_profile(
        inner_y, config.edge_median_window_mm, config.edge_smooth_window_mm, config.raster_mm
    )
    width = outer_y - inner_y
    if np.any(width <= 0.0):
        raise ValueError("smoothed inner edge crosses the outer edge")

    center_y = 0.5 * (outer_y + inner_y)
    center_z = sample_height_map(
        min_z, occupied, x_values, y_values, x_values, center_y
    )
    center_z = smooth_profile(
        center_z,
        config.height_median_window_mm,
        config.height_smooth_window_mm,
        config.raster_mm,
    )
    outer = np.column_stack((x_values, outer_y, center_z))
    inner = np.column_stack((x_values, inner_y, center_z))
    return outer, inner, strengths, {
        "valid_edge_pair_fraction": float(np.mean(valid)),
        "raw_edge_strength_min": float(np.nanmin(strengths[valid])),
        "raw_edge_strength_median": float(np.nanmedian(strengths[valid])),
        "top_wall_width_min_mm": float(np.min(width)),
        "top_wall_width_mean_mm": float(np.mean(width)),
        "top_wall_width_max_mm": float(np.max(width)),
    }


def build_top_face_cloud(
    occupied: np.ndarray,
    min_z: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray,
    height_tolerance_mm: float,
) -> np.ndarray:
    """두 경계 사이이며 중앙 Z와 가까운 raster 윗면점을 얇은 본딩 면 PCD로 만든다."""

    rows, cols = np.nonzero(occupied & np.isfinite(min_z))
    point_x = x_values[cols]
    point_y = y_values[rows]
    outer_y = np.interp(point_x, outer[:, 0], outer[:, 1])
    inner_y = np.interp(point_x, inner[:, 0], inner[:, 1])
    centre_z = np.interp(point_x, outer[:, 0], outer[:, 2])
    point_z = min_z[rows, cols]
    selected = (
        (point_y >= inner_y)
        & (point_y <= outer_y)
        & (np.abs(point_z - centre_z) <= height_tolerance_mm)
    )
    return np.column_stack((point_x[selected], point_y[selected], point_z[selected])).astype(np.float32)


def extract_top_bonding_face(
    pcd_path: Path, config: GlobalPathConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """raster 생성→경계 검출→얇은 면 추출을 묶어 수행하고 통계를 반환한다."""
    occupied, min_z, x_values, y_values, raster_stats = rasterise_top_envelope(
        pcd_path, config
    )
    outer, inner, _, edge_stats = detect_top_wall_edges(
        occupied, min_z, x_values, y_values, config
    )
    thin_face = build_top_face_cloud(
        occupied,
        min_z,
        x_values,
        y_values,
        outer,
        inner,
        config.top_face_height_tolerance_mm,
    )
    return outer, inner, thin_face, {
        "outer_raw_count": int(len(outer)),
        "inner_raw_count": int(len(inner)),
        "common_boundary_count": int(len(outer)),
        "front_boundary_source": "PCD XY upper silhouette (outer edge)",
        "back_boundary_source": "PCD top-envelope first groove edge (inner edge)",
        "thin_face_point_count": int(len(thin_face)),
        **dict(raster_stats),
        **dict(edge_stats),
    }


def trim_right_angle_endpoint_transitions(
    outer: np.ndarray,
    inner: np.ndarray,
    thin_face: np.ndarray,
    config: GlobalPathConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """좌우 직각 전환부를 제외하고 첫 급경사부터 마지막 급경사까지만 남긴다.

    인접 중심점의 XY 이동량과 Z 변화량으로
    ``slope=atan2(|dZ|, sqrt(dX^2+dY^2))``를 계산한다. 기본 55도 이상인 첫 구간과
    마지막 구간을 생산 경로의 양 끝으로 사용한다. 이 55도는 제품의 설계 각도를
    의미하지 않고, 로봇 자세가 갑자기 바뀌는 양 끝 전환부를 찾는 검출 기준이다.
    """

    centre = 0.5 * (outer + inner)
    xy_step = np.linalg.norm(np.diff(centre[:, :2], axis=0), axis=1)
    z_step = np.abs(np.diff(centre[:, 2]))
    slope_deg = np.degrees(
        np.arctan2(z_step, np.maximum(xy_step, 1.0e-12))
    )
    steep = np.flatnonzero(slope_deg >= config.endpoint_slope_threshold_deg)
    if len(steep) < 2:
        raise ValueError(
            "could not find both steep endpoint faces; lower "
            "endpoint_slope_threshold_deg"
        )

    left_index = int(steep[0])
    right_index = int(steep[-1] + 1)
    if right_index <= left_index + 2:
        raise ValueError("endpoint slope trimming produced an empty path")

    outer_trimmed = outer[left_index : right_index + 1]
    inner_trimmed = inner[left_index : right_index + 1]
    centre_trimmed = 0.5 * (outer_trimmed + inner_trimmed)
    span_mm = float(np.ptp(centre_trimmed[:, 0]))
    if span_mm < config.minimum_trimmed_span_mm:
        raise ValueError(
            f"trimmed path span {span_mm:.3f} mm is below the configured "
            f"minimum {config.minimum_trimmed_span_mm:.3f} mm"
        )

    x_low = float(np.min(centre_trimmed[:, 0]))
    x_high = float(np.max(centre_trimmed[:, 0]))
    face_mask = (thin_face[:, 0] >= x_low) & (thin_face[:, 0] <= x_high)
    thin_face_trimmed = thin_face[face_mask]
    return outer_trimmed, inner_trimmed, thin_face_trimmed, {
        "endpoint_trim_enabled": True,
        "endpoint_slope_threshold_deg": float(
            config.endpoint_slope_threshold_deg
        ),
        "left_source_index": left_index,
        "right_source_index": right_index,
        "left_endpoint_x_mm": x_low,
        "right_endpoint_x_mm": x_high,
        "trimmed_x_span_mm": span_mm,
        "left_first_segment_slope_deg": float(slope_deg[left_index]),
        "right_last_segment_slope_deg": float(slope_deg[right_index - 1]),
        "excluded_left_point_count": left_index,
        "excluded_right_point_count": int(len(centre) - right_index - 1),
        "thin_face_point_count": int(len(thin_face_trimmed)),
    }


def parameterise_edge_xy(
    points: np.ndarray,
    common_u: np.ndarray,
    config: GlobalPathConfig,
) -> np.ndarray:
    """경계 하나를 정규화 XY 호 길이 ``u=0~1``에 맞춰 보간하고 평활화한다.

    outer와 inner의 원본 점 개수 및 간격이 다르면 같은 인덱스끼리 바로 평균할 수
    없다. 따라서 두 경계를 각각 XY 누적 길이 기준으로 parameterisation한 다음 동일한
    ``common_u`` 위치에 재배치한다. XY에는 SG 필터를, Z에는 median+SG 필터를 적용하며
    측정된 시작점과 끝점은 변경하지 않는다.
    """

    points = remove_duplicate_xy(points)
    xy_s = cumulative_distance(points[:, :2])
    if xy_s[-1] <= 1.0e-9:
        raise ValueError("edge XY length is zero")
    source_u = xy_s / xy_s[-1]
    edge = np.column_stack(
        [np.interp(common_u, source_u, points[:, axis]) for axis in range(3)]
    )

    nominal_spacing = xy_s[-1] / max(len(common_u) - 1, 1)
    median_size = odd_window(
        config.median_window_mm, nominal_spacing, len(common_u)
    )
    smooth_size = odd_window(
        config.smooth_window_mm, nominal_spacing, len(common_u)
    )

    # X/Y already follow an ordered contour.  Apply only a mild polynomial
    # smoothing to remove raster stairs.  Z additionally receives a robust
    # median filter so isolated maximum-Z points on the inner edge cannot pull
    # the midpoint path away from the physical strip.
    result = edge.copy()
    if smooth_size >= 5:
        for axis in (0, 1):
            result[:, axis] = savgol_filter(
                edge[:, axis], smooth_size, polyorder=3, mode="interp"
            )
    robust_z = median_filter(edge[:, 2], size=median_size, mode="nearest")
    if smooth_size >= 5:
        robust_z = savgol_filter(
            robust_z, smooth_size, polyorder=3, mode="interp"
        )
    result[:, 2] = robust_z

    # Keep measured edge endpoints unchanged.
    result[0] = edge[0]
    result[-1] = edge[-1]
    return result


def load_and_pair_edges(
    archive_path: Path, config: GlobalPathConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """이전 Step1 NPZ 경계를 읽어 방향과 점 개수를 맞추는 호환용 경로이다.

    현재 기본 실행은 원본 PCD에서 ``extract_top_bonding_face``를 사용하지만, 예전
    Step1 archive를 입력하는 경우를 위해 남겨 둔 함수다. 가능한 경우 원본 PCD에
    snap된 front/back path를 우선 사용하고, 없으면 raw edge 배열을 사용한다.
    """
    with np.load(archive_path) as archive:
        # Prefer Step 1's final end-to-end, original-PCD-snapped paths.  The
        # raw arrays remain a backward-compatible fallback for old archives.
        # This prevents Step 2 from rebuilding a shorter, interior contour
        # and guarantees that the two downturned end curves are retained.
        use_snapped_paths = {"front_path_mm", "back_path_mm"}.issubset(archive.files)
        raw_pair_available = {
            "front_edge_raw_mm", "back_edge_raw_mm"
        }.issubset(archive.files)
        if not use_snapped_paths and not raw_pair_available:
            raise ValueError(
                "Step 1 archive needs front_path_mm/back_path_mm or "
                "front_edge_raw_mm/back_edge_raw_mm"
            )
        front_name = "front_path_mm" if use_snapped_paths else "front_edge_raw_mm"
        back_name = "back_path_mm" if use_snapped_paths else "back_edge_raw_mm"
        outer_raw = np.asarray(archive[front_name], dtype=np.float64)
        inner_raw = np.asarray(archive[back_name], dtype=np.float64)
        thin_face = (
            np.asarray(archive["thin_face_points_mm"], dtype=np.float32)
            if "thin_face_points_mm" in archive.files
            else np.empty((0, 3), dtype=np.float32)
        )

    outer_raw = remove_duplicate_xy(outer_raw)
    inner_raw = ensure_same_direction(outer_raw, remove_duplicate_xy(inner_raw))
    outer_xy_length = cumulative_distance(outer_raw[:, :2])[-1]
    inner_xy_length = cumulative_distance(inner_raw[:, :2])[-1]
    common_length = 0.5 * (outer_xy_length + inner_xy_length)
    common_count = max(
        3, int(math.ceil(common_length / config.boundary_resolution_mm)) + 1
    )
    common_u = np.linspace(0.0, 1.0, common_count)
    outer = parameterise_edge_xy(outer_raw, common_u, config)
    inner = parameterise_edge_xy(inner_raw, common_u, config)
    return outer, inner, thin_face, {
        "outer_raw_count": int(len(outer_raw)),
        "inner_raw_count": int(len(inner_raw)),
        "common_boundary_count": int(common_count),
        "front_boundary_source": front_name,
        "back_boundary_source": back_name,
    }


def resample_paired_path(
    outer: np.ndarray,
    inner: np.ndarray,
    config: GlobalPathConfig,
) -> Mapping[str, np.ndarray]:
    """두 3D 경계의 중간선을 계산하고 일정 간격의 최종 global path를 만든다.

    XY의 핵심 식은 ``center_y=(outer_y+inner_y)/2``이고 Z는 이 중앙 XY에서 읽은
    PCD top-envelope 높이다. 코드에서는 outer/inner 배열에 같은 중앙 Z를 넣었으므로
    ``surface_center=(outer+inner)/2``로 3D 중심선을 얻을 수 있다. 이 중심선을 3D 누적
    거리 기준 기본 0.5 mm 간격으로 보간한다. Step2 local-Z offset이 설정되었다면 최종
    경로의 Z에만 더한다. 이후 각 waypoint의 단위 접선과 면 폭도 계산한다.
    """

    # outer/inner는 같은 X 진행 순서와 대응 위치를 가진 경계다. XY는 서로 다르지만
    # Z에는 중앙 XY에서 측정한 동일한 윗면 높이가 들어 있다. 따라서 대응점의 산술
    # 평균으로 얇은 본딩 윗면의 중앙 XYZ 좌표를 얻는다.
    surface_center = 0.5 * (outer + inner)

    # 중심선의 실제 3D 길이를 기준으로 일정 거리의 목표 station을 만든다.
    source_s = cumulative_distance(surface_center)
    targets = np.arange(0.0, source_s[-1], config.output_spacing_mm)
    if not np.isclose(targets[-1], source_s[-1]):
        targets = np.r_[targets, source_s[-1]]

    def interpolate(values: np.ndarray) -> np.ndarray:
        """원본 누적 거리에서 일정 간격 target 거리로 XYZ 좌표를 선형 보간한다."""
        return np.column_stack(
            [np.interp(targets, source_s, values[:, axis]) for axis in range(3)]
        )

    # 두 경계를 동일 station에 보간한 뒤 다시 평균하여 대응 관계를 보존한다.
    outer_sampled = interpolate(outer)
    inner_sampled = interpolate(inner)
    surface_sampled = 0.5 * (outer_sampled + inner_sampled)
    # Step2 오프셋은 model-local Z에만 적용한다. 기본값 0이면 면 중심과 동일하다.
    # 로봇 실행 시의 위쪽 안전 높이는 Step3 surface_offset_mm가 담당한다.
    global_path = surface_sampled.copy()
    global_path[:, 2] += config.z_offset_mm
    global_distance = cumulative_distance(global_path)
    # 인접 waypoint의 미분으로 팁 자세 생성에 사용할 단위 접선을 계산한다.
    derivatives = np.gradient(global_path, global_distance, axis=0, edge_order=1)
    tangent_norm = np.linalg.norm(derivatives, axis=1, keepdims=True)
    tangents = derivatives / np.maximum(tangent_norm, 1.0e-12)
    width = np.linalg.norm(inner_sampled - outer_sampled, axis=1)

    return {
        "outer_edge_mm": outer_sampled,
        "inner_edge_mm": inner_sampled,
        "surface_center_mm": surface_sampled,
        "global_path_mm": global_path,
        "distance_mm": global_distance,
        "tangents": tangents,
        "face_width_mm": width,
    }


def attach_thin_face_fit(
    result: Mapping[str, np.ndarray], thin_face: np.ndarray
) -> Mapping[str, np.ndarray]:
    """KD-tree 최근접 거리로 중심 경로가 추출된 얇은 면 위에 있는지 검증한다."""

    updated = dict(result)
    if len(thin_face):
        distances, _ = cKDTree(np.asarray(thin_face, dtype=np.float64)).query(
            np.asarray(result["surface_center_mm"], dtype=np.float64), k=1
        )
    else:
        distances = np.full(len(result["surface_center_mm"]), np.nan)
    updated["surface_to_thin_face_pcd_mm"] = np.asarray(distances, dtype=np.float64)
    return updated


def write_csv(
    path: Path, result: Mapping[str, np.ndarray], z_offset_mm: float
) -> None:
    """경로, 면 중심, 양 경계, 폭, 접선과 검증값을 waypoint별 CSV로 저장한다."""
    outer = result["outer_edge_mm"]
    inner = result["inner_edge_mm"]
    surface = result["surface_center_mm"]
    global_path = result["global_path_mm"]
    distance = result["distance_mm"]
    tangents = result["tangents"]
    width = result["face_width_mm"]
    face_fit = result["surface_to_thin_face_pcd_mm"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "index",
                "path_distance_mm",
                "global_x_mm",
                "global_y_mm",
                "global_z_mm",
                "surface_center_x_mm",
                "surface_center_y_mm",
                "surface_center_z_mm",
                "outer_x_mm",
                "outer_y_mm",
                "outer_z_mm",
                "inner_x_mm",
                "inner_y_mm",
                "inner_z_mm",
                "face_width_mm",
                "surface_to_thin_face_pcd_mm",
                "z_offset_mm",
                "tangent_x",
                "tangent_y",
                "tangent_z",
            )
        )
        for index in range(len(global_path)):
            writer.writerow(
                (
                    index,
                    f"{distance[index]:.9f}",
                    *[f"{value:.9f}" for value in global_path[index]],
                    *[f"{value:.9f}" for value in surface[index]],
                    *[f"{value:.9f}" for value in outer[index]],
                    *[f"{value:.9f}" for value in inner[index]],
                    f"{width[index]:.9f}",
                    f"{face_fit[index]:.9f}",
                    f"{z_offset_mm:.9f}",
                    *[f"{value:.9f}" for value in tangents[index]],
                )
            )


def write_binary_pcd(path: Path, points: np.ndarray) -> None:
    """XYZ 점 배열을 PCD v0.7 binary 형식으로 저장한다."""
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


def write_preview(
    path: Path,
    result: Mapping[str, np.ndarray],
    thin_face: np.ndarray,
) -> None:
    """XY/XZ 경계, 최종 경로, 오프셋 및 좌측 끝 상세도를 PNG로 저장한다."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step2_pcd_path_matplotlib")
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unable to import Axes3D.*", category=UserWarning
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    outer = result["outer_edge_mm"]
    inner = result["inner_edge_mm"]
    surface = result["surface_center_mm"]
    global_path = result["global_path_mm"]
    distance = result["distance_mm"]
    coordinate_offset = global_path - surface
    z_offset_mm = float(np.mean(coordinate_offset[:, 2]))
    if len(thin_face):
        stride = max(1, int(math.ceil(len(thin_face) / 100_000)))
        face_sample = thin_face[::stride]
    else:
        face_sample = np.empty((0, 3))

    figure, axes = plt.subplots(2, 2, figsize=(17, 10))
    xy, xz, profiles, left_zoom = axes.ravel()
    if len(face_sample):
        xy.scatter(face_sample[:, 0], face_sample[:, 1], s=0.3, c="cyan", alpha=0.18)
        xz.scatter(face_sample[:, 0], face_sample[:, 2], s=0.3, c="cyan", alpha=0.18)
    xy.plot(outer[:, 0], outer[:, 1], color="red", linewidth=1.0, label="outer edge")
    xy.plot(inner[:, 0], inner[:, 1], color="blue", linewidth=1.0, label="groove-side inner edge")
    xy.plot(surface[:, 0], surface[:, 1], color="green", linewidth=2.0, label="thin-face midpoint XY")
    xy.plot(global_path[:, 0], global_path[:, 1], color="magenta", linewidth=2.0, linestyle="--", label="final path XY (same coordinates)")
    xy.set(title="Top view XY: extracted raised wall and its midpoint", xlabel="X [mm]", ylabel="Y [mm]")
    xy.set_aspect("equal", adjustable="box")
    xy.grid(alpha=0.25)
    xy.legend(loc="best", ncol=2, fontsize=8)

    xz.plot(outer[:, 0], outer[:, 2], color="red", linewidth=1.2, label="outer edge on top wall")
    xz.plot(inner[:, 0], inner[:, 2], color="blue", linewidth=1.2, label="inner edge on top wall")
    xz.plot(surface[:, 0], surface[:, 2], color="green", linewidth=2.0, label="surface midpoint")
    xz.plot(global_path[:, 0], global_path[:, 2], color="magenta", linewidth=1.8, linestyle="--", label=f"Step 2 path (local Z offset {z_offset_mm:g} mm)")
    xz.set(title="XZ: steep-face start to steep-face end", xlabel="X [mm]", ylabel="Z [mm]")
    xz.grid(alpha=0.25)
    xz.legend()

    profiles.plot(distance, coordinate_offset[:, 0], linewidth=1.5, label="ΔX = 0")
    profiles.plot(distance, coordinate_offset[:, 1], linewidth=1.5, label="ΔY = 0")
    profiles.plot(distance, coordinate_offset[:, 2], color="magenta", linewidth=2.0, label=f"ΔZ = {z_offset_mm:g} mm")
    profiles.set(
        title="Coordinate offset verification",
        xlabel="global path distance [mm]",
        ylabel="global path - surface midpoint [mm]",
    )
    profiles.set_ylim(min(-0.01, z_offset_mm - 0.02), max(0.02, z_offset_mm + 0.02))
    profiles.grid(alpha=0.25)
    profiles.legend()

    side = global_path[:, 0] < -62.0
    left_zoom.plot(outer[side, 0], outer[side, 2], color="red", linewidth=1.5)
    left_zoom.plot(inner[side, 0], inner[side, 2], color="blue", linewidth=1.5)
    left_zoom.plot(surface[side, 0], surface[side, 2], color="green", linewidth=2.0)
    left_zoom.plot(global_path[side, 0], global_path[side, 2], color="magenta", linewidth=1.8, linestyle="--")
    left_zoom.set(title="Left end XZ detail", xlabel="X [mm]", ylabel="Z [mm]")
    left_zoom.grid(alpha=0.25)

    figure.suptitle("Step 2: steep-face-to-steep-face thin bonding midpoint", fontsize=18, weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_z_height_visualization(
    path: Path,
    result: Mapping[str, np.ndarray],
    thin_face: np.ndarray,
    z_offset_mm: float,
) -> None:
    """XY가 유지되고 설정한 local-Z 오프셋만 적용됐는지 별도 그림으로 확인한다."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step2_pcd_path_matplotlib")
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unable to import Axes3D.*", category=UserWarning
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    outer = result["outer_edge_mm"]
    inner = result["inner_edge_mm"]
    surface = result["surface_center_mm"]
    global_path = result["global_path_mm"]
    distance = result["distance_mm"]
    offset_vectors = global_path - surface

    if len(thin_face):
        stride = max(1, int(math.ceil(len(thin_face) / 80_000)))
        face_sample = thin_face[::stride]
    else:
        face_sample = np.empty((0, 3))

    figure = plt.figure(figsize=(16, 11))
    grid = figure.add_gridspec(3, 1, height_ratios=(1.35, 1.0, 0.72))
    xy = figure.add_subplot(grid[0])
    components = figure.add_subplot(grid[1])
    magnitude = figure.add_subplot(grid[2])

    if len(face_sample):
        xy.scatter(
            face_sample[:, 0],
            face_sample[:, 1],
            s=0.25,
            c="0.75",
            alpha=0.18,
            label="PCD-derived raised thin face",
        )
    xy.plot(outer[:, 0], outer[:, 1], color="red", linewidth=0.9, label="Outer edge")
    xy.plot(inner[:, 0], inner[:, 1], color="blue", linewidth=0.9, label="Inner groove edge")
    xy.plot(
        surface[:, 0],
        surface[:, 1],
        color="green",
        linewidth=4.0,
        label="Boundary midpoint in XY",
    )
    xy.plot(
        global_path[:, 0],
        global_path[:, 1],
        color="magenta",
        linewidth=1.9,
        linestyle="--",
        label="Final path XY (same midpoint)",
    )
    xy.set(
        title="1. XY plane: the final path keeps the thin-face midpoint X/Y",
        xlabel="X [mm]",
        ylabel="Y [mm]",
    )
    xy.set_aspect("equal", adjustable="box")
    xy.grid(alpha=0.25)
    xy.legend(loc="best", ncol=2)

    components.plot(distance, offset_vectors[:, 0], linewidth=1.6, label="ΔX")
    components.plot(distance, offset_vectors[:, 1], linewidth=1.9, label="ΔY")
    components.plot(distance, offset_vectors[:, 2], linewidth=1.6, label="ΔZ")
    components.axhline(0.0, color="black", linewidth=0.8)
    components.set(
        title=f"2. Step 2 local offset: X/Y zero, Z = {z_offset_mm:g} mm",
        xlabel="Path distance [mm]",
        ylabel="global path - surface center [mm]",
    )
    components.grid(alpha=0.25)
    components.legend(loc="best", ncol=3)

    magnitude.plot(
        distance,
        offset_vectors[:, 2],
        color="magenta",
        linewidth=2.5,
        label=f"Z height increase = {z_offset_mm:g} mm",
    )
    magnitude.fill_between(distance, 0.0, offset_vectors[:, 2], color="cyan", alpha=0.65)
    magnitude.set_ylim(-0.01, max(0.07, z_offset_mm * 1.35))
    magnitude.set(
        title=f"3. Z verification: path - surface = {z_offset_mm:g} mm",
        xlabel="Path distance [mm]",
        ylabel="Z height difference [mm]",
    )
    magnitude.grid(alpha=0.25)
    magnitude.legend(loc="upper right")

    figure.suptitle(
        "Step 2 thin-face midpoint path — height clearance is applied in Step 3",
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.5,
        0.005,
        f"Step 2 formula: path = surface_center + (0, 0, {z_offset_mm:g} mm)",
        ha="center",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.965))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_thin_face_preview(
    path: Path, result: Mapping[str, np.ndarray], thin_face: np.ndarray
) -> None:
    """안경 전체를 제외하고 얇은 면의 Z 높이와 중심 경로 경사각을 색으로 표시한다."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step2_pcd_path_matplotlib")
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unable to import Axes3D.*", category=UserWarning
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    outer = result["outer_edge_mm"]
    inner = result["inner_edge_mm"]
    centre = result["surface_center_mm"]
    stride = max(1, int(math.ceil(len(thin_face) / 180_000)))
    face = thin_face[::stride]
    figure, (height_axis, slope_axis) = plt.subplots(
        2, 1, figsize=(16, 8.5), gridspec_kw={"height_ratios": (1.15, 1.0)}
    )

    height_norm = Normalize(vmin=float(np.min(face[:, 2])), vmax=float(np.max(face[:, 2])))
    height_points = height_axis.scatter(
        face[:, 0],
        face[:, 1],
        c=face[:, 2],
        cmap="turbo",
        norm=height_norm,
        s=1.2,
        alpha=0.82,
        label="PCD-derived raised top face",
    )
    height_axis.plot(outer[:, 0], outer[:, 1], color="black", linewidth=0.8, label="Outer edge")
    height_axis.plot(inner[:, 0], inner[:, 1], color="white", linewidth=0.8, label="Groove-side inner edge")
    height_axis.plot(centre[:, 0], centre[:, 1], color="#00ff4c", linewidth=1.3, label="Face midpoint")
    height_axis.set(
        title="PCD-derived thin bonding face — color represents local Z height",
        xlabel="X [mm]",
        ylabel="Y [mm]",
    )
    height_axis.set_aspect("equal", adjustable="box")
    height_axis.grid(alpha=0.22)
    height_axis.legend(loc="lower center", ncol=4)
    height_colorbar = figure.colorbar(height_points, ax=height_axis, pad=0.015)
    height_colorbar.set_label("Surface local Z [mm]")

    xy_step = np.linalg.norm(np.diff(centre[:, :2], axis=0), axis=1)
    z_step = np.abs(np.diff(centre[:, 2]))
    slope_deg = np.degrees(np.arctan2(z_step, np.maximum(xy_step, 1.0e-12)))
    path_segments = np.stack((centre[:-1, [0, 2]], centre[1:, [0, 2]]), axis=1)
    slope_lines = LineCollection(
        path_segments,
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=max(1.0, float(np.max(slope_deg)))),
        linewidth=3.0,
    )
    slope_lines.set_array(slope_deg)
    slope_axis.add_collection(slope_lines)
    slope_axis.autoscale()
    slope_axis.set(
        title="Centre-path steepness — color represents slope angle from the XY plane",
        xlabel="X [mm]",
        ylabel="Surface local Z [mm]",
    )
    slope_axis.grid(alpha=0.22)
    slope_colorbar = figure.colorbar(slope_lines, ax=slope_axis, pad=0.015)
    slope_colorbar.set_label("Slope angle [deg]")

    figure.suptitle("Step 2 thin-face height and path steepness", fontsize=17, weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_metadata(
    input_path: Path,
    config: GlobalPathConfig,
    counts: Mapping[str, object],
    result: Mapping[str, np.ndarray],
) -> Mapping[str, object]:
    """입력·설정·경계 출처·경로 길이·오차 통계를 JSON용 dictionary로 구성한다."""
    global_path = result["global_path_mm"]
    surface = result["surface_center_mm"]
    distance = result["distance_mm"]
    width = result["face_width_mm"]
    face_fit = result["surface_to_thin_face_pcd_mm"]
    offset_vectors = global_path - surface
    return {
        "input_source_pcd": str(input_path),
        "coordinate_frame": "source PCD model-local",
        "units": "millimetres",
        "edge_mapping": {
            "outer_edge": counts["front_boundary_source"],
            "groove_side_inner_edge": counts["back_boundary_source"],
        },
        "method": "PCD XY top-envelope raised-wall extraction; right-angle end transitions removed; outer/groove edge midpoint",
        "config": asdict(config),
        "counts": dict(counts),
        "result": {
            "waypoint_count": int(len(global_path)),
            "path_length_mm": float(distance[-1]),
            "start_mm": global_path[0].tolist(),
            "end_mm": global_path[-1].tolist(),
            "bounds_mm": {
                "min": np.min(global_path, axis=0).tolist(),
                "max": np.max(global_path, axis=0).tolist(),
            },
            "z_offset_mm": {
                "min": float(np.min(offset_vectors[:, 2])),
                "mean": float(np.mean(offset_vectors[:, 2])),
                "max": float(np.max(offset_vectors[:, 2])),
                "max_abs_from_requested": float(
                    np.max(np.abs(offset_vectors[:, 2] - config.z_offset_mm))
                ),
            },
            "xy_offset_max_abs_mm": {
                "x": float(np.max(np.abs(offset_vectors[:, 0]))),
                "y": float(np.max(np.abs(offset_vectors[:, 1]))),
            },
            "face_width_mm": {
                "min": float(np.min(width)),
                "mean": float(np.mean(width)),
                "max": float(np.max(width)),
            },
            "surface_midpoint_to_thin_face_pcd_mm": {
                "mean": float(np.nanmean(face_fit)),
                "p95": float(np.nanpercentile(face_fit, 95.0)),
                "max": float(np.nanmax(face_fit)),
            },
        },
    }


def generate_global_path(
    input_path: Path,
    output_dir: Path,
    config: GlobalPathConfig,
    *,
    make_preview: bool = True,
) -> Mapping[str, Path]:
    """Step2 전체 파이프라인을 실행하고 생성된 결과 파일 경로를 반환한다.

    실행 순서
    1. 설정 검사 및 출력 폴더 생성
    2. 원본 PCD에서 outer/inner 경계와 얇은 본딩 면 검출
    3. 선택 시 양 끝 직각 전환부 제거
    4. 두 경계의 중앙 계산 및 0.5 mm 간격 재샘플링
    5. 면 PCD 최근접 거리로 경로 위치 검증
    6. CSV/NPZ/PCD/JSON/PNG 저장
    """
    validate_config(config)
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 원본 PCD top-envelope에서 본딩 띠의 양 경계를 직접 검출한다.
    outer, inner, thin_face, counts = extract_top_bonding_face(input_path, config)
    if config.trim_endpoint_transitions:
        outer, inner, thin_face, trim_stats = trim_right_angle_endpoint_transitions(
            outer, inner, thin_face, config
        )
        counts = {**dict(counts), **dict(trim_stats)}
    else:
        counts = {**dict(counts), "endpoint_trim_enabled": False}
    # 2) 양 경계 중간 경로를 만들고 실제 면 PCD와의 거리를 검증한다.
    result = attach_thin_face_fit(
        resample_paired_path(outer, inner, config), thin_face
    )

    csv_path = output_dir / OUTPUT_PATH_CSV_NAME
    npz_path = output_dir / OUTPUT_PATH_NPZ_NAME
    pcd_path = output_dir / OUTPUT_PATH_PCD_NAME
    thin_face_pcd_path = output_dir / OUTPUT_THIN_FACE_PCD_NAME
    metadata_path = output_dir / OUTPUT_METADATA_JSON_NAME
    preview_path = output_dir / OUTPUT_PREVIEW_PNG_NAME
    z_height_preview_path = output_dir / OUTPUT_HEIGHT_PNG_NAME
    thin_face_preview_path = output_dir / OUTPUT_THIN_FACE_PNG_NAME

    # 3) 로봇이 읽을 경로와 사람이 검토할 분석 결과를 여러 형식으로 출력한다.
    write_csv(csv_path, result, config.z_offset_mm)
    np.savez_compressed(npz_path, thin_face_points_mm=thin_face, **result)
    write_binary_pcd(pcd_path, result["global_path_mm"])
    write_binary_pcd(thin_face_pcd_path, thin_face)
    metadata = build_metadata(input_path, config, counts, result)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs = {
        "csv": csv_path,
        "npz": npz_path,
        "pcd": pcd_path,
        "thin_face_pcd": thin_face_pcd_path,
        "metadata": metadata_path,
    }
    if make_preview:
        write_preview(preview_path, result, thin_face)
        write_z_height_visualization(
            z_height_preview_path, result, thin_face, config.z_offset_mm
        )
        write_thin_face_preview(thin_face_preview_path, result, thin_face)
        outputs["preview"] = preview_path
        outputs["z_height_visualization"] = z_height_preview_path
        outputs["thin_face_preview"] = thin_face_preview_path

    global_path = result["global_path_mm"]
    distance = result["distance_mm"]
    spacing = np.diff(distance)
    offset = global_path - result["surface_center_mm"]
    print(f"Input: {input_path}")
    print(
        f"Boundaries: outer={counts['outer_raw_count']:,}, "
        f"inner={counts['inner_raw_count']:,}, paired={counts['common_boundary_count']:,}"
    )
    print(
        "Boundary sources: "
        f"{counts['front_boundary_source']} / {counts['back_boundary_source']}"
    )
    print(
        f"Global path: {len(global_path)} points, length={distance[-1]:.3f} mm, "
        f"spacing mean={np.mean(spacing):.4f} mm"
    )
    print(
        f"Offset: dX max={np.max(np.abs(offset[:, 0])):.6f}, "
        f"dY max={np.max(np.abs(offset[:, 1])):.6f}, "
        f"dZ min/mean/max={np.min(offset[:, 2]):.6f}/"
        f"{np.mean(offset[:, 2]):.6f}/{np.max(offset[:, 2]):.6f} mm"
    )
    face_fit = result["surface_to_thin_face_pcd_mm"]
    print(
        "Midpoint -> thin-face PCD nearest distance: "
        f"mean={np.nanmean(face_fit):.6f}, "
        f"p95={np.nanpercentile(face_fit, 95.0):.6f}, "
        f"max={np.nanmax(face_fit):.6f} mm"
    )
    print(f"Start: {global_path[0].round(6).tolist()}")
    print(f"End:   {global_path[-1].round(6).tolist()}")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs


def parse_args() -> argparse.Namespace:
    """명령행 인자를 정의하고 사용자가 조절한 Step2 파라미터를 읽는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--input-pcd",
        dest="input",
        type=Path,
        help="original AI Glass XYZ binary PCD",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=-74.36, help="PCD analysis left limit [mm]")
    parser.add_argument("--x-max", type=float, default=74.36, help="PCD analysis right limit [mm]")
    parser.add_argument("--y-min", type=float, default=10.0)
    parser.add_argument("--y-max", type=float, default=21.0)
    parser.add_argument("--raster", type=float, default=0.02, help="XY top-envelope grid [mm]")
    parser.add_argument("--inner-search-min", type=float, default=0.35)
    parser.add_argument("--inner-search-max", type=float, default=2.80)
    parser.add_argument("--minimum-edge-strength", type=float, default=0.15)
    parser.add_argument("--boundary-resolution", type=float, default=0.02)
    parser.add_argument("--median-window", type=float, default=0.30)
    parser.add_argument("--smooth-window", type=float, default=0.80)
    parser.add_argument("--spacing", type=float, default=0.50)
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Step 2 model-local Z offset; keep 0 and tune world-Z height in Step 3",
    )
    parser.add_argument(
        "--endpoint-slope-deg",
        type=float,
        default=55.0,
        help="start/end on the first/last face this steep from the XY plane",
    )
    parser.add_argument(
        "--no-endpoint-trim",
        action="store_true",
        help="retain the horizontal end tabs and right-angle transitions",
    )
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI 인자로 설정 객체를 만든 뒤 Step2 경로 생성 파이프라인을 실행한다."""
    args = parse_args()
    input_path = args.input if args.input is not None else find_default_input()
    config = GlobalPathConfig(
        x_min_mm=args.x_min,
        x_max_mm=args.x_max,
        y_min_mm=args.y_min,
        y_max_mm=args.y_max,
        raster_mm=args.raster,
        inner_search_min_mm=args.inner_search_min,
        inner_search_max_mm=args.inner_search_max,
        minimum_edge_strength=args.minimum_edge_strength,
        boundary_resolution_mm=args.boundary_resolution,
        median_window_mm=args.median_window,
        smooth_window_mm=args.smooth_window,
        output_spacing_mm=args.spacing,
        z_offset_mm=args.z_offset,
        trim_endpoint_transitions=not args.no_endpoint_trim,
        endpoint_slope_threshold_deg=args.endpoint_slope_deg,
    )
    generate_global_path(
        input_path,
        args.output_dir,
        config,
        make_preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
