#!/usr/bin/env python3
"""Remove short-wavelength PCD waviness from the Step 2 bonding path.

The large glasses-frame shape is retained.  Only local edge spikes, rapid
centre-line oscillation, and rapid ribbon-width changes are regularised.
Step 3 remains the sole owner of the world +Z stand-off height.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

# Matplotlib이 읽기 전용 사용자 설정 폴더에 cache를 만들지 않도록 Step2 전용 임시
# cache를 지정한다. 현재 작업은 2D plot만 사용하므로 환경의 Axes3D 경고도 숨긴다.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "step2_accuracy_matplotlib")
)
warnings.filterwarnings(
    "ignore", message="Unable to import Axes3D.*", category=UserWarning
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


from ..defaults import SOURCE_OUTPUT_ROOT

# =============================================================================
# USER INPUT / OUTPUT SETTINGS
# Step 2 결과를 읽어 Step 2-1 보정을 저장하는 연결 지점이다.
# 다른 실험 폴더를 쓰려면 INPUT_DIR/OUTPUT_DIR 또는 개별 이름만 바꾸면 된다.
# =============================================================================
INPUT_DIR = SOURCE_OUTPUT_ROOT / "step2"
INPUT_PATH_CSV_NAME = "step2_global_path.csv"
INPUT_THIN_FACE_PCD_NAME = "step2_thin_bonding_face.pcd"

OUTPUT_DIR = INPUT_DIR
OUTPUT_PATH_CSV_NAME = "step2_1_accurate_global_path.csv"
OUTPUT_PATH_PCD_NAME = "step2_1_accurate_global_path.pcd"
OUTPUT_ARCHIVE_NPZ_NAME = "step2_1_accurate_global_path.npz"
OUTPUT_METRICS_JSON_NAME = "step2_1_accuracy_metrics.json"
OUTPUT_PREVIEW_PNG_NAME = "step2_1_accuracy_preview.png"
OUTPUT_FINAL_PREVIEW_PNG_NAME = "step2_final_path_with_surface.png"

DEFAULT_INPUT = INPUT_DIR / INPUT_PATH_CSV_NAME
DEFAULT_THIN_FACE = INPUT_DIR / INPUT_THIN_FACE_PCD_NAME
DEFAULT_OUTPUT = OUTPUT_DIR


@dataclass(frozen=True)
class AccuracyConfig:
    """Physical smoothing parameters in millimetres."""

    median_window_mm: float = 1.5
    lateral_smooth_window_mm: float = 7.0
    height_smooth_window_mm: float = 3.5
    width_smooth_window_mm: float = 9.0
    strip_direction_smooth_window_mm: float = 7.0
    max_center_correction_mm: float = 0.30
    outlier_sigma: float = 3.5
    polynomial_order: int = 3
    face_section_x_radius_mm: float = 0.08
    face_section_z_radius_mm: float = 0.08
    face_section_y_radius_mm: float = 2.0
    face_section_lower_percentile: float = 2.0
    face_section_upper_percentile: float = 98.0
    face_recentering_smooth_window_mm: float = 7.0
    face_recentering_max_mm: float = 0.35


REQUIRED_COLUMNS = (
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
)


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def odd_window(window_mm: float, spacing_mm: float, count: int, order: int) -> int:
    requested = max(order + 2, int(round(window_mm / max(spacing_mm, 1.0e-9))))
    if requested % 2 == 0:
        requested += 1
    maximum = count if count % 2 == 1 else count - 1
    value = min(requested, maximum)
    if value <= order:
        raise ValueError(f"Not enough samples for polynomial order {order}: {count}")
    return value


def read_step2_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 9:
        raise ValueError(f"Step 2 input needs at least 9 points: {path}")
    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"Missing Step 2 columns: {missing}")

    def xyz(prefix: str) -> np.ndarray:
        return np.asarray(
            [[float(row[f"{prefix}_{axis}_mm"]) for axis in "xyz"] for row in rows],
            dtype=float,
        )

    result: Dict[str, np.ndarray] = {
        "global_path_mm": xyz("global"),
        "surface_center_mm": xyz("surface_center"),
        "outer_edge_mm": xyz("outer"),
        "inner_edge_mm": xyz("inner"),
    }
    for name in ("surface_to_thin_face_pcd_mm", "z_offset_mm"):
        if name in rows[0]:
            result[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
    return result


def read_binary_xyz_pcd(path: Path) -> np.ndarray:
    """Read the binary XYZ PCD written by Step 2."""

    with path.open("rb") as stream:
        fields: Sequence[str] = ()
        count = 0
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD has no DATA line: {path}")
            words = line.decode("ascii").strip().split()
            if not words:
                continue
            if words[0] in ("FIELDS", "FIELD"):
                fields = words[1:]
            elif words[0] == "POINTS":
                count = int(words[1])
            elif words[0] == "DATA":
                if words[1].lower() != "binary":
                    raise ValueError(f"Expected binary PCD: {path}")
                break
        if tuple(fields) != ("x", "y", "z") or count <= 0:
            raise ValueError(f"Expected XYZ-only PCD with POINTS: {path}")
        values = np.fromfile(stream, dtype="<f4", count=count * 3)
    if values.size != count * 3:
        raise ValueError(f"Truncated PCD payload: {path}")
    return values.reshape(count, 3).astype(float)


def robust_lowpass(
    values: np.ndarray,
    *,
    median_window: int,
    smooth_window: int,
    polyorder: int,
    outlier_sigma: float,
) -> np.ndarray:
    """Median-despike then Savitzky-Golay low-pass without phase delay."""

    values = np.asarray(values, dtype=float)
    baseline = median_filter(values, size=median_window, mode="nearest")
    residual = values - baseline
    centre = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - centre)))
    if sigma > 1.0e-12:
        limit = outlier_sigma * sigma
        despiked = baseline + np.clip(residual, centre - limit, centre + limit)
    else:
        despiked = baseline
    return savgol_filter(despiked, smooth_window, polyorder, mode="interp")


def normalize_rows(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    bad = norm[:, 0] < 1.0e-12
    result = values / np.maximum(norm, 1.0e-12)
    result[bad] = fallback[bad]
    return result


def limit_vector_correction(
    raw: np.ndarray, corrected: np.ndarray, maximum_mm: float
) -> np.ndarray:
    delta = corrected - raw
    length = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, maximum_mm / np.maximum(length, 1.0e-12))
    return raw + delta * scale[:, None]


def measure_face_sections(
    centre: np.ndarray,
    thin_face_points: np.ndarray,
    config: AccuracyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure the actual thin-face Y centre/width around every path point."""

    measured_center = np.full(len(centre), np.nan, dtype=float)
    measured_width = np.full(len(centre), np.nan, dtype=float)
    sample_count = np.zeros(len(centre), dtype=int)
    for index, point in enumerate(centre):
        selected = thin_face_points[
            (np.abs(thin_face_points[:, 0] - point[0]) <= config.face_section_x_radius_mm)
            & (np.abs(thin_face_points[:, 2] - point[2]) <= config.face_section_z_radius_mm)
            & (np.abs(thin_face_points[:, 1] - point[1]) <= config.face_section_y_radius_mm),
            1,
        ]
        sample_count[index] = len(selected)
        if len(selected) < 10:
            continue
        lower, upper = np.percentile(
            selected,
            [config.face_section_lower_percentile, config.face_section_upper_percentile],
        )
        measured_center[index] = 0.5 * (lower + upper)
        measured_width[index] = upper - lower

    valid = np.isfinite(measured_center) & np.isfinite(measured_width)
    if np.count_nonzero(valid) < max(9, int(0.8 * len(centre))):
        raise ValueError(
            "Thin-face PCD cross sections were valid at fewer than 80% of path points"
        )
    indices = np.arange(len(centre), dtype=float)
    measured_center = np.interp(indices, indices[valid], measured_center[valid])
    measured_width = np.interp(indices, indices[valid], measured_width[valid])
    return measured_center, measured_width, sample_count


def regularise_path(
    source: Mapping[str, np.ndarray],
    config: AccuracyConfig,
    thin_face_points: np.ndarray,
) -> Dict[str, np.ndarray]:
    raw_center = np.asarray(source["surface_center_mm"], dtype=float)
    raw_outer = np.asarray(source["outer_edge_mm"], dtype=float)
    raw_inner = np.asarray(source["inner_edge_mm"], dtype=float)
    distance = cumulative_distance(raw_center)
    spacing = float(np.median(np.diff(distance)))
    count = len(raw_center)

    median_window = odd_window(config.median_window_mm, spacing, count, order=0)
    lateral_window = odd_window(
        config.lateral_smooth_window_mm, spacing, count, config.polynomial_order
    )
    height_window = odd_window(
        config.height_smooth_window_mm, spacing, count, config.polynomial_order
    )
    width_window = odd_window(
        config.width_smooth_window_mm, spacing, count, config.polynomial_order
    )
    direction_window = odd_window(
        config.strip_direction_smooth_window_mm,
        spacing,
        count,
        config.polynomial_order,
    )
    recenter_window = odd_window(
        config.face_recentering_smooth_window_mm,
        spacing,
        count,
        config.polynomial_order,
    )

    corrected_center = raw_center.copy()
    # Keep monotonic X and the trimmed start/end positions exactly.  The top
    # view wave is lateral Y; Z gets a shorter window to retain real slopes.
    corrected_center[:, 1] = robust_lowpass(
        raw_center[:, 1],
        median_window=median_window,
        smooth_window=lateral_window,
        polyorder=config.polynomial_order,
        outlier_sigma=config.outlier_sigma,
    )
    corrected_center[:, 2] = robust_lowpass(
        raw_center[:, 2],
        median_window=median_window,
        smooth_window=height_window,
        polyorder=config.polynomial_order,
        outlier_sigma=config.outlier_sigma,
    )
    corrected_center = limit_vector_correction(
        raw_center, corrected_center, config.max_center_correction_mm
    )

    measured_face_center_y, measured_face_width, face_section_count = measure_face_sections(
        corrected_center, thin_face_points, config
    )
    recenter_delta_y = measured_face_center_y - corrected_center[:, 1]
    recenter_delta_y = robust_lowpass(
        recenter_delta_y,
        median_window=median_window,
        smooth_window=recenter_window,
        polyorder=config.polynomial_order,
        outlier_sigma=config.outlier_sigma,
    )
    recenter_delta_y = np.clip(
        recenter_delta_y,
        -config.face_recentering_max_mm,
        config.face_recentering_max_mm,
    )
    corrected_center[:, 1] += recenter_delta_y
    corrected_center = limit_vector_correction(
        raw_center, corrected_center, config.max_center_correction_mm
    )

    raw_strip = raw_outer - raw_inner
    raw_width = np.linalg.norm(raw_strip, axis=1)
    raw_direction = normalize_rows(raw_strip, np.tile([0.0, 1.0, 0.0], (count, 1)))
    # Use the PCD cross-section width, not the wider search-boundary spacing.
    corrected_width = robust_lowpass(
        measured_face_width,
        median_window=median_window,
        smooth_window=width_window,
        polyorder=config.polynomial_order,
        outlier_sigma=config.outlier_sigma,
    )
    corrected_width = np.maximum(corrected_width, 0.05)

    direction_components = np.column_stack(
        [
            robust_lowpass(
                raw_direction[:, axis],
                median_window=median_window,
                smooth_window=direction_window,
                polyorder=config.polynomial_order,
                outlier_sigma=config.outlier_sigma,
            )
            for axis in range(3)
        ]
    )
    corrected_direction = normalize_rows(direction_components, raw_direction)
    corrected_outer = corrected_center + 0.5 * corrected_width[:, None] * corrected_direction
    corrected_inner = corrected_center - 0.5 * corrected_width[:, None] * corrected_direction

    corrected_distance = cumulative_distance(corrected_center)
    tangents = np.gradient(corrected_center, corrected_distance, axis=0, edge_order=2)
    tangents = normalize_rows(tangents, np.tile([1.0, 0.0, 0.0], (count, 1)))
    correction = np.linalg.norm(corrected_center - raw_center, axis=1)

    return {
        "raw_global_path_mm": np.asarray(source["global_path_mm"], dtype=float),
        "global_path_mm": corrected_center,
        "surface_center_mm": corrected_center,
        "outer_edge_mm": corrected_outer,
        "inner_edge_mm": corrected_inner,
        "raw_outer_edge_mm": raw_outer,
        "raw_inner_edge_mm": raw_inner,
        "distance_mm": corrected_distance,
        "tangents": tangents,
        "face_width_mm": corrected_width,
        "raw_face_width_mm": raw_width,
        "center_correction_mm": correction,
        "face_center_before_error_mm": measured_face_center_y - raw_center[:, 1],
        "face_center_after_error_mm": measured_face_center_y - corrected_center[:, 1],
        "face_recentering_delta_y_mm": recenter_delta_y,
        "face_section_sample_count": face_section_count,
        "surface_to_thin_face_pcd_mm": np.asarray(
            source.get("surface_to_thin_face_pcd_mm", np.full(count, np.nan)),
            dtype=float,
        ),
        "z_offset_mm": np.zeros(count, dtype=float),
        "windows_samples": np.asarray(
            [
                median_window,
                lateral_window,
                height_window,
                width_window,
                direction_window,
                recenter_window,
            ],
            dtype=int,
        ),
    }


def heading_total_variation(points: np.ndarray) -> float:
    delta = np.diff(points[:, :2], axis=0)
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    if len(heading) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(heading))))


def high_frequency_rms(values: np.ndarray, window: int) -> float:
    trend = savgol_filter(values, window, 3, mode="interp")
    return float(np.sqrt(np.mean((values - trend) ** 2)))


def build_metrics(
    input_path: Path, config: AccuracyConfig, result: Mapping[str, np.ndarray]
) -> Dict[str, object]:
    raw = result["raw_global_path_mm"]
    corrected = result["global_path_mm"]
    windows = result["windows_samples"]
    correction = result["center_correction_mm"]
    raw_width = result["raw_face_width_mm"]
    width = result["face_width_mm"]
    roughness_window = int(windows[1])
    return {
        "input_csv": str(input_path.resolve()),
        "config": asdict(config),
        "point_count": int(len(corrected)),
        "path_length_mm": float(result["distance_mm"][-1]),
        "windows_samples": {
            "median": int(windows[0]),
            "lateral": int(windows[1]),
            "height": int(windows[2]),
            "width": int(windows[3]),
            "strip_direction": int(windows[4]),
            "face_recentering": int(windows[5]),
        },
        "centre_correction_mm": {
            "mean": float(np.mean(correction)),
            "p95": float(np.percentile(correction, 95.0)),
            "max": float(np.max(correction)),
        },
        "pcd_face_centre_error_mm": {
            "before_mean_signed": float(np.mean(result["face_center_before_error_mm"])),
            "before_p95_abs": float(
                np.percentile(np.abs(result["face_center_before_error_mm"]), 95.0)
            ),
            "after_mean_signed": float(np.mean(result["face_center_after_error_mm"])),
            "after_p95_abs": float(
                np.percentile(np.abs(result["face_center_after_error_mm"]), 95.0)
            ),
        },
        "lateral_high_frequency_rms_mm": {
            "raw": high_frequency_rms(raw[:, 1], roughness_window),
            "corrected": high_frequency_rms(corrected[:, 1], roughness_window),
        },
        "xy_heading_total_variation_rad": {
            "raw": heading_total_variation(raw),
            "corrected": heading_total_variation(corrected),
        },
        "face_width_mm": {
            "raw_min": float(np.min(raw_width)),
            "raw_max": float(np.max(raw_width)),
            "raw_std": float(np.std(raw_width)),
            "corrected_min": float(np.min(width)),
            "corrected_max": float(np.max(width)),
            "corrected_std": float(np.std(width)),
        },
        "step2_1_z_offset_mm": 0.0,
    }


def write_csv(path: Path, result: Mapping[str, np.ndarray]) -> None:
    names = (
        "index", "path_distance_mm",
        "global_x_mm", "global_y_mm", "global_z_mm",
        "surface_center_x_mm", "surface_center_y_mm", "surface_center_z_mm",
        "outer_x_mm", "outer_y_mm", "outer_z_mm",
        "inner_x_mm", "inner_y_mm", "inner_z_mm",
        "face_width_mm", "surface_to_thin_face_pcd_mm", "z_offset_mm",
        "tangent_x", "tangent_y", "tangent_z",
        "raw_global_x_mm", "raw_global_y_mm", "raw_global_z_mm",
        "center_correction_mm", "face_recentering_delta_y_mm",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for index in range(len(result["global_path_mm"])):
            writer.writerow(
                [
                    index,
                    f"{result['distance_mm'][index]:.9f}",
                    *[f"{v:.9f}" for v in result["global_path_mm"][index]],
                    *[f"{v:.9f}" for v in result["surface_center_mm"][index]],
                    *[f"{v:.9f}" for v in result["outer_edge_mm"][index]],
                    *[f"{v:.9f}" for v in result["inner_edge_mm"][index]],
                    f"{result['face_width_mm'][index]:.9f}",
                    f"{result['surface_to_thin_face_pcd_mm'][index]:.9f}",
                    "0.000000000",
                    *[f"{v:.9f}" for v in result["tangents"][index]],
                    *[f"{v:.9f}" for v in result["raw_global_path_mm"][index]],
                    f"{result['center_correction_mm'][index]:.9f}",
                    f"{result['face_recentering_delta_y_mm'][index]:.9f}",
                ]
            )


def write_binary_pcd(path: Path, points: np.ndarray) -> None:
    values = np.asarray(points, dtype="<f4")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(values)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(values)}\nDATA binary\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        values.tofile(stream)


def write_preview(
    path: Path,
    result: Mapping[str, np.ndarray],
    thin_face_points: np.ndarray,
) -> None:
    """실제 얇은 면 PCD 위에 raw/보정 경로를 겹쳐 최종 결과를 PNG로 저장한다."""

    raw = result["raw_global_path_mm"]
    corrected = result["global_path_mm"]
    raw_outer = result["raw_outer_edge_mm"]
    raw_inner = result["raw_inner_edge_mm"]
    outer = result["outer_edge_mm"]
    inner = result["inner_edge_mm"]
    distance = result["distance_mm"]

    # 배경 PCD가 너무 조밀하면 그림 생성이 느리고 경로가 가려질 수 있으므로
    # 최대 약 80,000점만 균일하게 표시한다. 계산에는 전체 PCD를 사용한다.
    stride = max(1, int(np.ceil(len(thin_face_points) / 80_000)))
    face = np.asarray(thin_face_points[::stride], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    ax = axes[0, 0]
    face_plot = ax.scatter(
        face[:, 0],
        face[:, 1],
        c=face[:, 2],
        cmap="turbo",
        s=0.8,
        alpha=0.28,
        label="thin-face PCD (color=Z)",
        zorder=0,
    )
    ax.plot(raw_outer[:, 0], raw_outer[:, 1], color="red", alpha=0.25, lw=1.0, label="raw outer")
    ax.plot(raw_inner[:, 0], raw_inner[:, 1], color="blue", alpha=0.25, lw=1.0, label="raw inner")
    ax.plot(outer[:, 0], outer[:, 1], color="red", lw=1.8, label="corrected outer")
    ax.plot(inner[:, 0], inner[:, 1], color="blue", lw=1.8, label="corrected inner")
    ax.plot(raw[:, 0], raw[:, 1], color="gray", ls="--", lw=1.0, label="raw centre")
    ax.plot(
        corrected[:, 0], corrected[:, 1], color="#00a83b", lw=2.6,
        label="final corrected Step 2 path", zorder=5,
    )
    ax.set_title("Top view XY: thin bonding face and final corrected path")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    colorbar = figure.colorbar(face_plot, ax=ax, pad=0.01)
    colorbar.set_label("Thin-face local Z [mm]")

    ax = axes[0, 1]
    ax.scatter(
        face[:, 0], face[:, 2], s=0.6, color="0.65", alpha=0.16,
        label="thin-face PCD", zorder=0,
    )
    ax.plot(raw[:, 0], raw[:, 2], color="gray", ls="--", lw=1.2, label="raw centre Z")
    ax.plot(
        corrected[:, 0], corrected[:, 2], color="#00a83b", lw=2.4,
        label="final corrected path Z", zorder=5,
    )
    ax.set_title("Side profile XZ: PCD height and retained steep shape")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(distance, (corrected[:, 1] - raw[:, 1]) * 1000.0, color="purple", label="Y correction")
    ax.plot(distance, (corrected[:, 2] - raw[:, 2]) * 1000.0, color="orange", label="Z correction")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_title("Applied centre-line correction")
    ax.set_xlabel("Path distance [mm]")
    ax.set_ylabel("Correction [micrometre]")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(distance, result["raw_face_width_mm"], color="gray", alpha=0.55, label="raw width")
    ax.plot(distance, result["face_width_mm"], color="teal", lw=2.0, label="regularised width")
    ax.set_title("Ribbon width: rapid changes removed")
    ax.set_xlabel("Path distance [mm]")
    ax.set_ylabel("Width [mm]")
    ax.grid(alpha=0.25)
    ax.legend()
    figure.suptitle(
        "Step 2 final result — PCD thin face, raw path, and corrected Step 3 input",
        fontsize=16,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_final_path_preview(
    path: Path,
    result: Mapping[str, np.ndarray],
    thin_face_points: np.ndarray,
) -> None:
    """최종 결과를 '높이 색상 면 + 경사각 색상 경로' 고정 형식으로 출력한다.

    위 패널은 XY에서 실제 얇은 면의 local Z를 색으로 나타내고 보정된 outer/inner와
    최종 중앙 경로를 겹친다. 아래 패널은 XZ에서 중앙 경로의 구간 경사각을 색으로
    나타낸다. 이 함수의 레이아웃과 파일명은 Step2 표준 결과 화면으로 유지한다.
    """

    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    outer = np.asarray(result["outer_edge_mm"], dtype=float)
    inner = np.asarray(result["inner_edge_mm"], dtype=float)
    centre = np.asarray(result["global_path_mm"], dtype=float)
    stride = max(1, int(np.ceil(len(thin_face_points) / 180_000)))
    face = np.asarray(thin_face_points[::stride], dtype=float)

    figure, (height_axis, slope_axis) = plt.subplots(
        2,
        1,
        figsize=(16, 8.5),
        gridspec_kw={"height_ratios": (1.15, 1.0)},
    )

    # 1) 위에서 본 실제 얇은 면: 점 색상은 local Z 높이를 의미한다.
    # 보정 경계는 2~98 percentile과 평활화를 사용하므로 원본 면의 일부 점은 경계
    # 밖에 남는다. 이 점들을 회색으로 분리해 '원본 전체 면'과 '경로 계산에 사용한
    # robust 유효 면'을 혼동하지 않도록 한다.
    face_outer_y = np.interp(face[:, 0], outer[:, 0], outer[:, 1])
    face_inner_y = np.interp(face[:, 0], inner[:, 0], inner[:, 1])
    face_lower_y = np.minimum(face_outer_y, face_inner_y)
    face_upper_y = np.maximum(face_outer_y, face_inner_y)
    inside_x = (face[:, 0] >= outer[:, 0].min()) & (
        face[:, 0] <= outer[:, 0].max()
    )
    inside_robust_face = (
        inside_x
        & (face[:, 1] >= face_lower_y)
        & (face[:, 1] <= face_upper_y)
    )

    height_norm = Normalize(
        vmin=float(np.min(face[:, 2])),
        vmax=float(np.max(face[:, 2])),
    )
    height_axis.scatter(
        face[~inside_robust_face, 0],
        face[~inside_robust_face, 1],
        color="0.65",
        s=0.9,
        alpha=0.22,
        label="PCD points outside robust boundaries",
        zorder=0,
    )
    height_points = height_axis.scatter(
        face[inside_robust_face, 0],
        face[inside_robust_face, 1],
        c=face[inside_robust_face, 2],
        cmap="turbo",
        norm=height_norm,
        s=1.2,
        alpha=0.82,
        label="PCD-derived raised top face",
        zorder=1,
    )
    height_axis.plot(
        outer[:, 0], outer[:, 1], color="black", linewidth=0.8,
        linestyle="--", alpha=0.9,
        label="Corrected outer edge", zorder=3,
    )
    height_axis.plot(
        inner[:, 0], inner[:, 1], color="#404040", linewidth=0.8,
        linestyle=":", alpha=0.9,
        label="Corrected groove-side inner edge", zorder=3,
    )
    height_axis.plot(
        centre[:, 0], centre[:, 1], color="white", linewidth=1.4,
        label="Final corrected path", zorder=4,
    )
    height_axis.set(
        title="PCD-derived thin bonding face — color represents local Z height",
        xlabel="X [mm]",
        ylabel="Y [mm]",
    )
    height_axis.set_aspect("equal", adjustable="box")
    height_axis.grid(alpha=0.22)
    height_axis.legend(loc="lower center", ncol=3, fontsize=9)
    height_colorbar = figure.colorbar(height_points, ax=height_axis, pad=0.015)
    height_colorbar.set_label("Surface local Z [mm]")

    # 2) 옆에서 본 최종 경로: 각 선분 색상은 XY 평면 기준 경사각이다.
    xy_step = np.linalg.norm(np.diff(centre[:, :2], axis=0), axis=1)
    z_step = np.abs(np.diff(centre[:, 2]))
    slope_deg = np.degrees(
        np.arctan2(z_step, np.maximum(xy_step, 1.0e-12))
    )
    path_segments = np.stack(
        (centre[:-1, [0, 2]], centre[1:, [0, 2]]), axis=1
    )
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
        title="Final path steepness — color represents slope angle from the XY plane",
        xlabel="X [mm]",
        ylabel="Surface local Z [mm]",
    )
    slope_axis.grid(alpha=0.22)
    slope_colorbar = figure.colorbar(slope_lines, ax=slope_axis, pad=0.015)
    slope_colorbar.set_label("Slope angle [deg]")

    figure.suptitle(
        "Step 2 final thin-face height and path steepness",
        fontsize=17,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--thin-face-pcd", type=Path, default=DEFAULT_THIN_FACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--median-window-mm", type=float, default=1.5)
    parser.add_argument("--lateral-window-mm", type=float, default=7.0)
    parser.add_argument("--height-window-mm", type=float, default=3.5)
    parser.add_argument("--width-window-mm", type=float, default=9.0)
    parser.add_argument("--direction-window-mm", type=float, default=7.0)
    parser.add_argument("--max-correction-mm", type=float, default=0.30)
    parser.add_argument("--outlier-sigma", type=float, default=3.5)
    parser.add_argument("--face-recentering-max-mm", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AccuracyConfig(
        median_window_mm=args.median_window_mm,
        lateral_smooth_window_mm=args.lateral_window_mm,
        height_smooth_window_mm=args.height_window_mm,
        width_smooth_window_mm=args.width_window_mm,
        strip_direction_smooth_window_mm=args.direction_window_mm,
        max_center_correction_mm=args.max_correction_mm,
        outlier_sigma=args.outlier_sigma,
        face_recentering_max_mm=args.face_recentering_max_mm,
    )
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = read_step2_csv(input_path)
    thin_face_path = args.thin_face_pcd.expanduser().resolve()
    thin_face_points = read_binary_xyz_pcd(thin_face_path)
    result = regularise_path(source, config, thin_face_points)
    metrics = build_metrics(input_path, config, result)

    csv_path = output_dir / OUTPUT_PATH_CSV_NAME
    pcd_path = output_dir / OUTPUT_PATH_PCD_NAME
    archive_path = output_dir / OUTPUT_ARCHIVE_NPZ_NAME
    metrics_path = output_dir / OUTPUT_METRICS_JSON_NAME
    preview_path = output_dir / OUTPUT_PREVIEW_PNG_NAME
    final_preview_path = output_dir / OUTPUT_FINAL_PREVIEW_PNG_NAME
    write_csv(csv_path, result)
    write_binary_pcd(pcd_path, result["global_path_mm"])
    np.savez_compressed(archive_path, **result)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    # 진단용 4분할 그림과 사용자 확인용 고정 형식 최종 그림을 별도로 저장한다.
    write_preview(preview_path, result, thin_face_points)
    write_final_path_preview(final_preview_path, result, thin_face_points)

    roughness = metrics["lateral_high_frequency_rms_mm"]
    heading = metrics["xy_heading_total_variation_rad"]
    correction = metrics["centre_correction_mm"]
    print(f"Input: {input_path}")
    print(f"Output points: {metrics['point_count']}, length={metrics['path_length_mm']:.6f} mm")
    print(
        "Lateral high-frequency RMS: "
        f"{roughness['raw']:.6f} -> {roughness['corrected']:.6f} mm"
    )
    print(
        "XY heading total variation: "
        f"{heading['raw']:.6f} -> {heading['corrected']:.6f} rad"
    )
    print(
        "Centre correction mean/p95/max: "
        f"{correction['mean']:.6f}/{correction['p95']:.6f}/{correction['max']:.6f} mm"
    )
    for output in (
        csv_path,
        pcd_path,
        archive_path,
        metrics_path,
        preview_path,
        final_preview_path,
    ):
        print(f"  {output}")


if __name__ == "__main__":
    main()
