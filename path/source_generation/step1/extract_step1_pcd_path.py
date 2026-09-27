#!/usr/bin/env python3
"""Extract the Step 1 top bonding path from a large binary PCD file.

The reference image identifies Step 1 semantically.  Coordinates are obtained
only from the point cloud: in the model-local front view, X is horizontal and
+Y is upward.  The desired line is the lower-Z edge of the upper-Y envelope.

The input cloud contains about 92.7 million points, so this program uses a
NumPy memory map and three bounded-memory passes instead of loading the entire
cloud into Open3D.
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
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter


from ..defaults import SOURCE_PCD, SOURCE_OUTPUT_ROOT

# =============================================================================
# USER INPUT / OUTPUT SETTINGS
# 아래 값만 바꾸면 이 파일이 읽을 PCD와 결과 저장 위치/이름을 변경할 수 있다.
# 명령행의 --pcd, --output-dir 옵션을 주면 실행할 때만 이 값을 덮어쓴다.
# =============================================================================
INPUT_PCD = SOURCE_PCD
OUTPUT_DIR = SOURCE_OUTPUT_ROOT / "step1"

OUTPUT_RAW_CSV_NAME = "step1_pcd_path_raw.csv"
OUTPUT_PATH_CSV_NAME = "step1_pcd_path.csv"
OUTPUT_PATH_NPZ_NAME = "step1_pcd_path.npz"
OUTPUT_METADATA_JSON_NAME = "step1_pcd_path_metadata.json"
OUTPUT_PREVIEW_PNG_NAME = "step1_pcd_path_preview.png"

# 다른 Step 1 코드가 가져다 쓰는 이전 상수명은 호환을 위해 유지한다.
DEFAULT_PCD = INPUT_PCD
DEFAULT_OUTPUT_DIR = OUTPUT_DIR


@dataclass(frozen=True)
class ExtractionConfig:
    """Geometry and numerical parameters, all expressed in millimetres."""

    x_min_mm: float = -65.53
    x_max_mm: float = 65.53
    y_min_mm: float = 12.0
    z_min_mm: float = 9.0
    z_max_mm: float = 18.0
    bin_width_mm: float = 0.02
    top_band_mm: float = 0.03
    z_refine_band_mm: float = 0.03
    smooth_window_mm: float = 0.75
    output_spacing_mm: float = 0.5
    chunk_points: int = 5_000_000
    min_valid_fraction: float = 0.95


@dataclass(frozen=True)
class PcdHeader:
    """Information needed to memory-map an uncompressed binary PCD file."""

    fields: Tuple[str, ...]
    sizes: Tuple[int, ...]
    types: Tuple[str, ...]
    counts: Tuple[int, ...]
    width: int
    height: int
    points: int
    data: str
    data_offset: int


def _pcd_scalar_dtype(type_code: str, size: int) -> np.dtype:
    mapping = {
        ("F", 4): np.dtype("<f4"),
        ("F", 8): np.dtype("<f8"),
        ("I", 1): np.dtype("i1"),
        ("I", 2): np.dtype("<i2"),
        ("I", 4): np.dtype("<i4"),
        ("I", 8): np.dtype("<i8"),
        ("U", 1): np.dtype("u1"),
        ("U", 2): np.dtype("<u2"),
        ("U", 4): np.dtype("<u4"),
        ("U", 8): np.dtype("<u8"),
    }
    try:
        return mapping[(type_code.upper(), int(size))]
    except KeyError as exc:
        raise ValueError(f"unsupported PCD scalar TYPE/SIZE: {type_code}/{size}") from exc


def read_pcd_header(path: Path) -> PcdHeader:
    """Read a PCD header without decoding any following binary payload."""

    values: Dict[str, List[str]] = {}
    data_offset = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA line: {path}")
            data_offset += len(line)
            stripped = line.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            try:
                tokens = stripped.decode("ascii").split()
            except UnicodeDecodeError as exc:
                raise ValueError("binary payload started before a valid DATA line") from exc
            values[tokens[0].upper()] = tokens[1:]
            if tokens[0].upper() == "DATA":
                break

    required = {"FIELDS", "SIZE", "TYPE", "WIDTH", "HEIGHT", "POINTS", "DATA"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"missing PCD header entries: {sorted(missing)}")

    fields = tuple(values["FIELDS"])
    sizes = tuple(int(value) for value in values["SIZE"])
    types = tuple(value.upper() for value in values["TYPE"])
    counts = tuple(int(value) for value in values.get("COUNT", ["1"] * len(fields)))
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("FIELDS/SIZE/TYPE/COUNT lengths do not match")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PCD must contain x, y, z fields; found {fields}")

    data = values["DATA"][0].lower()
    if data != "binary":
        raise ValueError(
            f"only uncompressed 'DATA binary' PCD is supported, found {data!r}"
        )

    return PcdHeader(
        fields=fields,
        sizes=sizes,
        types=types,
        counts=counts,
        width=int(values["WIDTH"][0]),
        height=int(values["HEIGHT"][0]),
        points=int(values["POINTS"][0]),
        data=data,
        data_offset=data_offset,
    )


def build_pcd_dtype(header: PcdHeader) -> np.dtype:
    """Build the exact structured record layout described by the PCD header."""

    descriptions = []
    for name, size, type_code, count in zip(
        header.fields, header.sizes, header.types, header.counts
    ):
        scalar = _pcd_scalar_dtype(type_code, size)
        descriptions.append((name, scalar) if count == 1 else (name, scalar, (count,)))
    return np.dtype(descriptions, align=False)


def open_binary_pcd(path: Path) -> Tuple[PcdHeader, np.memmap]:
    """Validate and memory-map point records without copying them into RAM."""

    path = path.expanduser().resolve()
    header = read_pcd_header(path)
    dtype = build_pcd_dtype(header)
    expected_size = header.data_offset + header.points * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(
            f"truncated PCD: expected at least {expected_size} bytes, got {actual_size}"
        )
    records = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=header.data_offset,
        shape=(header.points,),
    )
    return header, records


def iter_chunks(records: np.ndarray, chunk_points: int) -> Iterator[Tuple[int, np.ndarray]]:
    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")
    for start in range(0, len(records), chunk_points):
        yield start, records[start : start + chunk_points]


def validate_config(config: ExtractionConfig) -> None:
    if not config.x_min_mm < config.x_max_mm:
        raise ValueError("x_min_mm must be smaller than x_max_mm")
    if not config.z_min_mm < config.z_max_mm:
        raise ValueError("z_min_mm must be smaller than z_max_mm")
    for name in (
        "bin_width_mm",
        "top_band_mm",
        "z_refine_band_mm",
        "smooth_window_mm",
        "output_spacing_mm",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < config.min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in (0, 1]")


def make_bins(config: ExtractionConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return uniform bin edges and centres spanning the configured X range."""

    count = max(2, int(math.ceil((config.x_max_mm - config.x_min_mm) / config.bin_width_mm)))
    edges = np.linspace(config.x_min_mm, config.x_max_mm, count + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return edges, centres


def point_bin_indices(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, x, side="right") - 1
    indices[x == edges[-1]] = len(edges) - 2
    return indices


def roi_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: ExtractionConfig,
) -> np.ndarray:
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x >= config.x_min_mm)
        & (x <= config.x_max_mm)
        & (y >= config.y_min_mm)
        & (z >= config.z_min_mm)
        & (z <= config.z_max_mm)
    )


def print_pass_progress(pass_name: str, processed: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1.0e-6)
    percent = 100.0 * processed / total
    rate = processed / elapsed / 1.0e6
    print(f"  {pass_name}: {percent:5.1f}% ({rate:5.1f} M points/s)", flush=True)


def compute_upper_envelope(
    records: np.ndarray,
    edges: np.ndarray,
    config: ExtractionConfig,
) -> Tuple[np.ndarray, int]:
    """Pass 1: compute maximum local Y in every X slice inside the ROI."""

    max_y = np.full(len(edges) - 1, -np.inf, dtype=np.float64)
    roi_count = 0
    started = time.monotonic()
    for chunk_number, (start, chunk) in enumerate(
        iter_chunks(records, config.chunk_points), start=1
    ):
        x = np.asarray(chunk["x"])
        y = np.asarray(chunk["y"])
        z = np.asarray(chunk["z"])
        mask = roi_mask(x, y, z, config)
        if np.any(mask):
            selected_x = x[mask]
            selected_y = y[mask]
            bins = point_bin_indices(selected_x, edges)
            valid_bins = (bins >= 0) & (bins < len(max_y))
            np.maximum.at(max_y, bins[valid_bins], selected_y[valid_bins])
            roi_count += int(np.count_nonzero(valid_bins))
        if chunk_number % 4 == 0 or start + len(chunk) == len(records):
            print_pass_progress("pass 1/3 upper Y", start + len(chunk), len(records), started)
    return max_y, roi_count


def compute_lower_z_edge(
    records: np.ndarray,
    edges: np.ndarray,
    max_y: np.ndarray,
    config: ExtractionConfig,
) -> Tuple[np.ndarray, int]:
    """Pass 2: find the minimum Z inside each slice's upper-Y band."""

    min_z = np.full(len(max_y), np.inf, dtype=np.float64)
    candidate_count = 0
    started = time.monotonic()
    for chunk_number, (start, chunk) in enumerate(
        iter_chunks(records, config.chunk_points), start=1
    ):
        x = np.asarray(chunk["x"])
        y = np.asarray(chunk["y"])
        z = np.asarray(chunk["z"])
        mask = roi_mask(x, y, z, config)
        if np.any(mask):
            selected_indices = np.flatnonzero(mask)
            bins = point_bin_indices(x[selected_indices], edges)
            valid = (bins >= 0) & (bins < len(max_y)) & np.isfinite(max_y[bins])
            selected_indices = selected_indices[valid]
            bins = bins[valid]
            top = y[selected_indices] >= max_y[bins] - config.top_band_mm
            selected_indices = selected_indices[top]
            bins = bins[top]
            np.minimum.at(min_z, bins, z[selected_indices])
            candidate_count += len(selected_indices)
        if chunk_number % 4 == 0 or start + len(chunk) == len(records):
            print_pass_progress("pass 2/3 lower Z", start + len(chunk), len(records), started)
    return min_z, candidate_count


def select_actual_edge_points(
    records: np.ndarray,
    edges: np.ndarray,
    centres: np.ndarray,
    max_y: np.ndarray,
    min_z: np.ndarray,
    config: ExtractionConfig,
) -> np.ndarray:
    """Pass 3: choose one actual PCD sample per slice near both envelopes."""

    point_count = len(max_y)
    best_score = np.full(point_count, np.inf, dtype=np.float64)
    best_points = np.full((point_count, 3), np.nan, dtype=np.float64)
    half_bin = max(0.5 * float(edges[1] - edges[0]), 1.0e-9)
    started = time.monotonic()

    for chunk_number, (start, chunk) in enumerate(
        iter_chunks(records, config.chunk_points), start=1
    ):
        x = np.asarray(chunk["x"])
        y = np.asarray(chunk["y"])
        z = np.asarray(chunk["z"])
        mask = roi_mask(x, y, z, config)
        if np.any(mask):
            selected_indices = np.flatnonzero(mask)
            bins = point_bin_indices(x[selected_indices], edges)
            valid = (
                (bins >= 0)
                & (bins < point_count)
                & np.isfinite(max_y[bins])
                & np.isfinite(min_z[bins])
            )
            selected_indices = selected_indices[valid]
            bins = bins[valid]
            edge = (
                (y[selected_indices] >= max_y[bins] - config.top_band_mm)
                & (z[selected_indices] <= min_z[bins] + config.z_refine_band_mm)
            )
            selected_indices = selected_indices[edge]
            bins = bins[edge]

            if len(bins):
                # First favour an actual sample near the slice centre, then the
                # upper-Y/lower-Z intersection.  Every selected result remains
                # a point that exists verbatim in the source PCD.
                score = (
                    ((x[selected_indices] - centres[bins]) / half_bin) ** 2
                    + 0.25
                    * ((max_y[bins] - y[selected_indices]) / config.top_band_mm) ** 2
                    + 0.25
                    * ((z[selected_indices] - min_z[bins]) / config.z_refine_band_mm) ** 2
                )
                order = np.lexsort((score, bins))
                sorted_bins = bins[order]
                first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
                local_choice = order[first]
                chosen_bins = bins[local_choice]
                chosen_score = score[local_choice]
                improve = chosen_score < best_score[chosen_bins]
                chosen_bins = chosen_bins[improve]
                local_choice = local_choice[improve]
                best_score[chosen_bins] = score[local_choice]
                best_points[chosen_bins, 0] = x[selected_indices[local_choice]]
                best_points[chosen_bins, 1] = y[selected_indices[local_choice]]
                best_points[chosen_bins, 2] = z[selected_indices[local_choice]]

        if chunk_number % 4 == 0 or start + len(chunk) == len(records):
            print_pass_progress("pass 3/3 actual points", start + len(chunk), len(records), started)

    return best_points


def odd_savgol_window(window_mm: float, bin_width_mm: float, point_count: int) -> int:
    requested = max(5, int(round(window_mm / bin_width_mm)))
    if requested % 2 == 0:
        requested += 1
    maximum = point_count if point_count % 2 == 1 else point_count - 1
    return min(requested, maximum)


def smooth_envelope(
    actual_points: np.ndarray,
    centres: np.ndarray,
    config: ExtractionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill small missing slices and smooth Y/Z on the uniform X grid."""

    valid = np.all(np.isfinite(actual_points), axis=1)
    fraction = float(np.mean(valid))
    if fraction < config.min_valid_fraction:
        raise ValueError(
            f"only {fraction:.1%} of X slices produced candidates; expected at least "
            f"{config.min_valid_fraction:.1%}. Check the ROI and axis convention."
        )
    valid_indices = np.flatnonzero(valid)
    first, last = int(valid_indices[0]), int(valid_indices[-1])
    grid_indices = np.arange(first, last + 1)
    grid_x = centres[grid_indices]
    raw_x = centres[valid]
    interpolated_y = np.interp(grid_x, raw_x, actual_points[valid, 1])
    interpolated_z = np.interp(grid_x, raw_x, actual_points[valid, 2])

    effective_bin = float(np.mean(np.diff(centres)))
    window = odd_savgol_window(
        config.smooth_window_mm, effective_bin, len(grid_indices)
    )
    if window >= 5:
        smooth_y = savgol_filter(interpolated_y, window, polyorder=3, mode="interp")
        smooth_z = savgol_filter(interpolated_z, window, polyorder=3, mode="interp")
    else:
        smooth_y = interpolated_y
        smooth_z = interpolated_z

    # Preserve measured endpoints while smoothing only the interior.
    smooth_y[[0, -1]] = interpolated_y[[0, -1]]
    smooth_z[[0, -1]] = interpolated_z[[0, -1]]
    envelope = np.column_stack((grid_x, smooth_y, smooth_z))
    return envelope, valid


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("at least two points are required")
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def resample_by_arclength(
    points: np.ndarray, spacing_mm: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a 3D polyline uniformly and return path distance and tangents."""

    distance = cumulative_distance(points)
    targets = np.arange(0.0, distance[-1], spacing_mm)
    if not np.isclose(targets[-1], distance[-1]):
        targets = np.r_[targets, distance[-1]]
    sampled = np.column_stack(
        [np.interp(targets, distance, points[:, axis]) for axis in range(3)]
    )
    derivatives = np.gradient(sampled, targets, axis=0, edge_order=1)
    lengths = np.linalg.norm(derivatives, axis=1, keepdims=True)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("zero-length tangent encountered after resampling")
    tangents = derivatives / lengths
    return sampled, targets, tangents


def write_raw_csv(path: Path, points: np.ndarray, valid: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("bin_index", "source_x_mm", "source_y_mm", "source_z_mm"))
        for index in np.flatnonzero(valid):
            writer.writerow((index, *[f"{value:.9f}" for value in points[index]]))


def write_path_csv(
    path: Path,
    points: np.ndarray,
    distances: np.ndarray,
    tangents: np.ndarray,
) -> None:
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


def sample_preview_cloud(records: np.ndarray, maximum_points: int) -> np.ndarray:
    stride = max(1, int(math.ceil(len(records) / maximum_points)))
    sampled = records[::stride]
    points = np.column_stack((sampled["x"], sampled["y"], sampled["z"]))
    return np.asarray(points[np.all(np.isfinite(points), axis=1)], dtype=np.float64)


def write_preview(
    path: Path,
    cloud_sample: np.ndarray,
    envelope: np.ndarray,
    final_path: np.ndarray,
    config: ExtractionConfig,
) -> None:
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

    # Keep preview generation compatible with minimal Matplotlib installs.
    # Some ROS/Isaac environments contain mismatched mpl_toolkits packages, so
    # requiring the optional 3D projection would make an otherwise successful
    # extraction fail at the final diagnostic-only step.
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axis_xy, axis_xz, axis_profile = axes

    axis_xy.scatter(
        cloud_sample[:, 0], cloud_sample[:, 1], s=0.25, c="0.75", alpha=0.35
    )
    axis_xy.plot(envelope[:, 0], envelope[:, 1], color="tab:blue", linewidth=1.0)
    axis_xy.plot(final_path[:, 0], final_path[:, 1], color="red", linewidth=2.0)
    axis_xy.set(title="Front view: upper-Y Step 1 path", xlabel="local X [mm]", ylabel="local Y [mm]")
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.grid(alpha=0.25)

    axis_xz.scatter(
        cloud_sample[:, 0], cloud_sample[:, 2], s=0.25, c="0.75", alpha=0.35
    )
    axis_xz.plot(envelope[:, 0], envelope[:, 2], color="tab:blue", linewidth=1.0)
    axis_xz.plot(final_path[:, 0], final_path[:, 2], color="red", linewidth=2.0)
    axis_xz.set(title="Depth view: lower-Z edge", xlabel="local X [mm]", ylabel="local Z [mm]")
    axis_xz.grid(alpha=0.25)

    distance = cumulative_distance(final_path)
    axis_profile.plot(distance, final_path[:, 1], color="red", linewidth=2.0, label="Y")
    axis_profile.plot(
        distance, final_path[:, 2], color="tab:blue", linewidth=2.0, label="Z"
    )
    axis_profile.set(
        title=f"Final path profile ({config.output_spacing_mm:g} mm spacing)",
        xlabel="path distance [mm]",
        ylabel="coordinate [mm]",
    )
    axis_profile.grid(alpha=0.25)
    axis_profile.legend()

    figure.suptitle("Step 1 path extracted directly from AI Glass PCD")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_metadata(
    pcd_path: Path,
    header: PcdHeader,
    config: ExtractionConfig,
    roi_count: int,
    candidate_count: int,
    raw_valid: np.ndarray,
    final_points: np.ndarray,
    distances: np.ndarray,
) -> Mapping[str, object]:
    return {
        "input_pcd": str(pcd_path),
        "pcd": {
            "points": header.points,
            "fields": list(header.fields),
            "data": header.data,
            "data_offset_bytes": header.data_offset,
        },
        "method": "upper-Y envelope followed by lower-Z edge selection",
        "axis_convention": {
            "horizontal": "+X",
            "front_view_up": "+Y",
            "depth": "Z",
        },
        "config_mm": asdict(config),
        "statistics": {
            "roi_point_count": roi_count,
            "upper_band_candidate_count": candidate_count,
            "x_bin_count": int(len(raw_valid)),
            "valid_x_bin_count": int(np.count_nonzero(raw_valid)),
            "valid_x_bin_fraction": float(np.mean(raw_valid)),
            "output_waypoint_count": int(len(final_points)),
            "path_length_mm": float(distances[-1]),
            "path_bounds_mm": {
                "min": np.min(final_points, axis=0).tolist(),
                "max": np.max(final_points, axis=0).tolist(),
            },
        },
    }


def extract_path(
    pcd_path: Path,
    output_dir: Path,
    config: ExtractionConfig,
    *,
    make_preview: bool = True,
    preview_max_points: int = 100_000,
) -> Mapping[str, Path]:
    validate_config(config)
    pcd_path = pcd_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {pcd_path}")
    header, records = open_binary_pcd(pcd_path)
    print(
        f"PCD: {header.points:,} points, fields={header.fields}, "
        f"record={records.dtype.itemsize} bytes"
    )
    edges, centres = make_bins(config)
    print(f"X slices: {len(centres):,} (nominal width {config.bin_width_mm:g} mm)")

    max_y, roi_count = compute_upper_envelope(records, edges, config)
    min_z, candidate_count = compute_lower_z_edge(records, edges, max_y, config)
    raw_points = select_actual_edge_points(
        records, edges, centres, max_y, min_z, config
    )
    envelope, raw_valid = smooth_envelope(raw_points, centres, config)
    final_points, distances, tangents = resample_by_arclength(
        envelope, config.output_spacing_mm
    )

    raw_csv = output_dir / OUTPUT_RAW_CSV_NAME
    path_csv = output_dir / OUTPUT_PATH_CSV_NAME
    path_npz = output_dir / OUTPUT_PATH_NPZ_NAME
    metadata_json = output_dir / OUTPUT_METADATA_JSON_NAME
    preview_png = output_dir / OUTPUT_PREVIEW_PNG_NAME

    write_raw_csv(raw_csv, raw_points, raw_valid)
    write_path_csv(path_csv, final_points, distances, tangents)
    np.savez_compressed(
        path_npz,
        points_mm=final_points,
        distance_mm=distances,
        tangents=tangents,
        smoothed_envelope_mm=envelope,
        raw_source_points_mm=raw_points[raw_valid],
    )
    metadata = build_metadata(
        pcd_path,
        header,
        config,
        roi_count,
        candidate_count,
        raw_valid,
        final_points,
        distances,
    )
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    outputs = {
        "path_csv": path_csv,
        "path_npz": path_npz,
        "raw_csv": raw_csv,
        "metadata_json": metadata_json,
    }
    if make_preview:
        preview_cloud = sample_preview_cloud(records, preview_max_points)
        write_preview(preview_png, preview_cloud, envelope, final_points, config)
        outputs["preview_png"] = preview_png

    print(f"Path: {len(final_points)} waypoints, {distances[-1]:.3f} mm")
    print(
        "Bounds [mm]: "
        f"min={np.min(final_points, axis=0).round(4).tolist()}, "
        f"max={np.max(final_points, axis=0).round(4).tolist()}"
    )
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, default=DEFAULT_PCD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=-65.53)
    parser.add_argument("--x-max", type=float, default=65.53)
    parser.add_argument("--y-min", type=float, default=12.0)
    parser.add_argument("--z-min", type=float, default=9.0)
    parser.add_argument("--z-max", type=float, default=18.0)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--top-band", type=float, default=0.03)
    parser.add_argument("--z-refine-band", type=float, default=0.03)
    parser.add_argument("--smooth-window", type=float, default=0.75)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--chunk-points", type=int, default=5_000_000)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--preview-max-points", type=int, default=100_000)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig(
        x_min_mm=args.x_min,
        x_max_mm=args.x_max,
        y_min_mm=args.y_min,
        z_min_mm=args.z_min,
        z_max_mm=args.z_max,
        bin_width_mm=args.bin_width,
        top_band_mm=args.top_band,
        z_refine_band_mm=args.z_refine_band,
        smooth_window_mm=args.smooth_window,
        output_spacing_mm=args.spacing,
        chunk_points=args.chunk_points,
        min_valid_fraction=args.min_valid_fraction,
    )
    extract_path(
        args.pcd,
        args.output_dir,
        config,
        make_preview=not args.no_preview,
        preview_max_points=args.preview_max_points,
    )


if __name__ == "__main__":
    main()
