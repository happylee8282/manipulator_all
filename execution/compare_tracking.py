#!/usr/bin/env python3
"""Create a strict old-vs-new A/B tracking report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..configuration import DEFAULT_CONFIG, load_config, output_path


def _old_value(metrics: dict[str, Any], name: str) -> float:
    if name == "tcp_p95_mm":
        return float(metrics["time_synchronised_position_error_mm"]["p95"])
    if name == "tcp_max_mm":
        return float(metrics["time_synchronised_position_error_mm"]["max"])
    if name == "orientation_p95_deg":
        return float(metrics["orientation_error_deg"]["p95"])
    if name == "joint_p95_rad":
        return float(metrics["joint_tracking_error_rad"]["max_abs_joint"]["p95"])
    raise KeyError(name)


def _new_value(metrics: dict[str, Any], name: str) -> float:
    if name == "tcp_p95_mm":
        return float(metrics["tcp_position_error_mm"]["p95"])
    if name == "tcp_max_mm":
        return float(metrics["tcp_position_error_mm"]["max"])
    if name == "orientation_p95_deg":
        return float(metrics["tcp_orientation_error_deg"]["p95"])
    if name == "joint_p95_rad":
        return float(metrics["max_abs_joint_error_rad"]["p95"])
    raise KeyError(name)


def compare(config_path: Path, old_path: Path, new_path: Path | None = None) -> dict[str, Any]:
    config, root = load_config(config_path)
    output = output_path(config, root, "comparison_json")
    old_metrics = json.loads(old_path.read_text(encoding="utf-8"))
    new_metrics_path = new_path or output_path(config, root, "tracking_metrics")
    if not new_metrics_path.exists():
        report = {
            "status": "PENDING_NEW_ISAAC_RUN",
            "reason": "Run execute_solved_path once, then repeat this comparison.",
            "old_metrics_file": str(old_path),
            "new_metrics_file": str(new_metrics_path),
            "old_baseline": {
                name: _old_value(old_metrics, name)
                for name in ("tcp_p95_mm", "tcp_max_mm", "orientation_p95_deg", "joint_p95_rad")
            },
            "warning": "The baseline must be regenerated if its Cartesian target does not match the new run.",
        }
    else:
        new_metrics = json.loads(new_metrics_path.read_text(encoding="utf-8"))
        names = ("tcp_p95_mm", "tcp_max_mm", "orientation_p95_deg", "joint_p95_rad")
        comparisons = {}
        for name in names:
            old = _old_value(old_metrics, name)
            new = _new_value(new_metrics, name)
            comparisons[name] = {
                "old": old,
                "new": new,
                "improvement_percent": 100.0 * (old - new) / old,
                "pass": new < old,
            }
        # Primary acceptance is TCP p95; all secondary regressions are still surfaced.
        report = {
            "status": "PASS_BETTER" if all(item["pass"] for item in comparisons.values()) else "NOT_YET_BETTER",
            "acceptance_rule": "All four measured p95/max indicators must be strictly lower.",
            "old_metrics_file": str(old_path),
            "new_metrics_file": str(new_metrics_path),
            "comparisons": comparisons,
        }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--old", type=Path, required=True, help="Explicit baseline tracking metrics JSON")
    parser.add_argument("--new", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare(args.config, args.old.expanduser().resolve(), args.new)


if __name__ == "__main__":
    main()
