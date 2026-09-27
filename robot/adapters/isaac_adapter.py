"""Build Isaac commands without importing Isaac into the ROS interpreter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def build_command(
    config: dict[str, Any],
    root: str | Path,
    python_executable: str,
    gui: bool = True,
) -> list[str]:
    """Return a simulator-only argv; runtime files are per-robot/run paths."""
    root = Path(root).resolve()
    robot = config["robot"]
    runner = _resolve(root, robot["isaac_runner_file"])
    stage = _resolve(root, config["resources"]["stage"])
    interpreter = Path(python_executable).expanduser()
    for path in (runner, stage, interpreter):
        if not path.is_file():
            raise FileNotFoundError(f"Isaac startup dependency does not exist: {path}")
    runtime = _resolve(root, config["glass_guard"]["status_file"]).parent
    runtime.mkdir(parents=True, exist_ok=True)
    command = [str(interpreter), str(runner), "--stage", str(stage)]
    if gui:
        command.append("--gui")
    command.extend(["--ready-file", str(runtime / "isaac.ready")])
    if robot["name"] == "ur3e":
        command.extend([
            "--repair-script", str(runner.with_name("ur3e_graph_repair.py")),
            "--diagnostics-json", str(runtime / "isaac_diagnostics.json"),
            "--update-hz", str(config["controller"]["publish_rate_hz"]),
            "--command-topic", config["controller"]["command_topic"],
            "--state-topic", config["controller"]["state_topic"],
        ])
        return command
    if robot["name"] != "rv4fl":
        raise ValueError(f"No Isaac runner is configured for {robot['name']}")
    settle = config["glass_settle"]
    guard = config["glass_guard"]
    command.extend([
        "--snapshot", str(_resolve(root, config["frame_transform"]["glass_snapshot"])),
        "--status", str(_resolve(root, guard["status_file"])),
        "--command", str(_resolve(root, guard["command_file"])),
        "--diagnostics", str(_resolve(root, config["non_contact_safety"]["isaac_diagnostics_file"])),
        "--glass-prim", settle["glass_prim"],
        "--glass-mesh-prim", settle["glass_mesh_prim"],
        "--robot-root", settle["robot_articulation_prim"],
        "--tool-prim", robot["isaac"]["tool_prim"],
        "--command-topic", config["controller"]["command_topic"],
        "--state-topic", config["controller"]["state_topic"],
        "--mass-kg", str(settle["mass_kg"]),
        "--stable-duration-s", str(settle["stable_duration_s"]),
        "--settle-timeout-s", str(settle["timeout_s"]),
        "--stable-linear-speed-m-s", str(settle["maximum_linear_speed_m_s"]),
        "--stable-angular-speed-deg-s", str(settle["maximum_angular_speed_deg_s"]),
        "--stable-translation-mm", str(settle["maximum_translation_in_window_mm"]),
        "--stable-rotation-deg", str(settle["maximum_rotation_in_window_deg"]),
        "--max-translation-mm", str(guard["max_translation_mm"]),
        "--max-rotation-deg", str(guard["max_rotation_deg"]),
        "--sdf-resolution", str(config["non_contact_safety"]["minimum_isaac_sdf_resolution"]),
        "--update-hz", str(config["controller"]["publish_rate_hz"]),
        "--initial-pose-deg",
        *[str(math.degrees(value)) for value in config["execution"]["initial_pose_rad"]],
    ])
    return command
