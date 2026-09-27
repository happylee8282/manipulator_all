"""Hardware backend extension point with explicit unsupported behavior."""

from __future__ import annotations

from typing import Any


def build_command(config: dict[str, Any], **_kwargs: Any) -> list[str]:
    robot_name = config.get("robot", {}).get("name", "unknown")
    raise NotImplementedError(
        f"Real driver for {robot_name} has not been commissioned in this package. "
        "Register a manufacturer driver, FollowJointTrajectory action, measured "
        "joint-state topic, tool calibration and controller capabilities first. "
        "Isaac JointState command publishing is not a real-robot transport."
    )
