"""Exactly-once gripper events at acknowledged, stopped path waypoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from .gripper_client import GripperCommandRequest, GripperCommandResult


@dataclass(frozen=True)
class GripperEvent:
    name: str
    waypoint: int
    command: GripperCommandRequest

    def __post_init__(self):
        if not self.name:
            raise ValueError("gripper event name must not be empty")
        if isinstance(self.waypoint, bool) or not isinstance(self.waypoint, int) or self.waypoint < 0:
            raise ValueError("event waypoint must be a non-negative integer")
        if not isinstance(self.command, GripperCommandRequest):
            raise ValueError("event command must be a GripperCommandRequest")


class WaypointGripperScheduler:
    """Caller splits trajectories at stop_waypoints and provides measured state.

    This helper does not interrupt a running trajectory or estimate arrival from
    elapsed time. A failed attempt latches the task stopped to avoid an implicit
    retry after uncertain physical actuation. Create a new scheduler only after
    explicit recovery and checking the current gripper state.
    """

    def __init__(
        self,
        events: Iterable[GripperEvent],
        *,
        stationary_tolerance_rad_s: float = 0.005,
        maximum_state_age_s: float = 0.2,
    ):
        self.events = tuple(events)
        if len({event.name for event in self.events}) != len(self.events):
            raise ValueError("gripper event names must be unique")
        for value, name in (
            (stationary_tolerance_rad_s, "stationary tolerance"),
            (maximum_state_age_s, "maximum state age"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.stationary_tolerance_rad_s = stationary_tolerance_rad_s
        self.maximum_state_age_s = maximum_state_age_s
        self.completed: set[str] = set()
        self.failed_event: str | None = None

    @property
    def stop_waypoints(self) -> tuple[int, ...]:
        return tuple(sorted({event.waypoint for event in self.events}))

    def execute_at_stop(
        self,
        waypoint: int,
        *,
        reached: bool,
        joint_velocities_rad_s: Iterable[float],
        state_age_s: float,
        command: Callable[[GripperCommandRequest], GripperCommandResult],
    ) -> tuple[str, ...]:
        if self.failed_event is not None:
            raise RuntimeError(f"gripper event {self.failed_event!r} failed; recovery required")
        pending = [event for event in self.events if event.name not in self.completed]
        if any(event.waypoint < waypoint for event in pending):
            raise RuntimeError("a required gripper stop waypoint was skipped")
        due = [event for event in pending if event.waypoint == waypoint]
        if not due:
            return ()
        if not reached:
            raise RuntimeError("arm waypoint arrival has not been confirmed")
        velocities = tuple(float(value) for value in joint_velocities_rad_s)
        if not velocities or any(not math.isfinite(value) for value in velocities):
            raise RuntimeError("measured arm joint velocities are required")
        if not math.isfinite(state_age_s) or not 0.0 <= state_age_s <= self.maximum_state_age_s:
            raise RuntimeError("arm velocity state is stale or invalid")
        if max(abs(value) for value in velocities) > self.stationary_tolerance_rad_s:
            raise RuntimeError("arm must stop before a gripper event")
        executed = []
        for event in due:
            try:
                result = command(event.command)
                if not isinstance(result, GripperCommandResult) or not result.accepted_for(event.command):
                    raise RuntimeError(f"gripper event {event.name!r} did not complete successfully")
            except Exception:
                self.failed_event = event.name
                raise
            self.completed.add(event.name)
            executed.append(event.name)
        return tuple(executed)
