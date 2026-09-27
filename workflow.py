"""Run path preparation stages sequentially, stopping on the first failure."""

from __future__ import annotations

import argparse
import fcntl
import math
from pathlib import Path
import subprocess
import sys
import time

from .configuration import load_config, read_mapping, require_capability, resolve_path


def wait_for_scene(config_path: Path, timeout: float) -> None:
    from .safety.glass_guard import read_guard_status

    raw = read_mapping(config_path)
    if raw["robot"].get("dry_run_only", False):
        return
    deadline = time.monotonic() + timeout
    last_error = "Scene has not reported ready"
    while time.monotonic() < deadline:
        try:
            config, root = load_config(config_path)
            read_guard_status(
                resolve_path(root, config["glass_guard"]["status_file"]),
                timeout_s=float(config["glass_guard"]["status_timeout_s"]),
                expected_snapshot_time=config["runtime_snapshot"]["captured_wall_time"],
            )
            return
        except (RuntimeError, FileNotFoundError, ValueError) as error:
            last_error = str(error)
            time.sleep(0.25)
    raise RuntimeError(f"Isaac readiness timeout: {last_error}")


def wait_for_moveit(config: dict, timeout: float) -> None:
    import rclpy
    from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPositionFK, GetPositionIK, GetStateValidity

    rclpy.init()
    node = rclpy.create_node("lee_manipulator_readiness")
    clients = [
        node.create_client(GetPositionIK, config["ik"]["service"]),
        node.create_client(GetPositionFK, config["ik"]["fk_service"]),
        node.create_client(GetStateValidity, config["ik"]["state_validity_service"]),
        node.create_client(ApplyPlanningScene, "/apply_planning_scene"),
        node.create_client(GetPlanningScene, "/get_planning_scene"),
    ]
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            missing = [client.srv_name for client in clients if not client.service_is_ready()]
            if not missing:
                return
            rclpy.spin_once(node, timeout_sec=0.1)
        raise RuntimeError(f"MoveIt readiness timeout; missing: {', '.join(missing)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def stages_for(operation: str, execute: bool) -> list[str]:
    choices = {
        "check": ["check"], "generate": ["generate"], "solve": ["solve"],
        "prepare": ["check", "generate", "solve"], "execute": ["execute"],
    }
    if operation not in choices:
        raise ValueError(f"Unknown operation: {operation}")
    if operation == "execute" and not execute:
        raise ValueError("operation:=execute also requires execute:=true")
    if execute and operation not in ("prepare", "execute"):
        raise ValueError("execute:=true requires operation:=prepare or operation:=execute")
    stages = list(choices[operation])
    if operation == "prepare" and execute:
        stages.append("execute")
    return stages


def run(config_path: Path, operation: str, execute: bool, ready_timeout: float) -> None:
    stages = stages_for(operation, execute)
    config = read_mapping(config_path)
    require_capability(config, execute=execute)
    if not math.isfinite(ready_timeout) or ready_timeout <= 0:
        raise ValueError("ready_timeout must be finite and positive")
    lock_path = Path(config["run_directory"]) / ".task.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another task is already using this robot's run directory") from None
        if operation != "check":
            wait_for_scene(config_path, ready_timeout)
        if "solve" in stages or "execute" in stages:
            wait_for_moveit(config, ready_timeout)
        modules = {
            "check": "safety.preflight", "generate": "path.cartesian_generator",
            "solve": "control.common.moveit_solver", "execute": "execution.guarded_executor",
        }
        for stage in stages:
            command = [sys.executable, "-m", f"lee_manipulator_part.{modules[stage]}",
                       "--config", str(config_path)]
            if stage == "execute":
                command.append("--execute")
            print(f"[{stage}] {config['selection']['robot']}", flush=True)
            subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--operation", choices=("check", "generate", "solve", "prepare", "execute"), default="prepare")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.operation, args.execute, args.ready_timeout)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Task stopped: {error}\n")


if __name__ == "__main__":
    main()
