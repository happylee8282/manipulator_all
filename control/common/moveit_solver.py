#!/usr/bin/env python3
"""Solve and verify the complete Cartesian path with MoveIt 2 services.

MoveIt 2 is used for the jobs it is good at: robot-model-aware IK, FK, joint
limits, self collision, and world collision.  This node does not ask MoveIt to
invent a Cartesian path; the Step 2/Cartesian generator already defined it.
Instead, a small beam search keeps continuous IK branches and rejects any final
path that fails FK or collision revalidation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Iterable

import numpy as np

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    CollisionObject,
    MoveItErrorCodes,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetPlanningScene,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive

from ...configuration import DEFAULT_CONFIG, load_config, output_path, resolve_path
from .geometry import quaternion_error_deg
from .ik_core import (
    BranchState,
    equivalent_near_reference,
    incremental_branch_cost,
    reconstruct_branch,
    seed_variants,
)
from .trajectory import time_parameterize


MOVEIT_SUCCESS = MoveItErrorCodes.SUCCESS


def duration_message(seconds: float) -> Duration:
    nanoseconds = max(0, int(round(seconds * 1.0e9)))
    return Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL and return unique vertices plus triangle indices."""

    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    if path.stat().st_size != 84 + triangle_count * 50:
        raise ValueError(f"only binary STL is supported: {path}")
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    records = np.fromfile(path, dtype=dtype, count=triangle_count, offset=84)
    vertices, inverse = np.unique(
        records["vertices"].reshape(-1, 3), axis=0, return_inverse=True
    )
    return vertices.astype(float), inverse.reshape(-1, 3)


class MoveItPathSolver(Node):
    """Offline path-wide IK solver with final FK and collision proof."""

    def __init__(self, config_path: Path, trajectory_path: Path | None = None) -> None:
        super().__init__("step3_moveit_path_solver")
        self.config, self.root = load_config(config_path)
        self.project = self.config["project"]
        self.ik = self.config["ik"]
        self.joint_names = tuple(self.project["joint_names"])
        self.lower = np.asarray(self.ik["joint_lower_rad"], float)
        self.upper = np.asarray(self.ik["joint_upper_rad"], float)
        input_path = trajectory_path or output_path(
            self.config, self.root, "cartesian_archive"
        )
        archive = np.load(input_path)
        self.process_positions = archive["positions_m"].astype(float)
        self.command_positions = archive.get(
            "command_positions_m", archive["positions_m"]
        ).astype(float)
        self.quaternions = archive["quaternions_xyzw"].astype(float)
        self.phases = archive["phase"].astype(str)
        if not (
            len(self.process_positions)
            == len(self.command_positions)
            == len(self.quaternions)
            == len(self.phases)
        ):
            raise ValueError("Cartesian archive arrays have different lengths")

        self.ik_client = self.create_client(GetPositionIK, str(self.ik["service"]))
        self.fk_client = self.create_client(GetPositionFK, str(self.ik["fk_service"]))
        self.validity_client = self.create_client(
            GetStateValidity, str(self.ik["state_validity_service"])
        )
        self.scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.scene_query_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.current_joints: np.ndarray | None = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            JointState,
            str(self.config["controller"]["state_topic"]),
            self._state_callback,
            qos,
        )
        self.service_counts = {"ik": 0, "fk": 0, "validity": 0}

    def _state_callback(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        if all(name in values for name in self.joint_names):
            self.current_joints = np.asarray([values[name] for name in self.joint_names], float)

    def _wait_for_services(self) -> None:
        for name, client in (
            ("IK", self.ik_client),
            ("FK", self.fk_client),
            ("state validity", self.validity_client),
            ("planning scene", self.scene_client),
            ("planning-scene query", self.scene_query_client),
        ):
            if not client.wait_for_service(timeout_sec=20.0):
                raise RuntimeError(f"MoveIt {name} service is unavailable")

    def _call(self, client, request, timeout: float):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def _apply_collision_scene(self) -> None:
        settings = self.config["collision_scene"]
        scene = PlanningScene()
        scene.is_diff = True
        if bool(settings["publish_glass"]):
            vertices_mm, faces = read_binary_stl(
                resolve_path(self.root, self.config["resources"]["glass_stl"])
            )
            transform = np.asarray(
                self.config["frame_transform"]["mesh_local_mm_to_world_m"], float
            )
            world = (
                transform
                @ np.column_stack((vertices_mm, np.ones(len(vertices_mm)))).T
            ).T[:, :3]
            world += np.asarray(
                self.config["frame_transform"]["task_world_offset_m"], float
            )
            mesh = Mesh()
            mesh.vertices = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in world]
            for indices in faces:
                triangle = MeshTriangle()
                triangle.vertex_indices = [int(value) for value in indices]
                mesh.triangles.append(triangle)
            glass = CollisionObject()
            glass.header.frame_id = str(self.project["planning_frame"])
            glass.id = "AI_Glass_Front"
            glass.operation = CollisionObject.ADD
            glass.meshes = [mesh]
            mesh_pose = Pose()
            mesh_pose.orientation.w = 1.0
            glass.mesh_poses = [mesh_pose]
            scene.world.collision_objects.append(glass)
        if bool(settings["publish_floor"]):
            scene.world.collision_objects.append(
                self._box_object(
                    "BlackGridFloor",
                    settings["floor_center_world_m"],
                    settings["floor_size_m"],
                )
            )
        if bool(settings["publish_worktable"]):
            scene.world.collision_objects.append(
                self._box_object(
                    "WhiteWorktable",
                    settings["worktable_center_world_m"],
                    settings["worktable_size_m"],
                )
            )
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self._call(self.scene_client, request, 30.0)
        if response is None or not response.success:
            raise RuntimeError("MoveIt rejected the collision scene")
        if bool(settings["publish_glass"]):
            # Always write the pair, including False. Otherwise an Allowed
            # Collision Matrix entry from an earlier permissive run survives
            # and silently defeats a later non-contact commissioning run.
            self._set_collision_pair(
                str(settings["allowed_glass_link"]),
                "AI_Glass_Front",
                bool(settings["allow_nozzle_glass_contact"]),
            )
        self.get_logger().info(
            "Collision objects: "
            + ", ".join(item.id for item in scene.world.collision_objects)
        )

    def _box_object(self, name: str, center: Iterable[float], size: Iterable[float]):
        center = np.asarray(center, float)
        size = np.asarray(size, float)
        if center.shape != (3,) or size.shape != (3,) or np.any(size <= 0.0):
            raise ValueError(f"invalid collision box {name}")
        item = CollisionObject()
        item.header.frame_id = str(self.project["planning_frame"])
        item.id = name
        item.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size.tolist()
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, center)
        pose.orientation.w = 1.0
        item.primitives = [primitive]
        item.primitive_poses = [pose]
        return item

    def _set_collision_pair(self, first: str, second: str, allowed: bool) -> None:
        query = GetPlanningScene.Request()
        query.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        response = self._call(self.scene_query_client, query, 10.0)
        if response is None:
            raise RuntimeError("could not query MoveIt Allowed Collision Matrix")
        matrix = response.scene.allowed_collision_matrix
        old_count = len(matrix.entry_names)
        while len(matrix.entry_values) < old_count:
            matrix.entry_values.append(AllowedCollisionEntry(enabled=[False] * old_count))
        for row in matrix.entry_values:
            row.enabled = list(row.enabled[:old_count]) + [False] * max(
                0, old_count - len(row.enabled)
            )
        for name in (first, second):
            if name in matrix.entry_names:
                continue
            new_count = len(matrix.entry_names) + 1
            matrix.entry_names.append(name)
            for row in matrix.entry_values:
                row.enabled.append(False)
            matrix.entry_values.append(AllowedCollisionEntry(enabled=[False] * new_count))
        left = matrix.entry_names.index(first)
        right = matrix.entry_names.index(second)
        matrix.entry_values[left].enabled[right] = bool(allowed)
        matrix.entry_values[right].enabled[left] = bool(allowed)
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = matrix
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self._call(self.scene_client, request, 10.0)
        if response is None or not response.success:
            raise RuntimeError(f"could not allow intentional contact {first} <-> {second}")

    def _ik(
        self,
        pose_index: int,
        seed: np.ndarray,
        *,
        avoid_collisions: bool | None = None,
    ) -> np.ndarray | None:
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = str(self.project["planning_group"])
        ik.ik_link_name = str(self.project["end_effector_link"])
        ik.avoid_collisions = (
            bool(self.ik["avoid_collisions"])
            if avoid_collisions is None
            else bool(avoid_collisions)
        )
        ik.timeout = duration_message(float(self.ik["timeout_s"]))
        ik.robot_state.is_diff = True
        ik.robot_state.joint_state.name = list(self.joint_names)
        ik.robot_state.joint_state.position = seed.tolist()
        pose = PoseStamped()
        pose.header.frame_id = str(self.project["planning_frame"])
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(
            float, self.command_positions[pose_index]
        )
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = map(float, self.quaternions[pose_index])
        ik.pose_stamped = pose
        self.service_counts["ik"] += 1
        response = self._call(
            self.ik_client, request, max(1.0, 5.0 * float(self.ik["timeout_s"]))
        )
        if response is None or response.error_code.val != MOVEIT_SUCCESS:
            return None
        incoming = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        if not all(name in incoming for name in self.joint_names):
            return None
        return np.asarray([incoming[name] for name in self.joint_names], float)

    def _diagnose_failed_pose(
        self, pose_index: int, beam: list[BranchState], attempts: int
    ) -> str:
        """Distinguish geometry/IK failure from collision rejection without accepting it."""
        contacts: set[str] = set()
        collision_free_ik_exists = False
        unconstrained_ik_exists = False
        for parent in beam:
            for seed in self._seed_variants(parent.q, attempts):
                solution = self._ik(pose_index, seed, avoid_collisions=False)
                if solution is None:
                    continue
                unconstrained_ik_exists = True
                solution = equivalent_near_reference(
                    solution, parent.q, self.lower, self.upper
                )
                if solution is None:
                    continue
                valid, state_contacts = self._is_valid(solution)
                contacts.update(state_contacts)
                collision_free_ik_exists = collision_free_ik_exists or valid
                if state_contacts:
                    break
            if contacts:
                break
        if contacts:
            return "collision rejection: " + ", ".join(sorted(contacts))
        if collision_free_ik_exists:
            return "continuous-branch or joint-step rejection (collision-free IK exists)"
        if unconstrained_ik_exists:
            return "joint-limit equivalence rejection after unconstrained IK"
        return "no IK solution even with collision checking disabled"

    def _seed_variants(self, seed: np.ndarray, count: int) -> list[np.ndarray]:
        return seed_variants(
            seed, count, float(self.ik["seed_perturbation_rad"]), self.lower, self.upper,
            self.ik.get("seed_perturbation_direction"),
        )

    def _initial_beam(self) -> list[BranchState]:
        seeds = [np.asarray(values, float) for values in self.ik["initial_seeds_rad"]]
        if self.current_joints is not None:
            seeds.insert(0, self.current_joints.copy())
        states: list[BranchState] = []
        tolerance = float(self.ik["deduplicate_tolerance_rad"])
        weights = self.ik["cost"]
        for seed in seeds:
            solution = self._ik(0, seed)
            if solution is None:
                continue
            solution = equivalent_near_reference(solution, seed, self.lower, self.upper)
            if solution is None or any(np.linalg.norm(solution - old.q) < tolerance for old in states):
                continue
            delta = solution - seed
            cost = incremental_branch_cost(
                solution, seed, np.zeros_like(seed), self.lower, self.upper, weights
            )
            states.append(BranchState(solution, delta, cost, None, 0))
        if not states:
            raise RuntimeError("no configured seed can solve the first Cartesian pose")
        return sorted(states, key=lambda item: item.total_cost)[: int(self.ik["beam_width"])]

    def _solve_beam(self) -> tuple[np.ndarray, float, int]:
        beam = self._initial_beam()
        width = int(self.ik["beam_width"])
        refresh = int(self.ik["branch_refresh_interval"])
        attempts = int(self.ik["seeds_per_branch"])
        tolerance = float(self.ik["deduplicate_tolerance_rad"])
        maximum_step = float(self.ik["max_joint_step_rad"])
        weights = self.ik["cost"]
        maximum_beam_seen = len(beam)
        for pose_index in range(1, len(self.command_positions)):
            candidate_states: list[BranchState] = []
            trial_count = attempts if pose_index % refresh == 0 else 1
            for parent in beam:
                solutions: list[np.ndarray] = []
                for seed in self._seed_variants(parent.q, trial_count):
                    solution = self._ik(pose_index, seed)
                    if solution is None:
                        continue
                    solution = equivalent_near_reference(
                        solution, parent.q, self.lower, self.upper
                    )
                    if solution is None:
                        continue
                    delta = solution - parent.q
                    if float(np.max(np.abs(delta))) > maximum_step:
                        continue
                    if any(np.linalg.norm(solution - old) < tolerance for old in solutions):
                        continue
                    solutions.append(solution)
                    incremental = incremental_branch_cost(
                        solution,
                        parent.q,
                        parent.previous_delta,
                        self.lower,
                        self.upper,
                        weights,
                    )
                    candidate_states.append(
                        BranchState(
                            solution,
                            delta,
                            parent.total_cost + incremental,
                            parent,
                            pose_index,
                        )
                    )
            if not candidate_states:
                # A local recovery only runs after all normal branches fail.
                for parent in beam:
                    for seed in self._seed_variants(parent.q, attempts):
                        solution = self._ik(pose_index, seed)
                        if solution is None:
                            continue
                        solution = equivalent_near_reference(
                            solution, parent.q, self.lower, self.upper
                        )
                        if solution is None:
                            continue
                        delta = solution - parent.q
                        if float(np.max(np.abs(delta))) <= maximum_step:
                            candidate_states.append(
                                BranchState(
                                    solution,
                                    delta,
                                    parent.total_cost
                                    + incremental_branch_cost(
                                        solution,
                                        parent.q,
                                        parent.previous_delta,
                                        self.lower,
                                        self.upper,
                                        weights,
                                    ),
                                    parent,
                                    pose_index,
                                )
                            )
                if not candidate_states:
                    diagnosis = self._diagnose_failed_pose(pose_index, beam, attempts)
                    raise RuntimeError(
                        f"all continuous IK branches failed at pose {pose_index}: {diagnosis}"
                    )
            candidate_states.sort(key=lambda item: item.total_cost)
            beam = []
            for candidate in candidate_states:
                if any(np.linalg.norm(candidate.q - old.q) < tolerance for old in beam):
                    continue
                beam.append(candidate)
                if len(beam) >= width:
                    break
            maximum_beam_seen = max(maximum_beam_seen, len(beam))
            if pose_index % 100 == 0 or pose_index == len(self.command_positions) - 1:
                self.get_logger().info(
                    f"IK beam: {pose_index + 1}/{len(self.command_positions)}, branches={len(beam)}"
                )
        selected = min(beam, key=lambda item: item.total_cost)
        path = reconstruct_branch(selected)
        if len(path) != len(self.command_positions):
            raise RuntimeError("IK branch reconstruction length mismatch")
        return path, float(selected.total_cost), maximum_beam_seen

    def _fk(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        request = GetPositionFK.Request()
        request.header.frame_id = str(self.project["planning_frame"])
        request.header.stamp = self.get_clock().now().to_msg()
        request.fk_link_names = [str(self.project["end_effector_link"])]
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(self.joint_names)
        request.robot_state.joint_state.position = joints.tolist()
        self.service_counts["fk"] += 1
        response = self._call(self.fk_client, request, 5.0)
        if (
            response is None
            or response.error_code.val != MOVEIT_SUCCESS
            or not response.pose_stamped
        ):
            raise RuntimeError("MoveIt FK request failed")
        pose = response.pose_stamped[0].pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        quaternion = np.array(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        )
        return position, quaternion / np.linalg.norm(quaternion)

    def _is_valid(self, joints: np.ndarray) -> tuple[bool, list[str]]:
        request = GetStateValidity.Request()
        request.group_name = str(self.project["planning_group"])
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(self.joint_names)
        request.robot_state.joint_state.position = joints.tolist()
        self.service_counts["validity"] += 1
        response = self._call(self.validity_client, request, 5.0)
        if response is None:
            raise RuntimeError("MoveIt state-validity request timed out")
        contacts = sorted(
            {
                " <-> ".join(sorted((item.contact_body_1, item.contact_body_2)))
                for item in response.contacts
            }
        )
        return bool(response.valid), contacts

    def _verify_path(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
        position_error_mm = np.empty(len(joints))
        orientation_error_deg = np.empty(len(joints))
        invalid: list[int] = []
        for index, q in enumerate(joints):
            position, quaternion = self._fk(q)
            position_error_mm[index] = (
                np.linalg.norm(position - self.command_positions[index]) * 1000.0
            )
            orientation_error_deg[index] = quaternion_error_deg(
                quaternion, self.quaternions[index]
            )[0]
            valid, contacts = self._is_valid(q)
            if not valid:
                invalid.append(index)
                self.get_logger().error(f"invalid state {index}: {contacts}")
            if index % 100 == 0 or index == len(joints) - 1:
                self.get_logger().info(f"FK/collision proof: {index + 1}/{len(joints)}")
        if invalid:
            raise RuntimeError(f"final path has {len(invalid)} collision-invalid states")
        if float(np.max(position_error_mm)) > float(self.ik["fk_position_tolerance_mm"]):
            raise RuntimeError(
                f"FK position error {np.max(position_error_mm):.6f} mm exceeds tolerance"
            )
        if float(np.max(orientation_error_deg)) > float(
            self.ik["fk_orientation_tolerance_deg"]
        ):
            raise RuntimeError(
                f"FK orientation error {np.max(orientation_error_deg):.6f} deg exceeds tolerance"
            )
        return position_error_mm, orientation_error_deg, invalid

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        self._wait_for_services()
        # Briefly process a live state when Isaac is present; configured seeds remain valid offline.
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.current_joints is not None:
                break
        self._apply_collision_scene()
        joints, branch_cost, max_beam = self._solve_beam()
        fk_position, fk_orientation, invalid = self._verify_path(joints)
        maximum_step = np.max(np.abs(np.diff(joints, axis=0)), axis=0)
        timed = time_parameterize(
            joints,
            self.process_positions,
            self.phases,
            self.config["time_parameterization"],
            float(self.config["cartesian_pose"]["endpoint_settle_time_s"]),
            self.lower,
            self.upper,
        )
        segment_distance_mm = (
            np.linalg.norm(np.diff(self.process_positions, axis=0), axis=1) * 1000.0
        )
        segment_dt = np.diff(timed.time_s)
        bonding_interval = (self.phases[:-1] == "bonding") & (
            self.phases[1:] == "bonding"
        )
        bonding_feedrate = segment_distance_mm[bonding_interval] / segment_dt[bonding_interval]
        archive_path = output_path(self.config, self.root, "joint_archive")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            archive_path,
            joint_names=np.asarray(self.joint_names),
            positions=timed.positions,
            velocities=timed.velocities,
            accelerations=timed.accelerations,
            time_s=timed.time_s,
            cartesian_positions_m=self.process_positions,
            command_cartesian_positions_m=self.command_positions,
            cartesian_quaternions_xyzw=self.quaternions,
            phase=self.phases,
            fk_position_error_mm=fk_position,
            fk_orientation_error_deg=fk_orientation,
        )
        metrics = {
            "status": "PASS",
            "pose_count": len(joints),
            "selected_branch_cost": branch_cost,
            "maximum_beam_width_seen": max_beam,
            "service_call_count": self.service_counts,
            "max_absolute_joint_step_rad": maximum_step.tolist(),
            "fk_position_error_mm": stats(fk_position),
            "fk_orientation_error_deg": stats(fk_orientation),
            "collision_invalid_state_count": len(invalid),
            "nozzle_glass_contact_allowed": bool(
                self.config["collision_scene"]["allow_nozzle_glass_contact"]
            ),
            "duration_s": float(timed.time_s[-1]),
            "bonding_duration_s": float(np.sum(segment_dt[bonding_interval])),
            "bonding_distance_mm": float(np.sum(segment_distance_mm[bonding_interval])),
            "configured_bonding_speed_mm_s": float(
                self.config["time_parameterization"]["bonding_speed_mm_s"]
            ),
            "realised_bonding_speed_mm_s": stats(bonding_feedrate),
            "global_time_scale": timed.global_time_scale,
            "max_joint_velocity_rad_s": timed.maximum_velocity.tolist(),
            "max_joint_acceleration_rad_s2": timed.maximum_acceleration.tolist(),
            "max_joint_jerk_rad_s3": timed.maximum_jerk.tolist(),
            "wall_time_s": time.monotonic() - started,
            "joint_archive": str(archive_path),
        }
        metrics_path = output_path(self.config, self.root, "solver_metrics")
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        self.get_logger().info(
            f"PASS: {len(joints)} poses, FK max={np.max(fk_position):.6f} mm, "
            f"duration={timed.time_s[-1]:.3f} s"
        )
        return metrics


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = MoveItPathSolver(args.config, args.trajectory)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
