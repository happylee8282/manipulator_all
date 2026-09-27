"""Repair and verify the UR3e ROS 2 OmniGraph in the currently open Isaac stage.

Run this file from Isaac Sim's Script Editor while the timeline is stopped:

    exec(open("/home/happy/modelsoultion/old_step2/glass_robotarm/repair_ur3e_omnigraph.py").read())
"""

import omni.timeline
import omni.usd
import omni.graph.core as og
from pxr import Sdf, UsdPhysics


ROBOT_PRIM = Sdf.Path("/World/ur3e_physics")
EXPECTED_ARTICULATION_ROOT = Sdf.Path("/World/ur3e_physics/root_joint")
PUBLISHER_PRIM = Sdf.Path("/Graph/ROS_JointStates/PublisherJointState")
CONTROLLER_PRIM = Sdf.Path("/Graph/ROS_JointStates/ArticulationController")
SYSTEM_TIME_PRIM = Sdf.Path("/Graph/ROS_JointStates/ReadSystemTime")


timeline = omni.timeline.get_timeline_interface()
if timeline.is_playing():
    timeline.stop()
    print("[UR3e] Timeline stopped. Press Play after this script completes.")

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage is open in Isaac Sim.")

robot = stage.GetPrimAtPath(ROBOT_PRIM)
if not robot.IsValid():
    raise RuntimeError(
        f"{ROBOT_PRIM} is missing. Open old_step2/glass_robotarm/ur_e_0813_nozzle.usda."
    )

articulation_roots = [
    prim.GetPath()
    for prim in stage.Traverse()
    if prim.GetPath().HasPrefix(ROBOT_PRIM)
    and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
]
if EXPECTED_ARTICULATION_ROOT not in articulation_roots:
    found = ", ".join(str(path) for path in articulation_roots) or "none"
    raise RuntimeError(
        "Expected PhysicsArticulationRootAPI at "
        f"{EXPECTED_ARTICULATION_ROOT}, but found: {found}. "
        "The local ur3e_physics.usd reference did not compose correctly."
    )

publisher = stage.GetPrimAtPath(PUBLISHER_PRIM)
controller = stage.GetPrimAtPath(CONTROLLER_PRIM)
for path, prim in ((PUBLISHER_PRIM, publisher), (CONTROLLER_PRIM, controller)):
    if not prim.IsValid():
        raise RuntimeError(f"Required OmniGraph node is missing: {path}")

    target = prim.GetRelationship("inputs:targetPrim")
    if not target.IsValid():
        raise RuntimeError(f"inputs:targetPrim relationship is missing on {path}")
    target.SetTargets([EXPECTED_ARTICULATION_ROOT])

robot_path = controller.GetAttribute("inputs:robotPath")
if robot_path.IsValid():
    robot_path.Set("")

graph = og.get_graph_by_path("/Graph/ROS_JointStates")
if not graph.is_valid():
    raise RuntimeError("OmniGraph is not valid: /Graph/ROS_JointStates")

system_time = stage.GetPrimAtPath(SYSTEM_TIME_PRIM)
if not system_time.IsValid():
    og.Controller.edit(
        graph,
        {
            og.Controller.Keys.CREATE_NODES: [
                ("ReadSystemTime", "isaacsim.core.nodes.IsaacReadSystemTime")
            ]
        },
    )
    system_time = stage.GetPrimAtPath(SYSTEM_TIME_PRIM)

node_type = system_time.GetAttribute("node:type").Get()
if node_type != "isaacsim.core.nodes.IsaacReadSystemTime":
    raise RuntimeError(f"Unexpected node type on {SYSTEM_TIME_PRIM}: {node_type}")

time_stamp = publisher.GetAttribute("inputs:timeStamp")
if not time_stamp.IsValid():
    raise RuntimeError(f"inputs:timeStamp is missing on {PUBLISHER_PRIM}")

# Change both USD backing and the live OmniGraph topology. Updating only the
# Sdf connection can leave a graph that was already compiled using simulation
# time until the stage is reopened.
time_stamp_og = og.Controller.attribute(
    "/Graph/ROS_JointStates/PublisherJointState.inputs:timeStamp"
)
system_time_og = og.Controller.attribute(
    "/Graph/ROS_JointStates/ReadSystemTime.outputs:systemTime"
)
if not time_stamp_og.is_valid() or not system_time_og.is_valid():
    raise RuntimeError("System-time OmniGraph ports were not initialized")

og.Controller.disconnect_all(time_stamp_og, update_usd=True, undoable=False)
og.Controller.connect(
    system_time_og,
    time_stamp_og,
    update_usd=True,
    undoable=False,
)

upstream = time_stamp_og.get_upstream_connections()
upstream_paths = [str(attribute.get_path()) for attribute in upstream]
expected_time_path = str(system_time_og.get_path())
if upstream_paths != [expected_time_path]:
    raise RuntimeError(f"Unexpected timestamp connections: {upstream_paths}")

print(f"[UR3e] Articulation roots: {articulation_roots}")
print(
    "[UR3e] Publisher target: "
    f"{publisher.GetRelationship('inputs:targetPrim').GetTargets()}"
)
print(
    "[UR3e] Controller target: "
    f"{controller.GetRelationship('inputs:targetPrim').GetTargets()}"
)
print(
    "[UR3e] Live timestamp source: "
    f"{upstream_paths}"
)
print("[UR3e] Repair complete. Press Play and check /isaac_joint_states.")
