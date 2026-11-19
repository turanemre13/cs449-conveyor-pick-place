"""Execute a checked transfer, release into a bin, and empty-hand retreat."""

from dataclasses import dataclass
from math import dist
from typing import TYPE_CHECKING, Callable

from conveyor.planning.moving_panda import PandaInterception
from conveyor.planning.panda import PandaPlanningError, TcpPose, tcp_pose_error
from conveyor.planning.panda_placement import plan_panda_placement, require_object_in_bin
from conveyor.prediction import tracked_states_from_ground_truth

if TYPE_CHECKING:
    from conveyor.simulation import ConveyorSimulation, ObjectGroundTruth, TargetBin


@dataclass(frozen=True)
class PandaPlacementResult:
    target_id: str
    release_time_s: float
    floor_gap_m: float


def release_into_bin(
    simulation: "ConveyorSimulation", gripper_frame: str, object_id: str, target: "TargetBin",
) -> "ObjectGroundTruth":
    """Check containment and preserve the placed world pose at handoff."""
    config = simulation.config
    parent = config.getFrame(object_id).getParent()
    if parent is None or parent.name != gripper_frame:
        raise PandaPlanningError(f"{object_id} is missing from the gripper at release")
    require_object_in_bin(config, object_id, target)
    before = simulation.get_ground_truth()[object_id]
    config.attach("world", object_id)
    return before


def _play_arm(simulation, names, samples, after_step):
    previous = 0.0
    for sample in samples[1:]:
        simulation.config.setJointState(sample.positions_rad, names)
        simulation.step(sample.elapsed_s - previous)
        previous = sample.elapsed_s
        if after_step is not None and not after_step(simulation.simulation_time_s):
            return False
    return True


def execute_panda_placement(
    simulation: "ConveyorSimulation",
    pickup: PandaInterception,
    *,
    after_step: Callable[[float], bool] | None = None,
) -> PandaPlacementResult | None:
    """Place a held object into the bin authored for the selected Panda."""
    closure = pickup.closure
    if closure is None:
        raise ValueError("Placement requires a verified pickup with jaw closure")
    truth = simulation.get_ground_truth(active_only=True)
    obj = truth.get(pickup.object_id)
    if obj is None or obj.motion_state != "released":
        raise PandaPlanningError("The box must have left belt control before placement")

    robot = simulation.get_robot_states()[pickup.robot_id]
    target = next((bin for bin in simulation.get_targets().values() if bin.preferred_robot_id == pickup.robot_id), None)
    if target is None:
        raise PandaPlanningError("No target bin is assigned to the selected Panda")
    config = simulation.config
    neighbors = tuple(tracked_states_from_ground_truth(truth).values())
    plan = plan_panda_placement(config, robot, pickup.object_id, target, closure, neighbors=neighbors)
    names = list(robot.arm_joint_names)
    for motion in (plan.transfer, plan.descent):
        if not _play_arm(simulation, names, motion, after_step):
            return None

    before = release_into_bin(simulation, robot.gripper_frame, pickup.object_id, target)
    release_time_s = simulation.simulation_time_s

    previous = 0.0
    for elapsed, coordinate in plan.opening[1:]:
        config.getFrame(closure.joint_frame).setJointState([coordinate])
        simulation.step(elapsed - previous)
        previous = elapsed
        if after_step is not None and not after_step(simulation.simulation_time_s):
            return None

    if not _play_arm(simulation, names, plan.retreat, after_step):
        return None
    after = simulation.get_ground_truth()[pickup.object_id]
    error = tcp_pose_error(TcpPose(before.position_m, before.quaternion_wxyz), after.position_m, after.quaternion_wxyz)
    if (config.getFrame(pickup.object_id).getParent().name != "world"
            or error.position_m > 1e-6 or error.orientation_rad > 1e-6
            or dist(simulation.get_robot_states()[pickup.robot_id].finger_joint_positions_m, (closure.initial_m,) * 2) > 1e-6):
        raise PandaPlanningError("The bin release did not remain stable during retreat")
    gap = require_object_in_bin(config, pickup.object_id, target)
    return PandaPlacementResult(target.target_id, release_time_s, gap)
