"""Close on a moving box, transfer its ownership, and execute a checked lift."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from conveyor.planning.moving_panda import PandaInterception
from conveyor.planning.panda import PandaPlanningError
from conveyor.planning.panda_gripper import FingerClosure, require_bilateral_contact
from conveyor.planning.panda_lift import LIFT_HEIGHT_M, plan_panda_lift
from conveyor.prediction import tracked_states_from_ground_truth

from .moving_panda import PandaInterceptionResult, execute_moving_panda

if TYPE_CHECKING:
    from conveyor.simulation import ConveyorSimulation


@dataclass(frozen=True)
class PandaPickupResult:
    interception: PandaInterceptionResult
    pickup_time_s: float
    lift_height_m: float


def attach_belt_object(simulation: "ConveyorSimulation", gripper_frame: str, object_id: str, closure: FingerClosure) -> None:
    """Transfer ownership at verified contact before advancing the clock."""
    if object_id not in simulation.on_belt_object_ids:
        raise PandaPlanningError(f"{object_id} left the belt before the planned pickup")
    config = simulation.config
    require_bilateral_contact(config, gripper_frame, object_id, closure)
    config.attach(gripper_frame, object_id)
    try:
        simulation.release_object(object_id)
    except (KeyError, RuntimeError, ValueError):
        config.attach("world", object_id)
        raise


def execute_panda_pickup(
    simulation: "ConveyorSimulation",
    plan: PandaInterception,
    *,
    after_step: Callable[[float], bool] | None = None,
) -> PandaPickupResult | None:
    """Lift with a rigid attachment after bilateral geometric contact succeeds."""
    closure = plan.closure
    if closure is None:
        raise ValueError("Pickup requires a moving Panda plan with jaw closure")
    config = simulation.config
    try:
        interception = execute_moving_panda(simulation, plan, after_step=after_step)
        if interception is None:
            return None
        if not interception.success:
            raise PandaPlanningError("Interception did not meet the tolerances for jaw closure")
        robot = simulation.get_robot_states()[plan.robot_id]
        truth = simulation.get_ground_truth(active_only=True)
        obj = truth.get(plan.object_id)
        if obj is None or obj.motion_state != "on_belt":
            raise PandaPlanningError("The box is no longer on the belt")
        tracks = tracked_states_from_ground_truth(truth)
        neighbors = tuple(track for key, track in tracks.items() if key != plan.object_id)
        lift = plan_panda_lift(
            config, robot, plan.object_id, closure, neighbors=neighbors,
            duration_s=1.2 if plan.collision_track is not None else 0.8,
        )

        attach_belt_object(simulation, robot.gripper_frame, plan.object_id, closure)

        pickup_time_s = simulation.simulation_time_s
        previous_elapsed = 0.0
        names = list(plan.joint_names)
        for sample in lift[1:]:
            config.setJointState(sample.positions_rad, names)
            simulation.step(sample.elapsed_s - previous_elapsed)
            previous_elapsed = sample.elapsed_s
            if after_step is not None and not after_step(simulation.simulation_time_s):
                return None

        lifted = simulation.get_ground_truth()[plan.object_id]
        height = lifted.position_m[2] - obj.position_m[2]
        if (height < LIFT_HEIGHT_M - 0.002 or lifted.motion_state != "released"
                or config.getFrame(plan.object_id).getParent().name != robot.gripper_frame):
            raise PandaPlanningError("The box did not stay attached and reach the lift height")
        return PandaPickupResult(interception, pickup_time_s, height)
    finally:
        if simulation.simulation_time_s > closure.start_s + 1e-6 and plan.object_id in simulation.on_belt_object_ids:
            # A cancelled or rejected closure leaves the box under belt control.
            config.getFrame(closure.joint_frame).setJointState([closure.initial_m])
