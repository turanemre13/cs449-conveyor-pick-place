"""Play a scheduled Panda interception and measure it against the live belt."""

from dataclasses import dataclass, replace
from math import dist
from typing import TYPE_CHECKING, Callable

from conveyor.planning.belt_motion import belt_position
from conveyor.planning.free_tcp import TcpSample
from conveyor.planning.moving_panda import MAX_RELATIVE_SPEED_M_S, PandaInterception, check_moving_panda_clearance
from conveyor.planning.panda import PandaPlanningError, TcpPose, tcp_pose_error
from conveyor.planning.panda_gripper import require_bilateral_contact
from conveyor.simulation.scenarios import planar_half_extents

from .free_tcp import InterceptionResult, evaluate_interception

if TYPE_CHECKING:
    from conveyor.simulation import ConveyorSimulation


@dataclass(frozen=True)
class PandaInterceptionResult:
    alignment: InterceptionResult
    orientation_error_rad: float

    @property
    def success(self) -> bool:
        return self.alignment.success and self.orientation_error_rad <= 0.02


def _measure_interception(simulation, plan, previous, current):
    actual = simulation.get_robot_states()[plan.robot_id]
    obj = simulation.get_ground_truth(active_only=True).get(plan.object_id)
    alignment = evaluate_interception(plan, previous, current, obj, velocity_tolerance_m_s=MAX_RELATIVE_SPEED_M_S)
    target = TcpPose(current.position_m, plan.gripper_quaternion_wxyz)
    orientation = tcp_pose_error(target, current.position_m, actual.gripper_quaternion_wxyz).orientation_rad
    return PandaInterceptionResult(alignment, orientation)


def execute_moving_panda(
    simulation: "ConveyorSimulation",
    plan: PandaInterception,
    *,
    after_step: Callable[[float], bool] | None = None,
) -> PandaInterceptionResult | None:
    """Play the verified samples while the object moves under belt control."""
    if simulation.exit_mode != "remove":
        raise ValueError("Moving Panda playback requires remove-at-exit mode")
    now = simulation.simulation_time_s
    if abs(now - plan.planned_at_s) > 1e-6:
        raise ValueError("The plan no longer starts at the current simulation time")
    belt_ids = {plan.object_id, *(neighbor.object_id for neighbor in plan.neighbors)}
    if belt_ids != set(simulation.on_belt_object_ids):
        raise ValueError("The on-belt objects no longer match the validated collision scene")
    robot = simulation.get_robot_states()[plan.robot_id]
    if robot.arm_joint_names != plan.joint_names or dist(robot.arm_joint_positions_rad, plan.samples[0].positions_rad) > 1e-6:
        raise ValueError("The arm no longer matches the validated start state")
    closing = plan.closure
    if closing is not None and any(abs(q - closing.initial_m) > 1e-6 for q in robot.finger_joint_positions_m):
        raise ValueError("The jaws no longer match the validated opening")
    truth = simulation.get_ground_truth(active_only=True)
    obj = truth[plan.object_id]
    for neighbor in plan.neighbors:
        actual = truth[neighbor.object_id]
        if (actual.velocity_m_s is None or dist(actual.velocity_m_s, neighbor.velocity_m_s) > 1e-5
                or dist(actual.position_m, belt_position(neighbor, now)) > 1e-5):
            raise ValueError(f"Conveyor collision track {neighbor.object_id} no longer matches the scene")
    initial = plan.collision_position_at(now)
    final = plan.collision_position_at(plan.end_time_s)
    expected_velocity = tuple((b - a) / (plan.end_time_s - now) for a, b in zip(initial, final))
    object_pose = TcpPose(initial, plan.object_quaternion_wxyz)
    object_error = tcp_pose_error(object_pose, obj.position_m, obj.quaternion_wxyz)
    if (obj.motion_state != "on_belt" or obj.velocity_m_s is None or object_error.position_m > 1e-5
            or object_error.orientation_rad > 1e-5 or dist(obj.velocity_m_s, expected_velocity) > 1e-5):
        raise ValueError("The actual object state no longer matches the plan")
    belt = simulation.config.getFrame("conveyor")
    spec = next(spec for spec in simulation.object_specs if spec.object_id == plan.object_id)
    if final[0] + planar_half_extents(spec)[0] >= belt.getPosition()[0] + belt.getSize()[0] / 2 - 1e-9:
        raise ValueError("The object would leave the physical belt before interception")
    del belt
    check_moving_panda_clearance(simulation.config, robot, plan)

    names = list(plan.joint_names)
    last_elapsed = 0.0
    previous = TcpSample(now, robot.gripper_position_m)
    current = previous
    interception = None
    for sample in plan.samples[1:]:
        simulation.config.setJointState(sample.positions_rad, names)
        if closing is not None and interception is not None:
            coordinate = closing.coordinate_at(now + sample.elapsed_s)
            simulation.config.getFrame(closing.joint_frame).setJointState([coordinate])
        simulation.step(sample.elapsed_s - last_elapsed)
        last_elapsed = sample.elapsed_s
        previous = current
        position = tuple(simulation.config.getFrame(robot.gripper_frame).getPosition())
        current = TcpSample(simulation.simulation_time_s, position)
        if after_step is not None and not after_step(current.simulation_time_s):
            return None
        if interception is None and current.simulation_time_s >= plan.intercept_time_s - 1e-6:
            interception = _measure_interception(simulation, plan, previous, current)
            if not interception.success:
                return interception

    if closing is not None:
        contact_event = replace(plan, intercept_time_s=plan.end_time_s)
        result = _measure_interception(simulation, contact_event, previous, current)
        if not result.success:
            raise PandaPlanningError("The box and TCP lost alignment during jaw closure")
        require_bilateral_contact(simulation.config, robot.gripper_frame, plan.object_id, closing)
    return interception
