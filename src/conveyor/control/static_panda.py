"""Replay a validated static Panda trajectory on the simulation clock."""

from math import dist
from typing import TYPE_CHECKING, Callable

from conveyor.planning.static_panda import PoseError, StaticPandaTrajectory, check_static_panda_clearance, tcp_pose_error

if TYPE_CHECKING:
    from conveyor.simulation import ConveyorSimulation


def execute_static_panda(
    simulation: "ConveyorSimulation",
    trajectory: StaticPandaTrajectory,
    *,
    after_step: Callable[[float], bool] | None = None,
) -> PoseError | None:
    """Replay on the same static scene used for validation. A false callback stops execution."""
    if simulation.belt_speed_m_s != 0.0:
        raise ValueError("Static Panda playback requires a stopped conveyor")
    robot = simulation.get_robot_states()[trajectory.robot_id]
    if (robot.arm_joint_names != trajectory.joint_names
            or dist(robot.arm_joint_positions_rad, trajectory.samples[0].positions_rad) > 1e-6):
        raise ValueError("The arm no longer matches the validated start state")
    check_static_panda_clearance(simulation.config, robot, trajectory)
    joint_names = list(trajectory.joint_names)
    previous_time = trajectory.samples[0].elapsed_s
    for sample in trajectory.samples[1:]:
        simulation.config.setJointState(sample.positions_rad, joint_names)
        simulation.step(sample.elapsed_s - previous_time)
        previous_time = sample.elapsed_s
        if after_step is not None and not after_step(sample.elapsed_s):
            return None

    actual = simulation.get_robot_states()[trajectory.robot_id]
    return tcp_pose_error(trajectory.target, actual.gripper_position_m, actual.gripper_quaternion_wxyz)
