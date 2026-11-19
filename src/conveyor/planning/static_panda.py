"""Plan and validate one Panda's motion to a static world-frame TCP pose."""

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from .panda import (
    JointSample, PandaPlanningError as StaticPlanningError, PoseError, TcpPose,
    add_tcp_orientation, arm_config, moving_collision_pairs, new_komo, solve_komo_samples, tcp_pose_error, validate_samples,
)

if TYPE_CHECKING:
    from robotic import Config

    from conveyor.simulation import RobotState


@dataclass(frozen=True)
class StaticPandaTrajectory:
    robot_id: str
    joint_names: tuple[str, ...]
    target: TcpPose
    samples: tuple[JointSample, ...]
    endpoint_error: PoseError
    required_clearance_m: float
    min_clearance_m: float
    closest_pair: tuple[str, str]
    peak_joint_speed_rad_s: float


def check_static_panda_clearance(config: "Config", robot: "RobotState", trajectory: StaticPandaTrajectory) -> None:
    """Verify current scene geometry before replaying a previously planned path."""
    scratch = arm_config(config, robot)
    pairs = moving_collision_pairs(scratch, robot.arm_joint_names[0])
    validate_samples(scratch, trajectory.samples, pairs, robot.arm_joint_limits_rad, trajectory.required_clearance_m)


def plan_static_panda(
    config: "Config",
    robot: "RobotState",
    target: TcpPose,
    *,
    duration_s: float = 3.0,
    max_joint_speed_rad_s: float = 1.0,
    clearance_m: float = 0.005,
) -> StaticPandaTrajectory:
    """Solve a static KOMO path and check every interpolated playback sample."""
    import robotic as ry

    if any(not isfinite(value) or value <= 0 for value in (duration_s, max_joint_speed_rad_s, clearance_m)):
        raise ValueError("Duration, joint speed limit and clearance must be finite and positive")
    scratch = arm_config(config, robot)
    initial = tuple(float(q) for q in scratch.getJointState())
    pairs = moving_collision_pairs(scratch, robot.arm_joint_names[0])
    start = (JointSample(0.0, initial),)
    start_clearance = validate_samples(scratch, start, pairs, robot.arm_joint_limits_rad, clearance_m)

    slices = 20
    komo = new_komo(scratch, slices, duration_s)
    komo.addObjective([1.0], ry.FS.position, [robot.gripper_frame], ry.OT.eq, [10.0], target.position_m)
    add_tcp_orientation(komo, [1.0], robot.gripper_frame, target.quaternion_wxyz)
    komo.addObjective([1.0], ry.FS.jointState, [], ry.OT.eq, [1.0], [], 1)
    samples, peak_speed = solve_komo_samples(komo, initial, slices, duration_s, max_joint_speed_rad_s)
    path_clearance = validate_samples(scratch, samples[1:], pairs, robot.arm_joint_limits_rad, clearance_m)
    minimum, closest = min(start_clearance, path_clearance)
    gripper = scratch.getFrame(robot.gripper_frame)
    error = tcp_pose_error(target, tuple(gripper.getPosition()), tuple(gripper.getQuaternion()))
    if not error.within_tolerance:
        raise StaticPlanningError("The path endpoint exceeds the TCP pose tolerance")
    return StaticPandaTrajectory(
        robot.robot_id, robot.arm_joint_names, target, samples, error, clearance_m, minimum, closest, peak_speed,
    )
