"""KOMO lift of a held conveyor object, including the carried collision shape."""

from math import isfinite

from conveyor.types import TrackedState

from .belt_motion import position_belt_frames, position_belt_slices
from .panda import (
    PandaPlanningError, TcpPose, add_tcp_orientation, arm_config, moving_collision_pairs,
    new_komo, solve_komo_samples, tcp_pose_error, validate_samples,
)
from .panda_gripper import FingerClosure, require_bilateral_contact


LIFT_HEIGHT_M = 0.12


def plan_panda_lift(
    config, robot, object_id: str, closure: FingerClosure, *,
    neighbors: tuple[TrackedState, ...] = (), duration_s: float = 0.8,
):
    """Plan a vertical lift on a copy with the object attached at its measured pose."""
    import robotic as ry

    if not isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Lift duration must be finite and positive")
    scratch = arm_config(config, robot)
    gripper = robot.gripper_frame
    require_bilateral_contact(scratch, gripper, object_id, closure)
    scratch.attach(gripper, object_id)
    box = scratch.getFrame(object_id)
    initial_box_z = float(box.getPosition()[2])
    pairs = moving_collision_pairs(scratch, robot.arm_joint_names[0])
    allowances = closure.contact_clearances(object_id)
    # At lift start the object rests on the belt. A small numerical tolerance
    # permits that support contact while rejecting downward penetration.
    allowances[frozenset((object_id, "conveyor"))] = -1e-5

    slices = 12
    komo = new_komo(scratch, slices, duration_s)
    position_belt_slices(komo, neighbors, robot.simulation_time_s, duration_s, slices)
    start = robot.gripper_position_m
    target = TcpPose((start[0], start[1], start[2] + LIFT_HEIGHT_M), robot.gripper_quaternion_wxyz)
    komo.addObjective([], ry.FS.position, [gripper], ry.OT.eq, [[10.0, 0, 0], [0, 10.0, 0]], start)
    komo.addObjective([1.0], ry.FS.position, [gripper], ry.OT.eq, [10.0], target.position_m)
    add_tcp_orientation(komo, [], gripper, target.quaternion_wxyz)
    komo.addObjective([1.0], ry.FS.jointState, [], ry.OT.eq, [1.0], [], 1)
    samples, _ = solve_komo_samples(komo, robot.arm_joint_positions_rad, slices, duration_s, 1.5)
    def position_scene(elapsed):
        position_belt_frames(scratch.getFrame, neighbors, robot.simulation_time_s + elapsed)

    validate_samples(
        scratch, samples, pairs, robot.arm_joint_limits_rad, 0.005,
        pair_clearances=allowances, position_scene=position_scene,
    )
    tcp = scratch.getFrame(gripper)
    error = tcp_pose_error(target, tuple(tcp.getPosition()), tuple(tcp.getQuaternion()))
    if not error.within_tolerance or box.getPosition()[2] - initial_box_z < LIFT_HEIGHT_M - 0.002:
        raise PandaPlanningError("The planned lift does not clear the box to the required height")
    return samples
