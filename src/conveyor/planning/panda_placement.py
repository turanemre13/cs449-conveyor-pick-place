"""Plan a collision-checked transfer, release, and retreat at an authored bin."""

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin

from conveyor.types import TrackedState

from .belt_motion import position_belt_frames, position_belt_slices
from .panda import (
    JointSample, PandaPlanningError, TcpPose, add_tcp_orientation, arm_config,
    moving_collision_pairs, new_komo, solve_komo_samples, tcp_pose_error, validate_samples,
)
from .panda_gripper import FingerClosure, require_bilateral_contact


BIN_FLOOR_TOLERANCE_M = 0.001
_FLOOR_PENETRATION_TOLERANCE_M = 0.0001
_MIN_CLEARANCE_M = 0.005
_OPENING_STEPS = 15
_OPENING_STEP_S = 0.02


@dataclass(frozen=True)
class PlacementPath:
    transfer: tuple[JointSample, ...]
    descent: tuple[JointSample, ...]
    opening: tuple[tuple[float, float], ...]
    retreat: tuple[JointSample, ...]


def _object_half_extent(frame, rotation):
    import numpy as np
    import robotic as ry

    size = frame.getSize()
    shape = frame.getShapeType()
    if shape in (ry.ST.box, ry.ST.ssBox):
        return np.abs(rotation) @ (size[:3] / 2)
    if shape == ry.ST.cylinder:
        height, radius = size
        return np.abs(rotation[:, 2]) * height / 2 + radius * np.linalg.norm(rotation[:, :2], axis=1)
    raise PandaPlanningError("Bin placement requires a box or cylinder collision shape")


def _object_extents(config, object_id, target):
    obj = config.getFrame(object_id)
    transform = obj.getTransform(config.getFrame(target.bin_frame))
    center = transform[:3, 3]
    extent = _object_half_extent(obj, transform[:3, :3])
    return center - extent, center + extent


def require_object_in_bin(config, object_id, target) -> float:
    """Require the entire object to lie within the bin and reach its floor."""
    import robotic as ry

    lower, upper = _object_extents(config, object_id, target)
    width, depth, height = target.inner_dimensions_m
    for axis, span in enumerate((width, depth)):
        if max(-lower[axis], upper[axis]) > span / 2 - _MIN_CLEARANCE_M:
            raise PandaPlanningError("The object footprint does not fit inside the target bin")
    if lower[2] < -_FLOOR_PENETRATION_TOLERANCE_M or upper[2] > height:
        raise PandaPlanningError("The object is outside the target bin's vertical extent")
    floor = f"{target.bin_frame}_bottom"
    gap = -float(config.eval(ry.FS.negDistance, [object_id, floor])[0][0])
    if not isfinite(gap) or not -_FLOOR_PENETRATION_TOLERANCE_M <= gap <= BIN_FLOOR_TOLERANCE_M:
        raise PandaPlanningError("The object did not reach the target bin floor")
    return gap


def _placement_poses(config, robot, object_id, target, bin_offset_m, align_bin_axes):
    import numpy as np

    bin_frame = config.getFrame(target.bin_frame)
    bin_rotation = bin_frame.getRotationMatrix()
    if not np.allclose(bin_rotation[:, 2], (0, 0, 1), atol=1e-6):
        raise PandaPlanningError("Placement requires an upright target bin")
    hand = config.getFrame(robot.gripper_frame)
    obj = config.getFrame(object_id)
    relative = obj.getTransform(hand)
    axis = 0 if target.inner_dimensions_m[0] >= target.inner_dimensions_m[1] else 1
    jaw_axis = bin_rotation[:, axis]
    if float(jaw_axis @ hand.getRotationMatrix()[:, 0]) < -0.001:
        jaw_axis = -jaw_axis
    if not align_bin_axes:
        jaw_axis = hand.getRotationMatrix()[:, 0]
    yaw = atan2(jaw_axis[1], jaw_axis[0])
    rotation = np.array(((cos(yaw), -sin(yaw), 0), (sin(yaw), cos(yaw), 0), (0, 0, 1)))
    quaternion = (cos(yaw / 2), 0.0, 0.0, sin(yaw / 2))
    offset = rotation @ relative[:3, 3]
    object_rotation = bin_rotation.T @ rotation @ relative[:3, :3]
    extent = _object_half_extent(obj, object_rotation)
    width, depth, height = target.inner_dimensions_m
    x, y = bin_offset_m
    if (abs(x) + extent[0] + _MIN_CLEARANCE_M > width / 2
            or abs(y) + extent[1] + _MIN_CLEARANCE_M > depth / 2 or 2 * extent[2] > height):
        raise PandaPlanningError("The held object does not fit in the target bin at the placement orientation")

    release = bin_frame.getPosition() + bin_rotation @ (x, y, 0.0) - offset
    release[2] += extent[2] + 0.0003
    above = release.copy()
    above[2] = max(target.approach_position_m[2], release[2] + height + 0.04)
    return TcpPose(tuple(above), quaternion), TcpPose(tuple(release), quaternion)


def _plan_motion(scratch, robot, pose, duration, slices, *, vertical, tracks, start_s, pair_clearances=None):
    import robotic as ry

    gripper = robot.gripper_frame
    komo = new_komo(scratch, slices, duration)
    position_belt_slices(komo, tracks, start_s, duration, slices)
    komo.addObjective([1.0], ry.FS.position, [gripper], ry.OT.eq, [10.0], pose.position_m)
    if vertical:
        komo.addObjective([], ry.FS.position, [gripper], ry.OT.eq, [[10, 0, 0], [0, 10, 0]], pose.position_m)
    else:
        komo.addObjective([], ry.FS.vectorZ, [gripper], ry.OT.eq, [10.0], (0.0, 0.0, 1.0))
    add_tcp_orientation(komo, [] if vertical else [1.0], gripper, pose.quaternion_wxyz)
    komo.addObjective([1.0], ry.FS.jointState, [], ry.OT.eq, [1.0], [], 1)
    samples, _ = solve_komo_samples(komo, scratch.getJointState(), slices, duration, 1.5)
    pairs = moving_collision_pairs(scratch, robot.arm_joint_names[0])
    validate_samples(
        scratch, samples, pairs, robot.arm_joint_limits_rad, _MIN_CLEARANCE_M, pair_clearances=pair_clearances,
        position_scene=lambda elapsed: position_belt_frames(scratch.getFrame, tracks, start_s + elapsed),
    )
    tcp = scratch.getFrame(gripper)
    error = tcp_pose_error(pose, tuple(tcp.getPosition()), tuple(tcp.getQuaternion()))
    if not error.within_tolerance:
        raise PandaPlanningError("The placement motion did not reach its planned TCP pose")
    return samples


def plan_panda_placement(
    config, robot, object_id: str, target, closure: FingerClosure, *, neighbors: tuple[TrackedState, ...] = (),
    bin_offset_m: tuple[float, float] = (0.0, 0.0),
    align_bin_axes: bool = True,
) -> PlacementPath:
    """Check the entire placement on a copy before changing live ownership."""
    scratch = arm_config(config, robot)
    parent = scratch.getFrame(object_id).getParent()
    if parent is None or parent.name != robot.gripper_frame:
        raise PandaPlanningError("The object must be attached to the selected gripper")
    require_bilateral_contact(scratch, robot.gripper_frame, object_id, closure)
    if len(bin_offset_m) != 2 or not all(isfinite(value) for value in bin_offset_m):
        raise ValueError("Bin offset must contain two finite coordinates")
    above, release = _placement_poses(scratch, robot, object_id, target, bin_offset_m, align_bin_axes)
    pad_allowances = closure.contact_clearances(object_id)
    transfer = _plan_motion(
        scratch, robot, above, 4.0, 40, vertical=False, tracks=neighbors,
        start_s=robot.simulation_time_s, pair_clearances=pad_allowances,
    )
    descent_allowances = {**pad_allowances, frozenset((object_id, f"{target.bin_frame}_bottom")): -_FLOOR_PENETRATION_TOLERANCE_M}
    descent_start_s = robot.simulation_time_s + transfer[-1].elapsed_s
    descent = _plan_motion(
        scratch, robot, release, 1.6, 20, vertical=True, tracks=neighbors,
        start_s=descent_start_s, pair_clearances=descent_allowances,
    )
    require_object_in_bin(scratch, object_id, target)

    scratch.attach("world", object_id)
    opening = tuple(
        (step * _OPENING_STEP_S, closure.closed_m + (closure.initial_m - closure.closed_m) * step / _OPENING_STEPS)
        for step in range(_OPENING_STEPS + 1)
    )
    coordinates = dict(opening)
    joints = tuple(scratch.getJointState())
    opening_start_s = descent_start_s + descent[-1].elapsed_s

    def set_opening(elapsed):
        position_belt_frames(scratch.getFrame, neighbors, opening_start_s + elapsed)
        scratch.getFrame(closure.joint_frame).setJointState([coordinates[elapsed]])

    validate_samples(
        scratch, (JointSample(elapsed, joints) for elapsed, _ in opening),
        moving_collision_pairs(scratch, robot.arm_joint_names[0]), robot.arm_joint_limits_rad, _MIN_CLEARANCE_M,
        position_scene=set_opening, pair_clearances=pad_allowances,
    )
    retreat = _plan_motion(
        scratch, robot, above, 1.3, 20, vertical=True, tracks=neighbors, start_s=opening_start_s + opening[-1][0],
    )
    require_object_in_bin(scratch, object_id, target)
    return PlacementPath(transfer, descent, opening, retreat)
