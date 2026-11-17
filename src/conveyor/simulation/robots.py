"""Immutable world-frame robot snapshots, without controller or backend imports."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import robotic as ry


@dataclass(frozen=True)
class RobotState:
    """Observed scene state, in metres/radians and simulation seconds.

    Arm tuples follow ``arm_joint_names`` in joint-1 through joint-7 order.
    Finger values are the two model prismatic joint coordinates in metres,
    including the mimic joint; they are not a measured jaw gap or grasp state.
    Availability, commanded motion and velocities belong to the controller and
    are deliberately absent rather than inferred from a stationary scene.
    """

    robot_id: str
    simulation_time_s: float
    base_frame: str
    gripper_frame: str
    base_position_m: tuple[float, float, float]
    base_quaternion_wxyz: tuple[float, float, float, float]
    gripper_position_m: tuple[float, float, float]
    gripper_quaternion_wxyz: tuple[float, float, float, float]
    arm_joint_names: tuple[str, ...]
    arm_joint_positions_rad: tuple[float, ...]
    arm_joint_limits_rad: tuple[tuple[float, float], ...]
    finger_joint_names: tuple[str, str]
    finger_joint_positions_m: tuple[float, float]
    frame: ClassVar[str] = "world"


def _finite_values(name: str, values, count: int) -> tuple[float, ...]:
    result = tuple(values)
    if len(result) != count or any(
        isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
        for value in result
    ):
        raise ValueError(f"{name} must contain {count} finite real numbers")
    return tuple(float(value) for value in result)


def get_robot_snapshots(config: "ry.Config", simulation_time_s: float) -> dict[str, RobotState]:
    """Read the two named Pandas without mutating joint selection or the scene.

    Native active joint state may also contain unrelated joints; arm values are
    resolved by their exact names rather than by their array positions. The
    supported RAI 0.4.1 backend returns limits as two rows (lower, upper).
    Missing robot frames or inactive/missing arm joints raise ``ValueError``.
    """
    if (
        isinstance(simulation_time_s, bool)
        or not isinstance(simulation_time_s, Real)
        or not isfinite(simulation_time_s)
        or simulation_time_s < 0
    ):
        raise ValueError("simulation_time_s must be finite and non-negative")

    robots = (("robot_a", "a_"), ("robot_b", "b_"))
    required = {
        f"{prefix}{suffix}"
        for _, prefix in robots
        for suffix in (
            "panda_base", "gripper", "panda_finger_joint1", "panda_finger_joint2",
            *(f"panda_joint{index}" for index in range(1, 8)),
        )
    }
    missing_frames = required - set(config.getFrameNames())
    if missing_frames:
        raise ValueError(f"Robot snapshots require missing frames: {', '.join(sorted(missing_frames))}")

    joint_names = tuple(config.getJointNames())
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Robot snapshots require unique active joint names")
    joint_indices = {name: index for index, name in enumerate(joint_names)}
    required_joints = {
        f"{prefix}panda_joint{index}" for _, prefix in robots for index in range(1, 8)
    }
    missing_joints = required_joints - set(joint_names)
    if missing_joints:
        raise ValueError(f"Robot arm joints must be active: {', '.join(sorted(missing_joints))}")
    joint_state = _finite_values("active joint state", config.getJointState(), len(joint_names))
    limits = config.getJointLimits()
    if len(limits) != 2 or any(len(row) != len(joint_names) for row in limits):
        raise ValueError("Expected RAI joint limits with two rows: lower and upper")

    result = {}
    for robot_id, prefix in robots:
        base_name, gripper_name = f"{prefix}panda_base", f"{prefix}gripper"
        base, gripper = config.getFrame(base_name), config.getFrame(gripper_name)
        arm_names = tuple(f"{prefix}panda_joint{index}" for index in range(1, 8))
        arm_indices = tuple(joint_indices[name] for name in arm_names)
        arm_limits = tuple(
            _finite_values(f"limits for {name}", (limits[0][index], limits[1][index]), 2)
            for name, index in zip(arm_names, arm_indices)
        )
        if any(lower > upper for lower, upper in arm_limits):
            raise ValueError(f"{robot_id} arm joints require defined lower/upper limits")
        finger_names = (f"{prefix}panda_finger_joint1", f"{prefix}panda_finger_joint2")
        finger_values = tuple(
            _finite_values(f"joint state for {name}", config.getFrame(name).getJointState(), 1)[0]
            for name in finger_names
        )
        result[robot_id] = RobotState(
            robot_id=robot_id, simulation_time_s=float(simulation_time_s),
            base_frame=base_name, gripper_frame=gripper_name,
            base_position_m=_finite_values(base_name + " position", base.getPosition(), 3),
            base_quaternion_wxyz=_finite_values(base_name + " quaternion", base.getQuaternion(), 4),
            gripper_position_m=_finite_values(gripper_name + " position", gripper.getPosition(), 3),
            gripper_quaternion_wxyz=_finite_values(gripper_name + " quaternion", gripper.getQuaternion(), 4),
            arm_joint_names=arm_names,
            arm_joint_positions_rad=tuple(joint_state[index] for index in arm_indices),
            arm_joint_limits_rad=arm_limits,
            finger_joint_names=finger_names, finger_joint_positions_m=finger_values,
        )
    return result
