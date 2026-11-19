"""Shared RAI helpers for Panda pose, timing, and trajectory validation."""

from dataclasses import dataclass
from math import acos, ceil, dist, fsum, hypot, isclose, isfinite
from typing import TYPE_CHECKING

from conveyor.types import Vector3

if TYPE_CHECKING:
    from robotic import Config

    from conveyor.simulation import RobotState


class PandaPlanningError(RuntimeError):
    """The Panda motion did not meet the planning or validation criteria."""


@dataclass(frozen=True)
class TcpPose:
    position_m: Vector3
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or not all(isfinite(value) for value in self.position_m):
            raise ValueError("TCP position must contain three finite world coordinates")
        if len(self.quaternion_wxyz) != 4 or not isclose(hypot(*self.quaternion_wxyz), 1.0, abs_tol=1e-6):
            raise ValueError("TCP quaternion must be a unit quaternion in (w, x, y, z) order")


@dataclass(frozen=True)
class PoseError:
    position_m: float
    orientation_rad: float

    @property
    def within_tolerance(self) -> bool:
        return self.position_m <= 0.002 and self.orientation_rad <= 0.02


@dataclass(frozen=True)
class JointSample:
    elapsed_s: float
    positions_rad: tuple[float, ...]


def tcp_pose_error(target: TcpPose, position_m: Vector3, quaternion_wxyz: tuple[float, ...]) -> PoseError:
    dot = fsum(a * b for a, b in zip(target.quaternion_wxyz, quaternion_wxyz))
    cosine = abs(dot) / (hypot(*target.quaternion_wxyz) * hypot(*quaternion_wxyz))
    return PoseError(dist(target.position_m, position_m), 2.0 * acos(min(1.0, cosine)))


def arm_config(config: "Config", robot: "RobotState") -> "Config":
    import robotic as ry

    result = ry.Config()
    # Keep the native frame array one-dimensional during a copy in RAI 0.4.1.
    anchor = "_panda_planning_anchor"
    names = set(config.getFrameNames())
    while anchor in names:
        anchor += "_"
    result.addFrame(anchor)
    result.addConfigurationCopy(config)
    result.delFrame(anchor)
    result.selectJoints(list(robot.arm_joint_names))
    if tuple(result.getJointNames()) != robot.arm_joint_names:
        raise PandaPlanningError("RAI joint order differs from the robot snapshot")
    if dist(result.getJointState(), robot.arm_joint_positions_rad) > 1e-6:
        raise PandaPlanningError("The robot snapshot differs from the scene")
    return result


def moving_collision_pairs(config: "Config", first_joint: str) -> tuple[tuple[str, str], ...]:
    def moves_with_arm(name: str) -> bool:
        frame = config.getFrame(name)
        while frame is not None:
            if frame.name == first_joint:
                return True
            frame = frame.getParent()
        return False

    names = config.getCollidablePairs()
    moving_frames = {name for name in set(names) if moves_with_arm(name)}
    pairs = tuple((a, b) for a, b in zip(names[::2], names[1::2]) if a in moving_frames or b in moving_frames)
    if not pairs:
        raise PandaPlanningError("No collision pairs were found for the arm")
    return pairs


def trajectory_samples(start, path, durations, max_step_s: float, max_speed_rad_s: float):
    elapsed = 0.0
    previous = tuple(float(q) for q in start)
    samples = [JointSample(elapsed, previous)]
    peak_speed = 0.0
    for next_q, duration in zip(path, durations):
        q = tuple(float(value) for value in next_q)
        if not isfinite(duration) or duration <= 0 or not all(isfinite(value) for value in q):
            raise PandaPlanningError("KOMO returned an invalid joint path")
        delta = tuple(b - a for a, b in zip(previous, q))
        speed = max(abs(value) for value in delta) / duration
        if speed > max_speed_rad_s:
            raise PandaPlanningError(f"Joint speed {speed:.3f} rad/s exceeds the playback limit")
        peak_speed = max(peak_speed, speed)
        steps = ceil(duration / max_step_s)
        for step in range(1, steps + 1):
            fraction = step / steps
            position = tuple(value + fraction * change for value, change in zip(previous, delta))
            samples.append(JointSample(float(elapsed + fraction * duration), position))
        elapsed += duration
        previous = q
    return tuple(samples), float(peak_speed)


def validate_samples(config, samples, pairs, joint_limits, clearance_m, *, position_scene=None, pair_clearances=None):
    import numpy as np

    from .clearance import PairDistances

    distances = PairDistances(config, pairs)
    allowances = pair_clearances or {}
    thresholds = np.array([allowances.get(frozenset(pair), clearance_m) for pair in pairs])
    closest = (float("inf"), pairs[0])
    for sample in samples:
        if position_scene is not None:
            position_scene(sample.elapsed_s)
        if any(not lower - 1e-8 <= q <= upper + 1e-8 for q, (lower, upper) in zip(sample.positions_rad, joint_limits)):
            raise PandaPlanningError(f"Joint limits violated at {sample.elapsed_s:.3f} s")
        config.setJointState(sample.positions_rad)
        for index, distance in distances.candidate_distances(np.maximum(thresholds, closest[0])):
            if not isfinite(distance) or distance < thresholds[index]:
                first, second = pairs[index]
                raise PandaPlanningError(
                    f"Insufficient clearance between {first} and {second} at {sample.elapsed_s:.3f} s"
                )
            if distance < closest[0]:
                closest = distance, pairs[index]
    return closest


def new_komo(config: "Config", slices: int, duration_s: float):
    import robotic as ry

    # RAI 0.4.1 constructs the time grid at setConfig().
    komo = ry.KOMO()
    komo.setTiming(1, slices, duration_s, 2)
    komo.setConfig(config, True)
    komo.addControlObjective([], 1, 0.1)
    komo.addControlObjective([], 2, 1.0)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [10.0])
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [10.0])
    return komo


def add_tcp_orientation(komo, times, gripper: str, quaternion_wxyz):
    import robotic as ry

    w, x, y, z = quaternion_wxyz
    axis_x = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    axis_z = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    komo.addObjective(times, ry.FS.vectorX, [gripper], ry.OT.eq, [10.0], axis_x)
    komo.addObjective(times, ry.FS.vectorZ, [gripper], ry.OT.eq, [10.0], axis_z)


def solve_komo_samples(komo, initial, slices: int, duration_s: float, max_speed_rad_s: float):
    import robotic as ry

    solved = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    if not solved.feasible:
        raise PandaPlanningError(f"KOMO did not find a feasible path (eq={solved.eq:.3g}, ineq={solved.ineq:.3g})")
    path, durations = komo.getPath(), komo.getPathTau()
    if path.shape != (slices, len(initial)) or durations.shape != (slices,) or not isclose(fsum(durations), duration_s):
        raise PandaPlanningError("KOMO returned an unexpected path or time grid")
    return trajectory_samples(initial, path, durations, 0.02, max_speed_rad_s)


