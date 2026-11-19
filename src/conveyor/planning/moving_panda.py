"""Schedule-driven KOMO interception of one kinematic belt object."""

from dataclasses import dataclass
from math import ceil, dist, isfinite
from typing import TYPE_CHECKING

from conveyor.prediction import ConstantVelocityPredictor
from conveyor.scheduling import Assignment
from conveyor.types import TrackedState, Vector3

from .belt_motion import belt_position, position_belt_frames, position_belt_slices
from .panda import (
    JointSample, PandaPlanningError, TcpPose, add_tcp_orientation, arm_config, moving_collision_pairs,
    new_komo, solve_komo_samples, tcp_pose_error, validate_samples,
)
from .panda_gripper import FingerClosure, fit_simulated_closure, plan_finger_closure, require_bilateral_contact

if TYPE_CHECKING:
    from robotic import Config

    from conveyor.simulation import RobotState


_PREDICTOR = ConstantVelocityPredictor(planar_motion=True, max_horizon_s=60.0)
MAX_RELATIVE_SPEED_M_S = 0.01


@dataclass(frozen=True)
class PandaInterception:
    robot_id: str
    joint_names: tuple[str, ...]
    samples: tuple[JointSample, ...]
    track: TrackedState
    object_quaternion_wxyz: tuple[float, float, float, float]
    planned_at_s: float
    intercept_time_s: float
    descent_start_s: float
    tracking_start_s: float
    grasp_offset_m: Vector3
    gripper_quaternion_wxyz: tuple[float, float, float, float]
    min_clearance_m: float
    closure: FingerClosure | None = None
    neighbors: tuple[TrackedState, ...] = ()
    collision_track: TrackedState | None = None

    @property
    def object_id(self) -> str:
        return self.track.object_id

    @property
    def end_time_s(self) -> float:
        return self.planned_at_s + self.samples[-1].elapsed_s

    def object_position_at(self, simulation_time_s: float) -> Vector3:
        return _PREDICTOR.predict(self.track, simulation_time_s).position_m

    def collision_position_at(self, simulation_time_s: float) -> Vector3:
        if self.collision_track is None:
            return self.object_position_at(simulation_time_s)
        return belt_position(self.collision_track, simulation_time_s)


def _check_path(scratch, robot, samples, object_id, object_position, closure=None, neighbors=()):
    pairs = moving_collision_pairs(scratch, robot.arm_joint_names[0])
    frame = scratch.getFrame(object_id)

    def position_scene(elapsed_s):
        frame.setPosition(object_position(elapsed_s))
        position_belt_frames(scratch.getFrame, neighbors, robot.simulation_time_s + elapsed_s)

    if closure is None:
        clearance, _ = validate_samples(
            scratch, samples, pairs, robot.arm_joint_limits_rad, 0.005,
            position_scene=position_scene,
        )
        return clearance

    # Only the scheduled pad pair may approach contact, and only during closure.
    approach = tuple(sample for sample in samples if robot.simulation_time_s + sample.elapsed_s <= closure.start_s + 1e-9)
    closing = samples[len(approach):]
    clearance, _ = validate_samples(scratch, approach, pairs, robot.arm_joint_limits_rad, 0.005, position_scene=position_scene)

    def position_closure(elapsed_s):
        position_scene(elapsed_s)
        coordinate = closure.coordinate_at(robot.simulation_time_s + elapsed_s)
        scratch.getFrame(closure.joint_frame).setJointState([coordinate])

    contact_clearance, _ = validate_samples(
        scratch, closing, pairs, robot.arm_joint_limits_rad, 0.005,
        position_scene=position_closure, pair_clearances=closure.contact_clearances(object_id),
    )
    require_bilateral_contact(scratch, robot.gripper_frame, object_id, closure)
    return min(clearance, contact_clearance)


def check_moving_panda_clearance(config: "Config", robot: "RobotState", plan: PandaInterception) -> None:
    """Revalidate the sampled path with the current scene and predicted object positions."""
    scratch = arm_config(config, robot)
    _check_path(
        scratch, robot, plan.samples, plan.object_id,
        lambda elapsed: plan.collision_position_at(plan.planned_at_s + elapsed), plan.closure, plan.neighbors,
    )


def plan_moving_panda(
    config: "Config", robot: "RobotState", assignment: Assignment, track: TrackedState,
    *, close_duration_s: float = 0.0, neighbors: tuple[TrackedState, ...] = (),
    collision_track: TrackedState | None = None,
) -> PandaInterception:
    """Approach the scheduled pre-grasp, descend, then track the box until interception."""
    import robotic as ry

    now = robot.simulation_time_s
    if not isfinite(close_duration_s) or close_duration_s < 0.0:
        raise ValueError("Closing duration must be finite and non-negative")
    grasp, intercept = assignment.grasp, assignment.intercept_time_s
    if (not assignment.is_assigned or grasp is None or not grasp.feasible or intercept is None
            or not isfinite(intercept) or assignment.intercept_position_m is None):
        raise PandaPlanningError("A feasible scheduled grasp and finite interception time are required")
    if assignment.robot_id != robot.robot_id or not assignment.object_id == track.object_id == grasp.object_id:
        raise PandaPlanningError("The scheduled robot and object must match the snapshots")
    if assignment.robot_available_at_s is not None and assignment.robot_available_at_s > now + 1e-6:
        raise PandaPlanningError("The assigned robot is busy")
    if assignment.object_exit_time_s is not None and intercept >= assignment.object_exit_time_s:
        raise PandaPlanningError("The planned interception is after the object's pickable interval")
    if assignment.deadline_s is not None and intercept > assignment.deadline_s:
        raise PandaPlanningError("The planned interception is after the assignment deadline")
    duration = intercept - now
    if duration <= 0.65 or track.simulation_time_s > now + 1e-9:
        raise PandaPlanningError("The assignment leaves insufficient time or uses a future-dated track")

    def object_position(elapsed_s):
        return _PREDICTOR.predict(track, now + elapsed_s).position_m

    if collision_track is not None and (collision_track.object_id != track.object_id or collision_track.simulation_time_s > now + 1e-9):
        raise PandaPlanningError("The collision track must identify the same object at the planning time")

    def collision_position(elapsed_s):
        if collision_track is None:
            return object_position(elapsed_s)
        return belt_position(collision_track, now + elapsed_s)

    centroid = object_position(duration)
    if dist(centroid, assignment.intercept_position_m) > 0.001:
        raise PandaPlanningError("The track no longer matches the scheduled interception position")
    scratch = arm_config(config, robot)
    names = set(scratch.getFrameNames())
    if track.object_id not in names:
        raise PandaPlanningError("The tracked object is missing from the scene")
    neighbor_ids = {neighbor.object_id for neighbor in neighbors}
    if track.object_id in neighbor_ids or len(neighbor_ids) != len(neighbors):
        raise PandaPlanningError("Conveyor collision tracks must have distinct object IDs")
    for neighbor in neighbors:
        if (neighbor.simulation_time_s > now + 1e-9 or neighbor.object_id not in names
                or dist(scratch.getFrame(neighbor.object_id).getPosition(), belt_position(neighbor, now)) > 1e-5):
            raise PandaPlanningError(f"Conveyor collision track {neighbor.object_id} does not match the starting scene")
    obj = scratch.getFrame(track.object_id)
    if dist(obj.getPosition(), collision_position(0.0)) > 1e-5:
        raise PandaPlanningError("The object no longer matches the starting scene")
    object_quaternion = tuple(obj.getQuaternion())
    end_offset = tuple(p - c for p, c in zip(grasp.position_m, assignment.intercept_position_m))
    pregrasp_offset = tuple(p - c for p, c in zip(grasp.pregrasp_position_m, assignment.intercept_position_m))

    intercept_slice = ceil(duration / 0.08)
    step = duration / intercept_slice
    tracking_slice = intercept_slice - ceil(0.16 / step)
    approach_slice = tracking_slice - ceil(0.40 / step)
    if approach_slice < 1:
        raise PandaPlanningError("Too late to complete approach and tracking before interception")

    # Keep interception on a knot and round closing up to a whole number of slices.
    slices = intercept_slice + ceil(close_duration_s / step)
    motion_duration = step * slices
    closing = None
    if close_duration_s > 0.0:
        end_time = now + motion_duration
        if assignment.object_exit_time_s is not None and end_time >= assignment.object_exit_time_s:
            raise PandaPlanningError("The object leaves the pickable interval before jaw closure finishes")
        if assignment.deadline_s is not None and end_time > assignment.deadline_s + 1e-9:
            raise PandaPlanningError("The reach window closes before jaw closure finishes")
        margin = 0.0003 if collision_track is None else 0.0001
        closing = plan_finger_closure(robot, grasp, intercept, end_time, gap_margin_m=margin)

    komo = new_komo(scratch, slices, motion_duration)
    position_belt_slices(komo, neighbors, now, motion_duration, slices)
    # Each KOMO configuration holds the collision geometry at its own simulation time.
    for index in range(1, slices + 1):
        komo.getFrame(track.object_id, index / slices).setPosition(collision_position(index * step))
        if closing is not None and index > intercept_slice:
            coordinate = closing.coordinate_at(now + index * step)
            komo.getFrame(closing.joint_frame, index / slices).setJointState([coordinate])
    for index in (approach_slice, *range(tracking_slice, slices + 1)):
        offset = pregrasp_offset if index == approach_slice else end_offset
        target = tuple(p + d for p, d in zip(object_position(index * step), offset))
        komo.addObjective([index / slices], ry.FS.position, [robot.gripper_frame], ry.OT.eq, [10.0], target)
    add_tcp_orientation(komo, [approach_slice / slices], robot.gripper_frame, grasp.quaternion_wxyz)
    add_tcp_orientation(komo, [tracking_slice / slices, 1.0], robot.gripper_frame, grasp.quaternion_wxyz)
    velocity, _ = _PREDICTOR.effective_velocity(track)
    komo.addObjective([intercept_slice / slices], ry.FS.position, [robot.gripper_frame], ry.OT.eq, [10.0], velocity, 1)
    if closing is not None:
        komo.addObjective([1.0], ry.FS.position, [robot.gripper_frame], ry.OT.eq, [10.0], velocity, 1)

    samples, _ = solve_komo_samples(komo, robot.arm_joint_positions_rad, slices, motion_duration, 1.5)
    if closing is not None and collision_track is not None:
        scratch.setJointState(samples[-1].positions_rad)
        obj.setPosition(collision_position(samples[-1].elapsed_s))
        closing = fit_simulated_closure(scratch, robot.gripper_frame, track.object_id, closing)
    clearance = _check_path(scratch, robot, samples, track.object_id, collision_position, closing, neighbors)
    tracking_elapsed = tracking_slice * step
    gripper = scratch.getFrame(robot.gripper_frame)
    for sample in samples:
        if sample.elapsed_s < tracking_elapsed - 1e-9:
            continue
        scratch.setJointState(sample.positions_rad)
        target = TcpPose(
            tuple(p + d for p, d in zip(object_position(sample.elapsed_s), end_offset)),
            grasp.quaternion_wxyz,
        )
        error = tcp_pose_error(target, tuple(gripper.getPosition()), tuple(gripper.getQuaternion()))
        if not error.within_tolerance:
            raise PandaPlanningError(f"TCP tracking error exceeds tolerance at {sample.elapsed_s:.3f} s")

    velocity_events = (duration, motion_duration) if closing is not None else (duration,)
    for elapsed in velocity_events:
        index = next(i for i, sample in enumerate(samples) if abs(sample.elapsed_s - elapsed) <= 1e-6)
        previous, current = samples[index - 1:index + 1]
        scratch.setJointState(previous.positions_rad)
        previous_tcp = tuple(gripper.getPosition())
        scratch.setJointState(current.positions_rad)
        current_tcp = tuple(gripper.getPosition())
        last_step = current.elapsed_s - previous.elapsed_s
        object_delta = tuple(b - a for a, b in zip(object_position(previous.elapsed_s), object_position(current.elapsed_s)))
        tcp_delta = tuple(b - a for a, b in zip(previous_tcp, current_tcp))
        if dist(tcp_delta, object_delta) / last_step > MAX_RELATIVE_SPEED_M_S:
            raise PandaPlanningError("TCP velocity does not match the predicted object at interception or closure")

    return PandaInterception(
        robot.robot_id, robot.arm_joint_names, samples, track, object_quaternion, now, intercept,
        now + approach_slice * step, now + tracking_elapsed, end_offset, grasp.quaternion_wxyz, clearance,
        closing, neighbors, collision_track,
    )
