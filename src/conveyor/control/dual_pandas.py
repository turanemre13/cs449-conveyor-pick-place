"""Prepare and validate a shared Panda timeline before starting live playback."""

from bisect import bisect_right
from dataclasses import dataclass
from itertools import islice
from math import dist, isfinite
from typing import Callable

from conveyor.planning.clearance import PairDistances
from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.planning.panda import PandaPlanningError, moving_collision_pairs
from conveyor.planning.panda_placement import require_object_in_bin
from conveyor.prediction import ConstantVelocityPredictor
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, SchedulableObject, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)
from conveyor.simulation import ConveyorSimulation

from .panda_pickup import attach_belt_object, execute_panda_pickup
from .panda_placement import execute_panda_placement, release_into_bin
from .rgbd_panda import CameraPickupPlan, CameraPickupPlanner, ConveyorCameraTracker, plan_camera_pickups
from .serial_pandas import PandaJobResult, execute_serial_pandas


@dataclass(frozen=True)
class ArmCommand:
    joints_rad: tuple[float, ...]
    finger_m: float


@dataclass(frozen=True)
class PlaybackFrame:
    time_s: float
    commands: tuple[ArmCommand, ...]


@dataclass(frozen=True)
class PreparedPandaBatch:
    robot_ids: tuple[str, ...]
    frames: tuple[PlaybackFrame, ...]
    jobs: tuple[PandaJobResult, ...]
    start_scene: tuple
    overlap_s: float
    camera_plans: tuple[CameraPickupPlan, ...] = ()
    candidate_retries: int = 0


@dataclass(frozen=True)
class CameraBatchPlayback:
    jobs: tuple[PandaJobResult, ...]
    camera_frames: int
    replanned_at_s: tuple[float, ...]
    candidate_retries: int = 0


_PREDICTOR = ConstantVelocityPredictor(planar_motion=True, max_horizon_s=60.0)
_REPLAN_POSITION_TOLERANCE_M = 0.005
_APPROACH_CAMERA_PERIOD_S = 0.20
_MAX_MISSED_APPROACH_FRAMES = 2
_MAX_REPLACEMENT_RETRIES = 6


def _pair_names(config):
    flat = config.getCollidablePairs()
    return tuple(zip(flat[::2], flat[1::2]))


def _start_scene(simulation):
    config = simulation.config
    pairs = frozenset(frozenset(pair) for pair in _pair_names(config))
    names = sorted({name for pair in pairs for name in pair})
    shapes = []
    for name in names:
        frame = config.getFrame(name)
        pose = tuple(round(float(value), 9) for value in frame.getPose())
        shapes.append((name, frame.getShapeType(), tuple(frame.getSize()), pose))
    return simulation.exit_mode, simulation.belt_speed_m_s, simulation.object_specs, pairs, tuple(shapes)


def _preview_scene(simulation):
    return ConveyorSimulation(
        simulation.belt_speed_m_s, scenario=simulation.scenario, seed=simulation.seed,
        object_count=simulation.object_count, exit_mode=simulation.exit_mode,
    )


def _restore_preview(preview, source):
    preview.reset()
    now = source.simulation_time_s
    if now > 0.0:
        # The source may have changed speed or pose since time zero. Restore
        # its current belt and arm state without advancing objects at the new speed.
        preview.belt_speed_m_s = 0.0
        preview.step(now)
        preview.belt_speed_m_s = source.belt_speed_m_s
        preview.config.setJointState(source.config.getJointState())
        for object_id in source.on_belt_object_ids:
            preview.config.getFrame(object_id).setPose(source.config.getFrame(object_id).getPose())


def _record(scene, streams, robot_ids, time_s):
    states = scene.get_robot_states()
    for robot_id in robot_ids:
        state = states[robot_id]
        streams[robot_id][round(time_s, 9)] = ArmCommand(state.arm_joint_positions_rad, state.finger_joint_positions_m[0])
    return True


def _record_serial(scene, streams):
    result = execute_serial_pandas(scene, after_step=lambda robot, phase, time: _record(scene, streams, streams, time))
    if result.unplaced:
        raise PandaPlanningError(f"Serial batch left objects unplaced: {result.unplaced}")
    return result.placed


def _ground_truth_plans(scene):
    robots = scene.get_robot_states()
    objects = schedulable_objects_from_ground_truth(scene.get_ground_truth(active_only=True))
    scheduler = DeadlineAwareGreedyScheduler()
    candidates = robot_configs_from_snapshots(robots, scene.get_targets())
    assigned = scheduler.schedule(objects, candidates, scene.simulation_time_s).assigned
    if len(assigned) != 2 or {item.robot_id for item in assigned} != set(robots):
        raise PandaPlanningError("The concurrent demo needs one scheduler assignment per Panda")

    tracks = {item.object_id: item.state for item in objects}
    plans = []
    for assignment in assigned:
        robot_id, object_id = assignment.robot_id, assignment.object_id
        plan = plan_moving_panda(
            scene.config, robots[robot_id], assignment, tracks[object_id],
            close_duration_s=scheduler.config.timing.gripper_close_s,
            neighbors=tuple(track for key, track in tracks.items() if key != object_id),
        )
        plans.append((None, plan))
    return plans


def _record_concurrent(scene, streams, plans, restore_scene):
    jobs = []
    for track_id, plan in plans:
        restore_scene()
        if plan.planned_at_s > scene.simulation_time_s:
            scene.step(plan.planned_at_s - scene.simulation_time_s)
        robot_id = plan.robot_id

        def record(time_s):
            return _record(scene, streams, (robot_id,), time_s)

        pickup = execute_panda_pickup(scene, plan, after_step=record)
        placement = execute_panda_placement(scene, plan, after_step=record)
        jobs.append(PandaJobResult(
            robot_id, plan.object_id, plan.planned_at_s, plan.intercept_time_s, scene.simulation_time_s, placement,
            pickup.pickup_time_s, plan.closure, track_id,
        ))
    return tuple(jobs)


def _interpolate(times, commands, now):
    right = bisect_right(times, now)
    before = commands[right - 1]
    if right == len(times) or times[right - 1] == now:
        return before
    after = commands[right]
    ratio = (now - times[right - 1]) / (times[right] - times[right - 1])
    joints = tuple(a + (b - a) * ratio for a, b in zip(before.joints_rad, after.joints_rad))
    finger = before.finger_m + (after.finger_m - before.finger_m) * ratio
    return ArmCommand(joints, finger)


def _merge_streams(streams, jobs, extra_times=()):
    times = {time for stream in streams.values() for time in stream}
    times.update(extra_times)
    times.update(round(time, 9) for job in jobs for time in (job.pickup_time_s, job.placement.release_time_s))
    segments = [tuple(zip(*sorted(stream.items()))) for stream in streams.values()]
    frames = []
    overlap = 0.0
    for time in sorted(times):
        commands = tuple(_interpolate(keys, values, time) for keys, values in segments)
        if frames:
            previous = frames[-1]
            if all(dist(a.joints_rad, b.joints_rad) > 1e-9 for a, b in zip(previous.commands, commands)):
                overlap += time - previous.time_s
        frames.append(PlaybackFrame(time, commands))
    return tuple(frames), overlap


def _apply_event(scene, robot, job, picking, targets):
    if picking:
        attach_belt_object(scene, robot.gripper_frame, job.object_id, job.closure)
    else:
        release_into_bin(scene, robot.gripper_frame, job.object_id, targets[job.placement.target_id])


def _clearance_limits(targets, jobs, now):
    limits = {}
    for job in jobs:
        if job.intercepted_at_s - 1e-8 <= now <= job.placement.release_time_s + 0.3 + 1e-8:
            limits.update(job.closure.contact_clearances(job.object_id))
        limits[frozenset((job.object_id, "conveyor"))] = -0.00001
        floor = targets[job.placement.target_id].bin_frame + "_bottom"
        limits[frozenset((job.object_id, floor))] = -0.0001
    return limits


def _validate_clearance(distances, targets, jobs, now):
    allowances = _clearance_limits(targets, jobs, now)
    limits = [allowances.get(pair, 0.005) for pair in distances.pair_keys]
    for index, separation in distances.candidate_distances(limits):
        if not isfinite(separation) or separation < limits[index]:
            first, second = distances.pairs[index]
            raise PandaPlanningError(f"Combined clearance failed at {now:.3f} s: {first} / {second}")


def _play(scene, batch, *, check_clearance=False, after_step=None):
    config = scene.config
    robots = scene.get_robot_states()
    robot_states = tuple(robots[robot_id] for robot_id in batch.robot_ids)
    targets = scene.get_targets()
    joint_names = [name for robot in robot_states for name in robot.arm_joint_names]
    events: dict[float, list[tuple[PandaJobResult, bool]]] = {}
    for job in batch.jobs:
        events.setdefault(round(job.pickup_time_s, 9), []).append((job, True))
        events.setdefault(round(job.placement.release_time_s, 9), []).append((job, False))
    distances = None
    previous_topology = ()
    previous = scene.simulation_time_s

    def current_distances():
        pairs = dict.fromkeys(
            pair for robot in robot_states for pair in moving_collision_pairs(config, robot.arm_joint_names[0])
        )
        return PairDistances(config, pairs)

    for frame in batch.frames:
        if frame.time_s > previous + 1e-10:
            config.setJointState([value for command in frame.commands for value in command.joints_rad], joint_names)
            for robot, command in zip(robot_states, frame.commands):
                config.getFrame(robot.finger_joint_names[0]).setJointState([command.finger_m])
            scene.step(frame.time_s - previous)
            previous = frame.time_s
        due = events.get(frame.time_s, ())
        if check_clearance:
            topology = scene.active_object_ids
            if distances is None or topology != previous_topology:
                distances = current_distances()
                previous_topology = topology
            _validate_clearance(distances, targets, batch.jobs, frame.time_s)
        for job, picking in due:
            _apply_event(scene, robots[job.robot_id], job, picking, targets)
        if check_clearance and due:
            distances = current_distances()
            _validate_clearance(distances, targets, batch.jobs, frame.time_s)
        if after_step is not None and not after_step(frame.time_s):
            return False
    for job in batch.jobs:
        require_object_in_bin(config, job.object_id, targets[job.placement.target_id])
    return True


def prepare_panda_batch(simulation: ConveyorSimulation, *, serial: bool = False, perception: bool = False) -> PreparedPandaBatch:
    """Compile on reproducible preview scenes and validate both arms together."""
    if simulation.simulation_time_s != 0.0 or simulation.exit_mode != "remove":
        raise ValueError("Prepared playback requires a fresh remove-at-exit scenario")
    if serial and perception:
        raise ValueError("Prepared camera pickup currently uses concurrent motion")
    return _prepare_panda_batch(simulation, serial=serial, perception=perception)


def _prepare_panda_batch(
    simulation: ConveyorSimulation, *, serial: bool = False, perception: bool = False,
    tracks: list[SchedulableObject] | None = None, bindings: dict[str, str] | None = None,
) -> PreparedPandaBatch:
    start = _start_scene(simulation)
    preview = _preview_scene(simulation)

    def restore_scene():
        _restore_preview(preview, simulation)

    def build_batch(camera_plans=(), *, retries=0):
        streams = {robot_id: {} for robot_id in preview.get_robot_states()}
        _record(preview, streams, streams, simulation.simulation_time_s)
        if serial:
            jobs = _record_serial(preview, streams)
        else:
            if perception:
                if len(camera_plans) != 2:
                    raise PandaPlanningError("The concurrent camera demo needs exactly two pickup plans")
                if tracks is None:
                    # Include the timestamps used to confirm the initial tracks.
                    for step in range(1, 51):
                        _record(preview, streams, streams, step * 0.01)
                plans = [(record.track_id, record.motion) for record in camera_plans]
            else:
                plans = _ground_truth_plans(preview)
            jobs = _record_concurrent(preview, streams, plans, restore_scene)
        frames, overlap = _merge_streams(streams, jobs)
        batch = PreparedPandaBatch(tuple(streams), frames, jobs, start, overlap, camera_plans, retries)
        restore_scene()
        _play(preview, batch, check_clearance=True)
        return batch

    try:
        if tracks is not None:
            restore_scene()
        if _start_scene(preview) != start:
            raise ValueError("The live scene differs from the reproducible preview")
        if tracks is None:
            return build_batch(plan_camera_pickups(preview) if perception else ())

        planner = CameraPickupPlanner(preview, tracks, expected_bindings=bindings)
        attempts = islice(planner.candidate_batches(), 1 + _MAX_REPLACEMENT_RETRIES)
        first_failure = None
        for attempt, assignments in enumerate(attempts):
            restore_scene()
            try:
                return build_batch(planner.plan(assignments), retries=attempt)
            except PandaPlanningError as exc:
                if first_failure is None:
                    first_failure = str(exc)
                last_failure = str(exc)
        raise PandaPlanningError(
            f"No validated replacement after {attempt + 1} attempts. "
            f"First failure: {first_failure}. Last failure: {last_failure}"
        )
    finally:
        preview.close()


def play_panda_batch(simulation: ConveyorSimulation, batch: PreparedPandaBatch, *, after_step: Callable[[float], bool] | None = None) -> bool:
    """Play validated commands and ownership changes without solving any motion."""
    if simulation.simulation_time_s != 0.0 or _start_scene(simulation) != batch.start_scene:
        raise ValueError("The scene no longer matches the validated playback start")
    return _play(simulation, batch, after_step=after_step)


def play_camera_panda_batch(
    simulation: ConveyorSimulation, batch: PreparedPandaBatch, *,
    after_step: Callable[[float], bool] | None = None,
    on_replanning: Callable[[float, bool], None] | None = None,
) -> CameraBatchPlayback | None:
    """Watch the approach and replan from the current state before either gripper descends."""
    if not batch.camera_plans or simulation.simulation_time_s != 0.0 or _start_scene(simulation) != batch.start_scene:
        raise ValueError("Camera playback requires a validated camera batch and its initial scene")
    camera = ConveyorCameraTracker(simulation.belt_speed_m_s)
    camera_frames = 0
    next_capture_s = 0.0
    replanned = []
    candidate_retries = 0
    belt_offset_m = 0.0
    previous_time_s = None
    previous_speed_m_s = simulation.belt_speed_m_s

    while True:
        plans = {plan.track_id: plan.motion for plan in batch.camera_plans}
        planned_at_s = max(plan.planned_at_s for plan in plans.values())
        cutoff_s = min(plan.descent_start_s for plan in plans.values())
        drift_counts = dict.fromkeys(plans, 0)
        missed_counts = dict.fromkeys(plans, 0)
        replacement_tracks = None
        approach_confirmed = False

        def observe(now):
            nonlocal camera_frames, next_capture_s, replacement_tracks, approach_confirmed
            nonlocal belt_offset_m, previous_time_s, previous_speed_m_s
            if now == previous_time_s:
                return True  # A replacement starts at the frame already applied.
            if previous_time_s is not None:
                belt_offset_m += (now - previous_time_s) * (previous_speed_m_s - camera.initial_belt_speed_m_s)
            previous_time_s = now
            if after_step is not None and not after_step(now):
                return False
            previous_speed_m_s = simulation.belt_speed_m_s
            if approach_confirmed:
                return True

            # The first pregrasp pose is on the playback grid. Allow for the
            # timestamp rounding used when combining the two arm streams.
            at_descent = now + 1e-8 >= cutoff_s
            capture_due = now + 1e-9 >= next_capture_s
            if not capture_due and not (at_descent and any(missed_counts.values())):
                approach_confirmed = at_descent
                return True
            if capture_due:
                next_capture_s += 0.05 if next_capture_s < 0.5 - 1e-9 else _APPROACH_CAMERA_PERIOD_S
            camera_frames += 1
            observed = {record.object_id: record for record in camera.observe(simulation, belt_offset_m=belt_offset_m)
                        if record.object_id in plans}
            if now + 1e-9 < planned_at_s:
                return True
            for track_id, plan in plans.items():
                track = observed.get(track_id)
                if track is None:
                    missed_counts[track_id] += 1
                    continue
                missed_counts[track_id] = 0
                deviates = dist(
                    _PREDICTOR.predict(track.state, plan.intercept_time_s).position_m,
                    plan.object_position_at(plan.intercept_time_s),
                ) > _REPLAN_POSITION_TOLERANCE_M
                drift_counts[track_id] = drift_counts[track_id] + 1 if deviates else 0
            missing = plans.keys() - observed.keys()
            if missing:
                if at_descent or any(missed_counts[track_id] > _MAX_MISSED_APPROACH_FRAMES for track_id in missing):
                    detail = ", ".join(sorted(missing))
                    raise PandaPlanningError(f"Camera tracks lost before descent at {now:.3f} s: {detail}")
                return True
            if max(drift_counts.values()) < 2:
                approach_confirmed = at_descent
                return True
            replacement_tracks = list(observed.values())
            return False

        if _play(simulation, batch, check_clearance=True, after_step=observe):
            return CameraBatchPlayback(batch.jobs, camera_frames, tuple(replanned), candidate_retries)
        if replacement_tracks is None:
            return None
        now = simulation.simulation_time_s
        if on_replanning is not None:
            on_replanning(now, True)
        batch = _prepare_panda_batch(
            simulation, perception=True, tracks=replacement_tracks,
            bindings={track_id: plan.object_id for track_id, plan in plans.items()},
        )
        replanned.append(now)
        candidate_retries += batch.candidate_retries
        if on_replanning is not None:
            on_replanning(now, False)
