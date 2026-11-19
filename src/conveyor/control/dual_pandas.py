"""Prepare and validate a shared Panda timeline before starting live playback."""

from bisect import bisect_right
from dataclasses import dataclass
from math import dist, isfinite
from typing import Callable

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
from .rgbd_panda import CameraPickupPlan, ConveyorCameraTracker, plan_camera_pickups, plan_tracked_pickups
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


@dataclass(frozen=True)
class CameraBatchPlayback:
    jobs: tuple[PandaJobResult, ...]
    camera_frames: int
    replanned_at_s: tuple[float, ...]


_PREDICTOR = ConstantVelocityPredictor(planar_motion=True, max_horizon_s=60.0)
_REPLAN_POSITION_TOLERANCE_M = 0.005


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
    return simulation.belt_speed_m_s, simulation.object_specs, pairs, tuple(shapes)


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


def _merge_streams(streams, jobs):
    times = {time for stream in streams.values() for time in stream}
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


def _validate_clearance(scene, pairs, targets, jobs, now):
    import robotic as ry

    limits = _clearance_limits(targets, jobs, now)
    for first, second in pairs:
        separation = -float(scene.config.eval(ry.FS.negDistance, [first, second])[0][0])
        if not isfinite(separation) or separation < limits.get(frozenset((first, second)), 0.005):
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
    pairs = ()
    previous_topology = ()
    previous = scene.simulation_time_s

    def current_pairs():
        return tuple(dict.fromkeys(
            pair for robot in robot_states for pair in moving_collision_pairs(config, robot.arm_joint_names[0])
        ))

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
            if not pairs or topology != previous_topology:
                pairs = current_pairs()
                previous_topology = topology
            _validate_clearance(scene, pairs, targets, batch.jobs, frame.time_s)
        for job, picking in due:
            _apply_event(scene, robots[job.robot_id], job, picking, targets)
        if check_clearance and due:
            pairs = current_pairs()
            _validate_clearance(scene, pairs, targets, batch.jobs, frame.time_s)
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

    try:
        if tracks is not None:
            restore_scene()
        if _start_scene(preview) != start:
            raise ValueError("The live scene differs from the reproducible preview")
        streams = {robot_id: {} for robot_id in preview.get_robot_states()}
        _record(preview, streams, streams, simulation.simulation_time_s)
        camera_plans = ()
        if serial:
            jobs = _record_serial(preview, streams)
        else:
            if tracks is not None:
                camera_plans = plan_tracked_pickups(preview, tracks, expected_bindings=bindings)
            elif perception:
                camera_plans = plan_camera_pickups(preview)
                # Include the timestamps used to confirm tracks in the preview.
                for step in range(1, 51):
                    _record(preview, streams, streams, step * 0.01)
            if perception:
                if len(camera_plans) != 2:
                    raise PandaPlanningError("The concurrent camera demo needs exactly two pickup plans")
                plans = [(record.track_id, record.motion) for record in camera_plans]
            else:
                plans = _ground_truth_plans(preview)
            jobs = _record_concurrent(preview, streams, plans, restore_scene)
        frames, overlap = _merge_streams(streams, jobs)
        batch = PreparedPandaBatch(tuple(streams), frames, jobs, start, overlap, camera_plans)
        restore_scene()
        _play(preview, batch, check_clearance=True)
        return batch
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
    belt_offset_m = 0.0
    previous_time_s = None
    previous_speed_m_s = simulation.belt_speed_m_s

    while True:
        plans = {plan.track_id: plan.motion for plan in batch.camera_plans}
        planned_at_s = max(plan.planned_at_s for plan in plans.values())
        cutoff_s = min(plan.descent_start_s for plan in plans.values())
        drift_counts = dict.fromkeys(plans, 0)
        replacement_tracks = None

        def observe(now):
            nonlocal camera_frames, next_capture_s, replacement_tracks
            nonlocal belt_offset_m, previous_time_s, previous_speed_m_s
            if now == previous_time_s:
                return True  # A replacement starts at the frame already applied.
            if previous_time_s is not None:
                belt_offset_m += (now - previous_time_s) * (previous_speed_m_s - camera.initial_belt_speed_m_s)
            previous_time_s = now
            if after_step is not None and not after_step(now):
                return False
            previous_speed_m_s = simulation.belt_speed_m_s
            if now >= cutoff_s or now + 1e-9 < next_capture_s:
                return True
            next_capture_s += 0.05 if next_capture_s < 0.5 - 1e-9 else 0.20
            camera_frames += 1
            observed = {record.object_id: record for record in camera.observe(simulation, belt_offset_m=belt_offset_m)
                        if record.object_id in plans}
            if now <= planned_at_s + 1e-9:
                return True
            for track_id, plan in plans.items():
                track = observed.get(track_id)
                deviates = track is not None and dist(
                    _PREDICTOR.predict(track.state, plan.intercept_time_s).position_m,
                    plan.object_position_at(plan.intercept_time_s),
                ) > _REPLAN_POSITION_TOLERANCE_M
                drift_counts[track_id] = drift_counts[track_id] + 1 if deviates else 0
            if max(drift_counts.values()) < 2:
                return True
            if observed.keys() != plans.keys():
                raise PandaPlanningError("A camera track is missing at the replanning checkpoint")
            replacement_tracks = list(observed.values())
            return False

        if _play(simulation, batch, check_clearance=True, after_step=observe):
            return CameraBatchPlayback(batch.jobs, camera_frames, tuple(replanned))
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
        if on_replanning is not None:
            on_replanning(now, False)
