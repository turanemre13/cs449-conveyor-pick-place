"""Classify a moving product feed and prepare a checked sorting run."""

from dataclasses import dataclass, replace

from conveyor.grasping import select_top_down_grasps
from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.planning.panda import PandaPlanningError, TcpPose
from conveyor.planning.static_panda import plan_static_panda
from conveyor.prediction import tracked_states_from_ground_truth
from conveyor.scheduling import DeadlineAwareGreedyScheduler, robot_configs_from_snapshots
from conveyor.simulation import ConveyorSimulation

from .dual_pandas import (
    PlaybackFrame, PreparedPandaBatch, _interpolate, _merge_streams, _play, _preview_scene, _record, _start_scene,
)
from .panda_pickup import execute_panda_pickup
from .panda_placement import _play_arm, execute_panda_placement
from .rgbd_panda import ConveyorCameraTracker, _bind_scene_objects
from .serial_pandas import PandaJobResult


_PREFERRED_ARM = {"box": "robot_a", "cylinder": "robot_b"}
_BIN_SLOTS_M = ((0.0, 0.0), (.105, 0.0), (-.105, 0.0), (0.0, .10), (0.0, -.10))
_CAMERA_PERIOD_S = 0.20
_PICK_LOOKAHEAD_S = 4.0
_PREFERRED_START_MARGIN_S = 0.50


@dataclass(frozen=True)
class FeedRun:
    batch: PreparedPandaBatch
    missed: tuple[tuple[str, str], ...]


def _pickup_alternatives(assignment, track, model):
    yield assignment
    close_s = model.config.timing.gripper_close_s
    deadline = min(assignment.deadline_s, assignment.object_exit_time_s or float("inf"))
    geometry = track.geometry
    for delay in (.20, .40, .60):
        now = assignment.intercept_time_s + delay
        if now + close_s >= deadline:
            break
        centroid = model.predict(track.object_id, now)
        offset = tuple(q - p for q, p in zip(centroid, assignment.intercept_position_m))
        grasp = replace(assignment.grasp,
                        position_m=tuple(p + d for p, d in zip(assignment.grasp.position_m, offset)),
                        pregrasp_position_m=tuple(p + d for p, d in zip(assignment.grasp.pregrasp_position_m, offset)))
        yield replace(assignment, intercept_time_s=now, intercept_position_m=centroid, grasp=grasp)

    alternatives = select_top_down_grasps(
        geometry, assignment.intercept_position_m, config=model.config.gripper,
        object_velocity_m_s=track.state.velocity_m_s,
    ).feasible
    for grasp in (candidate for candidate in alternatives[:3] if candidate.candidate_id != assignment.grasp.candidate_id):
        yield replace(assignment, grasp=grasp)


def _candidate_picks(scene, pending, slots, active, reasons):
    truth = scene.get_ground_truth(active_only=True)
    on_belt = {key: truth[key] for key in scene.on_belt_object_ids}
    collision_tracks = tracked_states_from_ground_truth(on_belt)
    robots = scene.get_robot_states()
    available = {job.robot_id: job.finished_at_s for job in active}
    reservations = {job.object_id for job in active}
    candidates = robot_configs_from_snapshots(robots, scene.get_targets(), available_at_s=available)
    candidates = [replace(robot, tool_position_m=robot.place_position_m)
                  if robot.robot_id in available else robot for robot in candidates]
    scheduler = DeadlineAwareGreedyScheduler()
    now = scene.simulation_time_s
    jobs = []
    close_s = scheduler.config.timing.gripper_close_s
    for track in pending:
        # Wait until the full product is inside the camera's belt ROI.
        if track.state.position_m[0] < -1.09:
            continue
        try:
            scene_id = _bind_scene_objects((track,), on_belt)[track.object_id]
        except PandaPlanningError:
            continue  # Moving arms also pass through the overhead image.
        if scene_id in reservations:
            continue

        # Products move with the commanded belt speed. RGB-D supplies the
        # observed centroid and geometry, including the class used for routing.
        track = replace(track, state=replace(track.state, velocity_m_s=(scene.belt_speed_m_s, 0.0, 0.0)))

        preferred = _PREFERRED_ARM[track.geometry.geometry_type]
        options = []
        for robot in sorted(candidates, key=lambda candidate: candidate.robot_id != preferred):
            if len(slots[robot.robot_id]) >= len(_BIN_SLOTS_M):
                reasons[scene_id] = f"bin full for {robot.robot_id}"
                continue
            schedule = scheduler.schedule((track,), (robot,), now)
            if not schedule.assigned:
                reasons[scene_id] = schedule.rejected[0].reason
                continue
            assignment = schedule.assigned[0]
            if assignment.intercept_time_s + close_s >= min(
                assignment.deadline_s, assignment.object_exit_time_s or float("inf"),
            ):
                reasons[scene_id] = f"closure does not fit the window for {robot.robot_id}"
                continue
            if robot.robot_id in available or assignment.intercept_time_s - now > _PICK_LOOKAHEAD_S:
                break  # Give the preferred arm time to reach the arriving product.
            options.append((assignment, scheduler.model))
            if robot.robot_id == preferred and assignment.latest_start_s - now > _PREFERRED_START_MARGIN_S:
                break
        if options:
            jobs.append((options[0][0].deadline_s, scene_id, track, options))

    for _, scene_id, track, options in sorted(jobs, key=lambda item: item[:2]):
        for assignment, model in options:
            for candidate in _pickup_alternatives(assignment, track, model):
                grasp = replace(candidate.grasp, object_id=scene_id)
                target = replace(candidate, object_id=scene_id, grasp=grasp)
                try:
                    plan = plan_moving_panda(
                        scene.config, robots[target.robot_id], target, replace(track.state, object_id=scene_id),
                        close_duration_s=close_s, collision_track=collision_tracks[scene_id],
                        neighbors=tuple(state for key, state in collision_tracks.items() if key != scene_id),
                    )
                except PandaPlanningError as exc:
                    reasons[scene_id] = str(exc)
                    continue
                yield track.object_id, plan


def _place_in_free_slot(scene, plan, occupied, after_step):
    now = scene.simulation_time_s
    failures = []
    for align_bin_axes in (True, False):
        for slot in _BIN_SLOTS_M:
            if slot in occupied:
                continue
            try:
                placement = execute_panda_placement(scene, plan, after_step=after_step, bin_offset_m=slot,
                                                   align_bin_axes=align_bin_axes)
            except PandaPlanningError as exc:
                if scene.simulation_time_s != now:
                    raise
                failures.append(str(exc))
                continue
            return placement, slot
    detail = failures[-1] if failures else "all positions were used"
    raise PandaPlanningError(f"No usable bin position for {plan.object_id}: {detail}")


def _make_batch(streams, jobs, start, times=()):
    frames, overlap = _merge_streams(streams, jobs, times)
    return PreparedPandaBatch(tuple(streams), frames, tuple(jobs), start, overlap)


def _plan_job(scene, batch, streams, track_id, plan, occupied, ready):
    """Preview one job and validate it alongside every reserved arm motion."""
    robot_id = plan.robot_id
    preview = scene.copy_for_planning()
    trial_streams = {**streams, robot_id: streams[robot_id].copy()}
    try:
        def record(now):
            return _record(preview, trial_streams, (robot_id,), now)

        record(preview.simulation_time_s)
        pickup = execute_panda_pickup(preview, plan, after_step=record)
        placement, slot = _place_in_free_slot(preview, plan, occupied, record)
        robot = preview.get_robot_states()[robot_id]
        try:
            home = plan_static_panda(
                preview.config, robot, ready[robot_id], duration_s=2.4, max_joint_speed_rad_s=1.5,
            )
        except PandaPlanningError:
            pass  # The validated bin retreat is also a usable waiting pose.
        else:
            _play_arm(preview, list(robot.arm_joint_names), home.samples, record)
        job = PandaJobResult(
            robot_id, plan.object_id, plan.planned_at_s, plan.intercept_time_s, preview.simulation_time_s,
            placement, pickup.pickup_time_s, plan.closure, track_id,
        )
    finally:
        preview.close()

    trial = _make_batch(trial_streams, (*batch.jobs, job), batch.start_scene)
    start_s = round(scene.simulation_time_s, 9)
    suffix = replace(trial, frames=tuple(frame for frame in trial.frames if frame.time_s > start_s))
    verify = scene.copy_for_planning()
    try:
        _play(verify, suffix, check_clearance=True)
    finally:
        verify.close()
    return trial, trial_streams, slot


def _advance(scene, batch, stop_s):
    """Advance the reserved motions to the next camera timestamp."""
    start_s = round(scene.simulation_time_s, 9)
    frames = tuple(frame for frame in batch.frames if start_s < frame.time_s <= stop_s)
    if not frames or frames[-1].time_s < stop_s:
        times = tuple(frame.time_s for frame in batch.frames)
        commands = tuple(_interpolate(times, stream, stop_s) for stream in zip(*(frame.commands for frame in batch.frames)))
        frames = (*frames, PlaybackFrame(stop_s, commands))
    _play(scene, replace(batch, frames=frames), after_step=lambda now: now < stop_s)


def prepare_feed_run(simulation: ConveyorSimulation) -> FeedRun:
    """Use successive RGB-D observations to compile one finite feed without resetting it."""
    if simulation.scenario != "feed" or simulation.exit_mode != "remove" or simulation.simulation_time_s != 0.0:
        raise ValueError("Feed sorting requires a fresh feed scenario with remove-at-exit mode")
    if simulation.belt_speed_m_s <= 0.0:
        raise ValueError("Feed sorting requires a positive belt speed")

    start = _start_scene(simulation)
    preview = _preview_scene(simulation)
    try:
        if _start_scene(preview) != start:
            raise ValueError("The live feed differs from its reproducible preview")
        streams = {robot_id: {} for robot_id in preview.get_robot_states()}
        ready = {key: TcpPose(robot.gripper_position_m, robot.gripper_quaternion_wxyz)
                 for key, robot in preview.get_robot_states().items()}
        camera = ConveyorCameraTracker(preview.belt_speed_m_s)
        slots = {robot_id: set() for robot_id in streams}
        reasons = {}
        times = set()

        _record(preview, streams, streams, 0.0)
        batch = _make_batch(streams, (), start)
        while True:
            now = preview.simulation_time_s
            active = [job for job in batch.jobs if job.finished_at_s > now + 1e-8]
            if not (active or preview.on_belt_object_ids or preview.queued_object_ids):
                break
            current = camera.observe(preview)
            while len(active) < len(streams) and preview.on_belt_object_ids:
                for track_id, plan in _candidate_picks(preview, current, slots, active, reasons):
                    try:
                        result = _plan_job(preview, batch, streams, track_id, plan, slots[plan.robot_id], ready)
                    except (PandaPlanningError, ValueError) as exc:
                        reasons[plan.object_id] = str(exc)
                        continue
                    batch, streams, slot = result
                    slots[plan.robot_id].add(slot)
                    active.append(batch.jobs[-1])
                    break
                else:
                    break

            stop_s = round(now + _CAMERA_PERIOD_S, 9)
            _advance(preview, batch, stop_s)
            times.update(round(now + step * _CAMERA_PERIOD_S / 8, 9) for step in range(1, 9))

        completed = {job.object_id for job in batch.jobs}
        missed = tuple((key, reasons.get(key, "no confirmed camera track"))
                       for key in preview.object_ids if key not in completed)
        batch = _make_batch(streams, batch.jobs, start, times)
        preview.reset()
        _play(preview, batch, check_clearance=True)
        return FeedRun(batch, missed)
    finally:
        preview.close()
