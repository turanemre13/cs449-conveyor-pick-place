"""Verify the actual camera-to-scheduler-to-placement handoff."""

import os

import pytest

from conveyor.control import dual_pandas
from conveyor.control.dual_pandas import play_camera_panda_batch, prepare_panda_batch
from conveyor.control.panda_pickup import execute_panda_pickup
from conveyor.control.panda_placement import execute_panda_placement
from conveyor.control.rgbd_panda import plan_camera_pickup
from conveyor.perception import RGBDObjectDetector
from conveyor.planning.panda import PandaPlanningError
from conveyor.simulation import ConveyorSimulation


def _occlude_cylinder(monkeypatch, start_s, end_s=float("inf")):
    detect = RGBDObjectDetector.detect
    dropped_frames = []

    def detect_with_occlusion(detector, observation):
        detections = detect(detector, observation)
        if start_s <= observation.simulation_time_s < end_s:
            dropped_frames.append(observation.simulation_time_s)
            return [item for item in detections if item.geometry_type != "cylinder"]
        return detections

    monkeypatch.setattr(RGBDObjectDetector, "detect", detect_with_occlusion)
    return dropped_frames


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires an OpenGL/X11 display")
def test_real_rgbd_track_drives_grasp_and_placement():
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
    simulation = ConveyorSimulation(0.12, scenario="single")
    try:
        camera_plan = plan_camera_pickup(simulation)
        motion = camera_plan.motion
        assert camera_plan.track_id == "track_001"
        assert motion.object_id == "object_001"
        assert motion.planned_at_s == pytest.approx(0.5)
        assert motion.intercept_time_s > motion.planned_at_s
        assert motion.collision_track is not None
        assert motion.track.position_m != motion.collision_track.position_m

        pickup = execute_panda_pickup(simulation, motion)
        placed = execute_panda_placement(simulation, motion)
        assert pickup.interception.success
        assert placed.target_id == "target_bin_a"
        assert placed.floor_gap_m < 0.001
        assert simulation.config.getFrame(motion.object_id).getParent().name == "world"
    finally:
        simulation.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires an OpenGL/X11 display")
def test_camera_playback_holds_when_replacement_fails_validation(monkeypatch):
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
    simulation = ConveyorSimulation(0.022, scenario="sparse", seed=21, object_count=2)
    try:
        batch = prepare_panda_batch(simulation, perception=True)
        checkpoints = []
        validate = dual_pandas._validate_clearance

        def reject_replacement(distances, targets, jobs, now):
            if distances.config is not simulation.config:
                raise PandaPlanningError("Replacement violates shared clearance")
            validate(distances, targets, jobs, now)

        def change_speed(now):
            if now >= 0.75:
                simulation.belt_speed_m_s = 0.025
            return True

        def checkpoint(now, active):
            assert active
            checkpoints.append((now, simulation.get_robot_states(), simulation.get_ground_truth()))

        monkeypatch.setattr(dual_pandas, "_validate_clearance", reject_replacement)
        with pytest.raises(PandaPlanningError, match="Replacement violates shared clearance"):
            play_camera_panda_batch(simulation, batch, after_step=change_speed, on_replanning=checkpoint)

        assert checkpoints == [(simulation.simulation_time_s, simulation.get_robot_states(), simulation.get_ground_truth())]
        assert simulation.on_belt_object_ids == simulation.object_ids
    finally:
        simulation.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires an OpenGL/X11 display")
@pytest.mark.parametrize(
    "new_speed_m_s, occlusion, retries_expected",
    ((None, False, False), (0.025, False, False), (0.0245, True, False), (0.025, True, True)),
    ids=("constant_speed", "speed_change", "speed_change_with_occlusion", "alternate_grasp_after_occlusion"),
)
def test_two_live_rgbd_tracks_drive_concurrent_placements(new_speed_m_s, occlusion, retries_expected, monkeypatch):
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
    if occlusion:
        dropped_frames = _occlude_cylinder(monkeypatch, 0.90, 1.22)
    simulation = ConveyorSimulation(0.022, scenario="sparse", seed=21, object_count=2)
    try:
        before = simulation.get_ground_truth()
        batch = prepare_panda_batch(simulation, perception=True)
        assert simulation.get_ground_truth() == before
        assert batch.overlap_s > 5.0
        assert {(job.track_id, job.object_id, job.robot_id) for job in batch.jobs} == {
            ("track_001", "object_001", "robot_a"), ("track_002", "object_002", "robot_b"),
        }
        assert all(job.planning_time_s == pytest.approx(0.5) for job in batch.jobs)
        replanning_states = []
        observed_times = []

        def update_speed(now):
            observed_times.append(now)
            if new_speed_m_s is not None and now >= 0.75:
                simulation.belt_speed_m_s = new_speed_m_s
            return True

        def record_replanning(now, active):
            replanning_states.append((now, active, simulation.get_robot_states()))

        playback = play_camera_panda_batch(
            simulation, batch, after_step=update_speed, on_replanning=record_replanning,
        )
        assert playback is not None
        assert playback.camera_frames > 11
        assert bool(playback.candidate_retries) is retries_expected
        assert all(first < second for first, second in zip(observed_times, observed_times[1:]))
        if new_speed_m_s is not None:
            start, finish = replanning_states
            assert start[1] is True and finish[1] is False
            assert start[2] == finish[2]
            assert playback.replanned_at_s == (start[0],)
            assert 0.75 < start[0] < min(plan.motion.descent_start_s for plan in batch.camera_plans)
            assert all(job.planning_time_s == start[0] for job in playback.jobs)
            assert {job.intercepted_at_s for job in playback.jobs}.isdisjoint(job.intercepted_at_s for job in batch.jobs)
            assert {job.track_id: job.object_id for job in playback.jobs} == {
                job.track_id: job.object_id for job in batch.jobs
            }
            if occlusion:
                assert len(dropped_frames) == 2
                assert start[0] >= 1.22
        else:
            assert not replanning_states
            assert playback.jobs == batch.jobs

        targets = simulation.get_targets()
        for job in playback.jobs:
            target = targets[job.placement.target_id]
            obj = simulation.get_ground_truth()[job.object_id]
            assert target.preferred_robot_id == job.robot_id
            assert obj.position_m[:2] == pytest.approx(target.position_m[:2], abs=0.001)
            assert simulation.config.getFrame(job.object_id).getParent().name == "world"
    finally:
        simulation.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires an OpenGL/X11 display")
def test_camera_playback_tries_later_interception_after_rejection(monkeypatch):
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
    simulation = ConveyorSimulation(0.022, scenario="sparse", seed=21, object_count=2)
    try:
        batch = prepare_panda_batch(simulation, perception=True)
        validate = dual_pandas._validate_clearance
        rejected_at_s = None

        def reject_first_interception(distances, targets, jobs, now):
            nonlocal rejected_at_s
            if distances.config is not simulation.config and now >= 1.7:
                intercept = next(job.intercepted_at_s for job in jobs if job.robot_id == "robot_a")
                if rejected_at_s is None:
                    rejected_at_s = intercept
                if intercept == rejected_at_s:
                    raise PandaPlanningError("Shared clearance rejects the first interception time")
            validate(distances, targets, jobs, now)

        def change_speed(now):
            if now >= 0.75:
                simulation.belt_speed_m_s = 0.025
            return True

        monkeypatch.setattr(dual_pandas, "_validate_clearance", reject_first_interception)
        playback = play_camera_panda_batch(simulation, batch, after_step=change_speed)

        assert playback is not None
        assert playback.candidate_retries > 0
        assert len(playback.jobs) == 2
        retried = next(job for job in playback.jobs if job.robot_id == "robot_a")
        assert rejected_at_s is not None and retried.intercepted_at_s > rejected_at_s
        assert not simulation.on_belt_object_ids
    finally:
        simulation.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires an OpenGL/X11 display")
@pytest.mark.parametrize("lost_at_s, missed_frames", ((0.90, 3), (2.20, 2)), ids=("timeout", "at_descent"))
def test_camera_playback_holds_if_track_stays_lost(lost_at_s, missed_frames, monkeypatch):
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
    dropped = _occlude_cylinder(monkeypatch, lost_at_s)
    simulation = ConveyorSimulation(0.022, scenario="sparse", seed=21, object_count=2)
    try:
        batch = prepare_panda_batch(simulation, perception=True)
        descent_s = min(plan.motion.descent_start_s for plan in batch.camera_plans)
        with pytest.raises(PandaPlanningError, match="Camera tracks lost before descent.*track_002"):
            play_camera_panda_batch(simulation, batch)

        assert len(dropped) == missed_frames
        assert simulation.simulation_time_s == pytest.approx(dropped[-1])
        assert simulation.simulation_time_s <= descent_s + 1e-8
        assert simulation.on_belt_object_ids == simulation.object_ids
    finally:
        simulation.close()
