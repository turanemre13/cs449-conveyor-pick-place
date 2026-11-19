"""Verify the actual camera-to-scheduler-to-placement handoff."""

import os

import pytest

from conveyor.control.dual_pandas import play_camera_panda_batch, prepare_panda_batch
from conveyor.control.panda_pickup import execute_panda_pickup
from conveyor.control.panda_placement import execute_panda_placement
from conveyor.control.rgbd_panda import plan_camera_pickup
from conveyor.simulation import ConveyorSimulation


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
@pytest.mark.parametrize("change_speed", (False, True), ids=("constant_speed", "speed_change"))
def test_two_live_rgbd_tracks_drive_concurrent_placements(change_speed):
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
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
            if change_speed and now >= 0.75:
                simulation.belt_speed_m_s = 0.025
            return True

        def record_replanning(now, active):
            replanning_states.append((now, active, simulation.get_robot_states()))

        playback = play_camera_panda_batch(
            simulation, batch, after_step=update_speed, on_replanning=record_replanning,
        )
        assert playback is not None
        assert playback.camera_frames > 11
        assert all(first < second for first, second in zip(observed_times, observed_times[1:]))
        if change_speed:
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
