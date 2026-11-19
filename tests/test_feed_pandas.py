"""Check the physical infeed and the vision-driven sorting run."""

import os
from dataclasses import replace

import pytest

from conveyor.control import feed_pandas
from conveyor.control.dual_pandas import play_panda_batch
from conveyor.control.feed_pandas import prepare_feed_run
from conveyor.simulation import ConveyorSimulation


@pytest.mark.parametrize("exit_mode", ("remove", "fall"))
def test_compact_infeed_releases_queued_products_under_cover(exit_mode):
    ry = pytest.importorskip("robotic")
    simulation = ConveyorSimulation(0.12, scenario="feed", seed=7, object_count=4, exit_mode=exit_mode)
    try:
        original = simulation.get_ground_truth()
        upstream = simulation.config.getFrame("conveyor_infeed")
        assert upstream.getSize()[0] == pytest.approx(0.56)
        assert upstream.getPosition()[2] + upstream.getSize()[2] / 2 == pytest.approx(0.75)
        del upstream
        assert all(product.position_m[0] < -1.2 for product in original.values())
        first, second, *rest = simulation.object_ids
        assert simulation.on_belt_object_ids == (first,)
        assert simulation.queued_object_ids == (second, *rest)
        assert set(simulation.get_ground_truth(active_only=True)) == {first}
        assert all(simulation.config.getFrame(key).getShapeType() == ry.ST.none for key in (second, *rest))
        simulation.step(2.0)
        current = simulation.get_ground_truth()
        assert current.keys() == original.keys()
        assert current[first].position_m[0] > -1.2
        for key, product in current.items():
            assert product.position_m[0] == pytest.approx(original[key].position_m[0] + 0.24)

        simulation.step(5.0)
        assert simulation.on_belt_object_ids == (first, second)
        assert simulation.queued_object_ids == tuple(rest)
        cover = simulation.config.getFrame("conveyor_infeed_cover")
        half_cover = cover.getSize()[0] / 2
        product = simulation.get_ground_truth()[second]
        half_product = product.dimensions_m[0] / 2
        assert abs(product.position_m[0] - cover.getPosition()[0]) + half_product < half_cover
        assert simulation.config.getFrame(second).getShapeType() == ry.ST.cylinder
        assert product.position_m[0] == pytest.approx(original[second].position_m[0] + 0.84)
    finally:
        simulation.close()


def test_planning_copy_preserves_ownership_without_mutating_the_source():
    pytest.importorskip("robotic")
    simulation = ConveyorSimulation(0.12, scenario="feed", seed=7, object_count=2)
    try:
        simulation.step(2.0)
        simulation.release_object("object_001")
        simulation.config.attach("a_gripper", "object_001")
        before = simulation.get_ground_truth()
        preview = simulation.copy_for_planning()
        try:
            assert preview.get_ground_truth() == before
            assert preview.config.getFrame("object_001").getParent().name == "a_gripper"
            preview.step(1.0)
            assert preview.simulation_time_s == pytest.approx(3.0)
            assert preview.get_ground_truth()["object_002"].position_m[0] == pytest.approx(
                before["object_002"].position_m[0] + 0.12,
            )
            assert simulation.get_ground_truth() == before
        finally:
            preview.close()
    finally:
        simulation.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires an OpenGL/X11 display")
def test_feed_sorts_four_successive_rgbd_tracks_into_distinct_bin_positions():
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
    simulation = ConveyorSimulation(0.12, scenario="feed", seed=7, object_count=4)
    try:
        before = simulation.get_ground_truth()
        result = prepare_feed_run(simulation)
        assert simulation.get_ground_truth() == before
        assert result.missed == ()
        assert len({job.track_id for job in result.batch.jobs}) == 4
        assert all(job.track_id.startswith("track_") for job in result.batch.jobs)
        assert {job.object_id: job.robot_id for job in result.batch.jobs} == {
            "object_001": "robot_a", "object_002": "robot_b", "object_003": "robot_a", "object_004": "robot_b",
        }
        assert result.batch.overlap_s > 0
        assert play_panda_batch(simulation, result.batch)
        truth = simulation.get_ground_truth()
        assert not simulation.on_belt_object_ids
        assert len({tuple(round(value, 3) for value in product.position_m[:2]) for product in truth.values()}) == 4
        assert all(product.motion_state == "released" for product in truth.values())
    finally:
        simulation.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires an OpenGL/X11 display")
def test_other_arm_takes_camera_confirmed_box_when_a_is_unavailable(monkeypatch):
    pytest.importorskip("robotic")
    pytest.importorskip("cv2")
    configs = feed_pandas.robot_configs_from_snapshots

    def mark_a_busy(*args, **kwargs):
        return [replace(robot, available_at_s=100.0) if robot.robot_id == "robot_a" else robot
                for robot in configs(*args, **kwargs)]

    monkeypatch.setattr(feed_pandas, "robot_configs_from_snapshots", mark_a_busy)
    simulation = ConveyorSimulation(0.12, scenario="feed", seed=7, object_count=1)
    try:
        result = prepare_feed_run(simulation)
        assert result.missed == ()
        job, = result.batch.jobs
        assert job.track_id == "track_001"
        assert job.robot_id == "robot_b"
        assert job.placement.target_id == "target_bin_b"
        assert play_panda_batch(simulation, result.batch)
        assert simulation.get_ground_truth()[job.object_id].motion_state == "released"
    finally:
        simulation.close()
