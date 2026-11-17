"""RAI integration checks for seeded batches, independent exits, and resets."""

import os

import pytest

from conveyor.simulation import ConveyorSimulation

pytest.importorskip("robotic")


@pytest.mark.parametrize("scenario", ["sparse", "clutter"])
def test_batch_has_no_penetrations_and_objects_exit_independently(scenario):
    simulation = ConveyorSimulation(.2, scenario=scenario, seed=7, object_count=8)
    initial = simulation.get_ground_truth()
    assert len(initial) == len(set(simulation.object_ids)) == 8
    assert {state.geometry_type for state in initial.values()} == {"box", "cylinder"}
    assert all(state.exit_time_s is None for state in initial.values())
    partial_exit_observed = False
    previous_exited = 0
    for _ in range(25):
        simulation.config.computeCollisions()
        assert simulation.config.getCollisions(-1e-5) == []
        active = simulation.get_ground_truth(active_only=True)
        all_states = simulation.get_ground_truth()
        exited = sum(state.motion_state == "exited" for state in all_states.values())
        assert exited >= previous_exited
        previous_exited = exited
        partial_exit_observed |= 0 < exited < 8
        assert set(active) == set(simulation.active_object_ids)
        for key, state in all_states.items():
            if state.motion_state == "exited":
                assert key not in simulation.config.getFrameNames()
                assert state.simulation_time_s == state.exit_time_s <= simulation.simulation_time_s
            else:
                assert state.simulation_time_s == simulation.simulation_time_s
                assert state.position_m[0] - initial[key].position_m[0] == pytest.approx(.2 * simulation.simulation_time_s)
                assert state.position_m[1:] == initial[key].position_m[1:]
        simulation.step(.5)
    assert partial_exit_observed
    assert simulation.all_objects_finished
    assert simulation.active_object_ids == ()


def test_reset_restores_all_removed_objects_and_seed_controls_initial_layout():
    simulation = ConveyorSimulation(.2, scenario="clutter", seed=7, object_count=8)
    config = simulation.config
    initial = simulation.get_ground_truth()
    simulation.step(30)
    assert simulation.all_objects_finished
    assert simulation.get_ground_truth(active_only=True) == {}
    simulation.reset()
    assert simulation.config is config
    assert simulation.get_ground_truth() == initial
    simulation.reset(seed=9)
    assert simulation.seed == 9
    assert simulation.get_ground_truth() != initial
    assert set(simulation.get_ground_truth()) == set(initial)
    changed = simulation.get_ground_truth()
    with pytest.raises(ValueError):
        simulation.reset(seed=True)
    assert simulation.get_ground_truth() == changed


def test_release_preserves_one_object_while_other_objects_exit():
    simulation = ConveyorSimulation(.2, scenario="clutter", object_count=4)
    key = simulation.object_ids[0]
    simulation.release_object(key)
    simulation.config.getFrame(key).setPosition([0, .4, 1.0])
    simulation.step(30)
    assert simulation.active_object_ids == (key,)
    assert not simulation.all_objects_finished
    assert simulation.get_ground_truth()[key].position_m == (0, .4, 1.0)
    with pytest.raises(ValueError):
        simulation.release_object(simulation.object_ids[1])


def test_exit_timestamp_is_independent_of_step_partition():
    large_step = ConveyorSimulation(.2)
    small_steps = ConveyorSimulation(.2)
    large_step.step(20)
    for _ in range(100):
        small_steps.step(.2)
    first = large_step.get_ground_truth()["object_001"]
    second = small_steps.get_ground_truth()["object_001"]
    assert first.exit_time_s == pytest.approx(11.1)
    assert second.exit_time_s == pytest.approx(first.exit_time_s)
    assert second.position_m == pytest.approx(first.position_m)


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires OpenGL/WSLg")
def test_initial_clutter_objects_are_visible_to_the_overhead_camera():
    np = pytest.importorskip("numpy")
    simulation = ConveyorSimulation(scenario="clutter", object_count=8, seed=7)
    try:
        observation = simulation.get_observation()
        calibration = observation.calibration
        world_to_camera = np.linalg.inv(np.asarray(calibration.camera_to_world))
        intrinsics = np.asarray(calibration.intrinsics)
        for state in simulation.get_ground_truth(active_only=True).values():
            point_world = np.array([*state.position_m, 1.0])
            point_world[2] += state.dimensions_m[2] / 2
            point_camera = world_to_camera @ point_world
            pixel = intrinsics @ point_camera[:3]
            u, v = (int(round(float(x))) for x in pixel[:2] / pixel[2])
            assert 0 <= u < calibration.width and 0 <= v < calibration.height
            assert observation.depth_m[v, u] == pytest.approx(point_camera[2], abs=1e-3)
    finally:
        simulation.close()
