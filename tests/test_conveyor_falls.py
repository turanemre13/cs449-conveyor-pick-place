"""End-to-end handoff from prescribed transport to actual CPU dynamics."""

import os

import pytest

from conveyor.simulation import ConveyorSimulation

pytest.importorskip("robotic")


def test_exact_edge_handoff_inherits_velocity_then_falls_and_settles():
    simulation = ConveyorSimulation(.2, exit_mode="fall")
    try:
        handoff_s = (1.204 + 1.05) / .2
        simulation.step(handoff_s)
        state = simulation.get_ground_truth()["object_001"]
        assert state.motion_state == "falling"
        assert state.exit_time_s == pytest.approx(handoff_s)
        assert state.position_m == pytest.approx((1.204, 0, .79), abs=1e-6)
        assert state.velocity_m_s == pytest.approx((.2, 0, 0), abs=1e-6)
        simulation.step(.2)
        falling = simulation.get_ground_truth()["object_001"]
        assert falling.position_m[2] < .78
        assert falling.velocity_m_s[2] < 0
        assert falling.quaternion_wxyz != pytest.approx(state.quaternion_wxyz)
        simulation.step(3.)
        landed = simulation.get_ground_truth()["object_001"]
        assert landed.motion_state == "settled"
        assert .01 < landed.position_m[2] < .12
        assert simulation.all_objects_finished
        assert "object_001" in simulation.config.getFrameNames()
    finally:
        simulation.close()


def test_ordered_batch_stays_visible_on_floor_and_reset_restores_transport():
    simulation = ConveyorSimulation(.2, scenario="ordered", object_count=8, exit_mode="fall")
    try:
        initial = simulation.get_ground_truth()
        simulation.step(17.)
        states = simulation.get_ground_truth(active_only=True)
        assert simulation.all_objects_finished
        assert simulation.on_belt_object_ids == ()
        assert set(simulation.fallen_object_ids) == set(initial)
        assert len(states) == 8
        for state in states.values():
            assert state.motion_state == "settled"
            assert .01 < state.position_m[2] < .15
            assert state.simulation_time_s == simulation.simulation_time_s
            assert state.exit_time_s < simulation.simulation_time_s
        simulation.reset()
        assert simulation.get_ground_truth() == initial
        assert simulation.on_belt_object_ids == simulation.object_ids
        assert simulation.fallen_object_ids == ()
        assert not simulation.all_objects_finished
    finally:
        simulation.close()


def test_release_of_falling_body_relinquishes_physics_ownership():
    simulation = ConveyorSimulation(.4, exit_mode="fall")
    try:
        simulation.step(5.8)
        assert simulation.fallen_object_ids == ("object_001",)
        simulation.release_object("object_001")
        simulation.config.getFrame("object_001").setPosition([0, .4, 1.0])
        simulation.step(1.)
        state = simulation.get_ground_truth()["object_001"]
        assert state.motion_state == "released"
        assert state.position_m == (0, .4, 1.)
        assert state.velocity_m_s is None
        assert not simulation.all_objects_finished
    finally:
        simulation.close()


def test_fall_handoff_time_is_independent_of_outer_step_partition():
    single_step = ConveyorSimulation(.2, exit_mode="fall")
    many_steps = ConveyorSimulation(.2, exit_mode="fall")
    try:
        single_step.step(12.)
        for _ in range(600):
            many_steps.step(.02)
        assert many_steps.get_ground_truth()["object_001"].exit_time_s == pytest.approx(
            single_step.get_ground_truth()["object_001"].exit_time_s, abs=1e-8,
        )
    finally:
        single_step.close()
        many_steps.close()


def test_close_retains_final_ground_truth_and_reset_reopens_simulation():
    simulation = ConveyorSimulation(.3, exit_mode="fall")
    try:
        simulation.step(11.)
        final = simulation.get_ground_truth()
        assert final["object_001"].motion_state == "settled"
        simulation.close()
        assert simulation.get_ground_truth() == final
        assert simulation.get_ground_truth(active_only=True) == final
        returned = simulation.get_ground_truth()
        returned.clear()
        assert simulation.get_ground_truth() == final
        simulation.close()  # Repeated cleanup must be safe.
        for operation in (
            lambda: simulation.step(.1), simulation.get_observation,
            lambda: simulation.release_object("object_001"),
        ):
            with pytest.raises(RuntimeError, match="reset"):
                operation()
        simulation.reset()
        assert simulation.get_ground_truth()["object_001"].motion_state == "on_belt"
        assert simulation.simulation_time_s == 0
        simulation.step(.1)
    finally:
        simulation.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires OpenGL/WSLg")
def test_rgbd_observes_native_fall_when_the_exit_is_in_line_of_sight():
    np = pytest.importorskip("numpy")
    simulation = ConveyorSimulation(.3, exit_mode="fall")
    try:
        initial = simulation.get_observation()
        simulation.step(12.)
        # The belt camera naturally cannot see the floor behind the deck.
        # Look vertically at the exit for this renderer/physics update check.
        simulation.config.getFrame("camera_rgbd").setPosition([1.5, 0, 2.5])
        observation = simulation.get_observation()
        state = simulation.get_ground_truth()["object_001"]
        assert state.motion_state == "settled"
        assert observation.simulation_time_s == 12.
        calibration = observation.calibration
        world_to_camera = np.linalg.inv(np.asarray(calibration.camera_to_world))
        centre = world_to_camera @ np.array([*state.position_m, 1.0])
        projection = np.asarray(calibration.intrinsics) @ centre[:3]
        u, v = (int(round(float(x))) for x in projection[:2] / projection[2])
        assert 0 <= u < calibration.width and 0 <= v < calibration.height
        assert 2.35 < observation.depth_m[v, u] < 2.495
        assert not np.array_equal(initial.rgb, observation.rgb)
    finally:
        simulation.close()
