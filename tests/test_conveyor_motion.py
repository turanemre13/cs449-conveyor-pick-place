"""Check deterministic transport, stopping, replay, and controller handoff."""

from math import pi

import pytest

from conveyor.simulation import ConveyorSimulation

pytest.importorskip("robotic")


def test_speed_and_time_are_independent_of_step_partition():
    first = ConveyorSimulation(.2)
    second = ConveyorSimulation(.2)
    start = first.get_ground_truth()["object_001"]
    first.step(1.0)
    for _ in range(10):
        second.step(.1)
    state = first.get_ground_truth()["object_001"]
    other = second.get_ground_truth()["object_001"]
    assert state.simulation_time_s == pytest.approx(1.0)
    assert state.position_m == pytest.approx((start.position_m[0] + .2, 0, .79))
    assert other.position_m == pytest.approx(state.position_m)
    assert state.velocity_m_s == (.2, 0, 0)
    assert state.quaternion_wxyz == start.quaternion_wxyz
    assert state.dimensions_m == (.06, .045, .08)
    assert state.frame == "world"


def test_transport_path_has_no_penetrations_and_stops_at_exit():
    simulation = ConveyorSimulation(.2)
    for _ in range(12):
        simulation.step(1.0)
        simulation.config.computeCollisions()
        assert simulation.config.getCollisions(-1e-5) == []
    state = simulation.get_ground_truth()["object_001"]
    assert simulation.object_at_exit
    assert state.position_m[0] == pytest.approx(1.17)
    assert state.velocity_m_s == (0, 0, 0)
    simulation.step(2.0)
    assert simulation.get_ground_truth()["object_001"].position_m == state.position_m
    simulation.reset()
    reset = simulation.get_ground_truth()["object_001"]
    assert reset.simulation_time_s == 0
    assert reset.position_m == pytest.approx((-1.05, 0, .79))
    assert reset.motion_state == "on_belt"


def test_release_stops_prescribed_motion_and_zero_speed_is_stationary():
    simulation = ConveyorSimulation(0)
    initial = simulation.get_ground_truth()["object_001"]
    seam_initial = simulation.config.getFrame("belt_seam_00").getPose().copy()
    simulation.step(3.0)
    assert simulation.get_ground_truth()["object_001"].position_m == initial.position_m
    assert simulation.config.getFrame("belt_seam_00").getPose() == pytest.approx(seam_initial)
    simulation.release_object("object_001")
    simulation.config.getFrame("object_001").setPosition([0, .4, 1.0])
    simulation.step(.5)
    state = simulation.get_ground_truth()["object_001"]
    assert state.position_m == (0, .4, 1.0)
    assert state.motion_state == "released"
    assert state.velocity_m_s is None  # External controller velocity is unknown.
    with pytest.raises(KeyError):
        simulation.release_object("missing")


def test_visual_seams_loop_without_moving_the_chassis():
    simulation = ConveyorSimulation(.12)
    belt_pose = simulation.config.getFrame("conveyor").getPose().copy()
    initial = simulation.config.getFrame("belt_seam_00").getPosition().copy()
    simulation.step(1.0)
    assert simulation.config.getFrame("belt_seam_00").getPosition()[0] == pytest.approx(initial[0] + .12)
    assert simulation.config.getFrame("conveyor").getPose() == pytest.approx(belt_pose)
    one_lap_seconds = (2 * 2.4 + 2 * pi * .0401) / .12
    simulation.step(one_lap_seconds - 1.0)
    assert simulation.config.getFrame("belt_seam_00").getPosition() == pytest.approx(initial)


def test_invalid_speed_and_time_are_rejected_before_mutation():
    for speed in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            ConveyorSimulation(speed)
    simulation = ConveyorSimulation()
    for dt in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            simulation.step(dt)
    assert simulation.simulation_time_s == 0
