"""Robot/target API integration and physical destination-container checks."""

from dataclasses import FrozenInstanceError
from math import sqrt
import os

import pytest

from conveyor.simulation import ConveyorSimulation
from conveyor.simulation.falling import FallingObjects
from conveyor.simulation.scene import build_dual_arm_scene, build_minimal_scene
from conveyor.simulation.scenarios import generate_scenario
from conveyor.simulation.targets import get_target_snapshots

pytest.importorskip("robotic")


def test_target_snapshots_read_authored_geometry_and_parent_transforms():
    config = build_dual_arm_scene()
    targets = get_target_snapshots(config, 1.25)
    assert set(targets) == {"target_bin_a", "target_bin_b"}
    for name, x, robot_id in (("target_bin_a", -1., "robot_a"), ("target_bin_b", 1., "robot_b")):
        target = targets[name]
        assert target.target_id == target.bin_frame == name
        assert target.preferred_robot_id == robot_id
        assert target.position_m == pytest.approx((x, -.85, .75))
        assert target.inner_dimensions_m == (.30, .24, .14)
        assert target.place_position_m == pytest.approx((x, -.85, .85))
        assert target.approach_position_m == pytest.approx((x, -.85, 1.05))
        assert target.simulation_time_s == 1.25
        assert target.frame == "world"
    with pytest.raises(FrozenInstanceError):
        targets["target_bin_a"].target_id = "changed"

    root = config.getFrame("target_bin_a")
    root.setPosition([-.9, -.8, .8]).setQuaternion([sqrt(.5), sqrt(.5), 0, 0])
    moved = get_target_snapshots(config, 2.)["target_bin_a"]
    assert moved.place_position_m == pytest.approx((-.9, -.9, .8))
    assert moved.approach_position_m == pytest.approx((-.9, -1.1, .8))
    assert targets["target_bin_a"].position_m == (-1., -.85, .75)


def test_invalid_target_timestamp_geometry_and_missing_frames_are_actionable():
    config = build_dual_arm_scene()
    for timestamp in (-1, True, float("nan"), "1"):
        with pytest.raises(ValueError, match="simulation_time_s"):
            get_target_snapshots(config, timestamp)
    with pytest.raises(ValueError, match="target_bin_a"):
        get_target_snapshots(build_minimal_scene(), 0)
    frame = config.getFrame("target_bin_a")
    attrs = frame.getAttributes()
    attrs["inner_size_m"] = [0, .24, .14]
    frame.setAttributes(attrs)
    with pytest.raises(ValueError, match="positive"):
        get_target_snapshots(config, 0)


def test_robot_target_api_times_close_cache_and_reset_remain_consistent():
    simulation = ConveyorSimulation()
    try:
        initial = simulation.get_robot_states()
        simulation.config.setJointState([-1.1], ["a_panda_joint2"])
        simulation.config.getFrame("a_panda_finger_joint1").setJointState([.025])
        simulation.step(.25)
        robots, targets = simulation.get_robot_states(), simulation.get_targets()
        assert robots["robot_a"].arm_joint_positions_rad[1] == pytest.approx(-1.1)
        assert robots["robot_b"].arm_joint_positions_rad == pytest.approx(initial["robot_b"].arm_joint_positions_rad)
        assert robots["robot_a"].finger_joint_positions_m == pytest.approx((.025, .025))
        assert all(state.simulation_time_s == .25 for state in (*robots.values(), *targets.values()))
        simulation.close()
        assert simulation.get_robot_states() == robots
        assert simulation.get_targets() == targets
        returned = simulation.get_robot_states()
        returned.clear()
        assert simulation.get_robot_states() == robots
        simulation.reset()
        reset = simulation.get_robot_states()
        assert reset["robot_a"].arm_joint_positions_rad[1] == pytest.approx(-1.)
        assert reset["robot_a"].finger_joint_positions_m == pytest.approx((.04, .04))
        assert reset["robot_a"].simulation_time_s == 0
    finally:
        simulation.close()


def test_simulation_step_refreshes_collision_candidates_after_frame_changes():
    simulation = ConveyorSimulation(0)
    try:
        config = simulation.config
        assert config.getCollisions(-1e-5) == []  # Prime the native broadphase.
        joints = config.getJointState().copy()
        target = simulation.get_targets()["target_bin_a"]
        simulation.release_object("object_001")
        config.getFrame("object_001").setPosition([
            target.position_m[0] - .156, target.position_m[1], target.position_m[2] + .07,
        ])
        simulation.step(.02)
        assert config.getJointState() == pytest.approx(joints)
        assert any(
            {first, second} == {"object_001", "target_bin_a_wall_left"}
            for first, second, _ in config.getCollisions(-1e-5)
        )
    finally:
        simulation.close()


def test_falling_world_includes_the_actual_target_bottom_and_walls():
    config = build_dual_arm_scene()
    target = get_target_snapshots(config, 0)["target_bin_a"]
    physics = FallingObjects(config)
    try:
        spec = generate_scenario("single")[0]
        x, y, floor_z = target.position_m
        physics.add_object(spec, [x, y, floor_z + .25, 1, 0, 0, 0], (0, 0, 0))
        physics.step(2.)
        position = config.getFrame("object_001").getPosition()
        assert position == pytest.approx((x, y, floor_z + .04), abs=.005)
        assert physics.is_settled("object_001")
    finally:
        physics.close()


@pytest.mark.skipif(os.environ.get("CS449_TEST_RGBD") != "1", reason="Requires OpenGL/WSLg")
def test_camera_and_config_survive_removal_and_reset_collision_rebuilds():
    simulation = ConveyorSimulation(.2)
    try:
        config = simulation.config
        original = simulation.get_observation()
        assert original.depth_m[180, 139] == pytest.approx(1.67, abs=1e-4)
        config.computeCollisions()
        simulation.step(20.)
        removed = simulation.get_observation()
        assert simulation.config is config
        assert removed.simulation_time_s == 20.
        assert removed.depth_m[180, 139] == pytest.approx(1.75, abs=1e-4)
        assert config.getCollisions(-1e-5) == []
        simulation.reset()
        restored = simulation.get_observation()
        assert simulation.config is config
        assert restored.simulation_time_s == 0
        assert restored.depth_m[180, 139] == pytest.approx(1.67, abs=1e-4)
        assert restored.calibration == original.calibration
        assert original.depth_m[180, 139] == pytest.approx(1.67, abs=1e-4)
    finally:
        simulation.close()
