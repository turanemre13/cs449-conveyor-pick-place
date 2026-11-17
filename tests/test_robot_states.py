"""Snapshot contracts against actual RAI frames and joint state."""

from dataclasses import FrozenInstanceError
from math import sqrt

import pytest

from conveyor.simulation.robots import get_robot_snapshots
from conveyor.simulation.scene import build_dual_arm_scene, build_minimal_scene

ry = pytest.importorskip("robotic")


def test_dual_robot_snapshots_use_world_poses_and_exact_arm_names():
    config = build_dual_arm_scene()
    states = get_robot_snapshots(config, 2.5)
    assert set(states) == {"robot_a", "robot_b"}
    assert config.getJointDimension() == 14  # Reading does not activate fingers.
    names = config.getJointNames()
    positions = config.getJointState()
    limits = config.getJointLimits()
    for robot_id, prefix, x in (("robot_a", "a_", -.55), ("robot_b", "b_", .55)):
        state = states[robot_id]
        assert state.robot_id == robot_id
        assert state.simulation_time_s == 2.5
        assert state.frame == "world"
        assert state.base_frame == f"{prefix}panda_base"
        assert state.gripper_frame == f"{prefix}gripper"
        assert state.base_position_m == pytest.approx((x, -.6, .75))
        assert state.base_quaternion_wxyz == pytest.approx((sqrt(.5), 0, 0, sqrt(.5)))
        gripper = config.getFrame(state.gripper_frame)
        assert state.gripper_position_m == pytest.approx(gripper.getPosition())
        assert state.gripper_quaternion_wxyz == pytest.approx(gripper.getQuaternion())
        assert state.arm_joint_names == tuple(f"{prefix}panda_joint{i}" for i in range(1, 8))
        assert len(state.arm_joint_positions_rad) == len(state.arm_joint_limits_rad) == 7
        for name, value, bounds in zip(state.arm_joint_names, state.arm_joint_positions_rad, state.arm_joint_limits_rad):
            index = names.index(name)
            assert value == positions[index]
            assert bounds == (limits[0][index], limits[1][index])
        assert state.finger_joint_positions_m == pytest.approx((.04, .04))
        assert state.finger_joint_names == (f"{prefix}panda_finger_joint1", f"{prefix}panda_finger_joint2")


def test_changes_to_one_arm_and_inactive_finger_are_reported_without_mutating_old_snapshots():
    config = build_dual_arm_scene()
    before = get_robot_snapshots(config, 0)
    config.setJointState([-1.0], ["a_panda_joint2"])
    config.getFrame("a_panda_finger_joint1").setJointState([.025])
    after = get_robot_snapshots(config, 0)
    assert after["robot_b"] == before["robot_b"]
    assert before["robot_a"].arm_joint_positions_rad[1] == pytest.approx(-.5)
    assert after["robot_a"].arm_joint_positions_rad[1] == pytest.approx(-1.0)
    assert after["robot_a"].gripper_position_m != before["robot_a"].gripper_position_m
    assert before["robot_a"].finger_joint_positions_m == pytest.approx((.04, .04))
    assert after["robot_a"].finger_joint_positions_m == pytest.approx((.025, .025))
    assert config.getJointDimension() == 14
    with pytest.raises(FrozenInstanceError):
        after["robot_a"].robot_id = "changed"


def test_unrelated_active_joint_does_not_shift_arm_mapping():
    config = ry.Config()
    config.addFrame("fixture_root")
    config.addFrame("fixture_slide", "fixture_root").setJoint(ry.JT.transX, [-1, 1]).setJointState([.3])
    config.addConfigurationCopy(build_dual_arm_scene())
    assert config.getJointNames()[0] == "fixture_slide"
    states = get_robot_snapshots(config, .5)
    assert states["robot_a"].arm_joint_positions_rad == pytest.approx((0, -.5, 0, -2, 0, 2, -.5))
    assert states["robot_b"].arm_joint_positions_rad == states["robot_a"].arm_joint_positions_rad
    assert len(config.getJointNames()) == 15


def test_invalid_timestamp_missing_frames_and_inactive_arm_have_clear_errors():
    config = build_dual_arm_scene()
    for timestamp in (-1, True, float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError, match="simulation_time_s"):
            get_robot_snapshots(config, timestamp)
    with pytest.raises(ValueError, match="b_panda_base"):
        get_robot_snapshots(build_minimal_scene(), 0)
    config.selectJoints(["a_panda_joint3"], True)
    with pytest.raises(ValueError, match="a_panda_joint3"):
        get_robot_snapshots(config, 0)
