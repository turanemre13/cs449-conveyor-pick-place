"""Optional RAI checks for geometry, naming, and working collision detection."""

from math import isfinite, sqrt

import pytest

from conveyor.simulation import build_minimal_scene
from conveyor.simulation.scene import build_dual_arm_scene

pytest.importorskip("robotic")


def test_minimal_scene_geometry_and_initial_collisions():
    config = build_minimal_scene()
    names = config.getFrameNames()
    assert len(names) == len(set(names))
    assert {"world", "conveyor", "object_001", "a_panda_base", "a_gripper"} <= set(names)
    assert config.getJointDimension() == 7
    assert all(name.startswith("a_") for name in config.getJointNames())
    lower_limits, upper_limits = config.getJointLimits()
    for value, lower, upper in zip(config.getJointState(), lower_limits, upper_limits):
        assert isfinite(value)
        assert lower <= value <= upper

    base = config.getFrame("a_panda_base")
    assert base.getPosition() == pytest.approx((0.0, -0.6, 0.75))
    assert base.getQuaternion() == pytest.approx((sqrt(0.5), 0.0, 0.0, sqrt(0.5)))

    belt = config.getFrame("conveyor")
    box = config.getFrame("object_001")
    belt_top = belt.getPosition()[2] + belt.getSize()[2] / 2
    box_bottom = box.getPosition()[2] - box.getSize()[2] / 2
    assert belt_top == pytest.approx(0.75)
    assert box_bottom == pytest.approx(belt_top)
    for axis in (0, 1):
        box_edge = abs(box.getPosition()[axis] - belt.getPosition()[axis]) + box.getSize()[axis] / 2
        assert box_edge < belt.getSize()[axis] / 2

    config.computeCollisions()
    assert config.getCollisions(-1e-5) == []


def test_object_can_move_independently_and_penetration_is_detected():
    config = build_minimal_scene()
    other = build_minimal_scene()
    config.getFrame("object_001").setPosition([-.25, 0.0, .74])
    assert other.getFrame("object_001").getPosition() == pytest.approx((-.25, 0.0, .79))

    config.computeCollisions()
    penetrations = config.getCollisions(-1e-5)
    assert any(
        {first, second} == {"object_001", "conveyor"} and distance < -1e-5
        for first, second, distance in penetrations
    )


def test_dual_arm_scene_names_poses_and_initial_collisions():
    config = build_dual_arm_scene()
    names = config.getFrameNames()
    assert len(names) == len(set(names))
    assert {"a_panda_base", "a_gripper", "b_panda_base", "b_gripper"} <= set(names)
    assert config.getJointDimension() == 14
    joints = config.getJointNames()
    assert sum(name.startswith("a_") for name in joints) == 7
    assert sum(name.startswith("b_") for name in joints) == 7
    lower_limits, upper_limits = config.getJointLimits()
    for value, lower, upper in zip(config.getJointState(), lower_limits, upper_limits):
        assert isfinite(value)
        assert lower <= value <= upper

    for prefix, x in (("a_", -.55), ("b_", .55)):
        base = config.getFrame(f"{prefix}panda_base")
        assert base.getPosition() == pytest.approx((x, -.6, .75))
        assert base.getQuaternion() == pytest.approx((sqrt(.5), 0.0, 0.0, sqrt(.5)))
    assert config.getFrame("conveyor").getSize()[:3] == pytest.approx((2.4, .4, .08))
    assert config.getFrame("object_001").getPosition() == pytest.approx((-1.05, 0.0, .79))

    config.computeCollisions()
    assert config.getCollisions(-1e-5) == []
