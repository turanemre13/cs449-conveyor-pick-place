"""Real CPU PhysX checks; these tests do not open a graphics window."""

from dataclasses import replace
from math import isfinite, sqrt

import pytest

from conveyor.simulation.falling import FallingObjects
from conveyor.simulation.scenarios import generate_scenario

ry = pytest.importorskip("robotic")


def _scene(specs):
    config = ry.Config()
    config.addFrame("floor").setShape(ry.ST.box, [6, 4, .1]).setPosition([0, 0, -.05]).setContact(1)
    config.addFrame("conveyor").setShape(ry.ST.ssBox, [2.4, .4, .08, .01]).setPosition([0, 0, .71]).setContact(1)
    for spec in specs:
        frame = config.addFrame(spec.object_id)
        if spec.geometry_type == "box":
            frame.setShape(ry.ST.ssBox, [*spec.dimensions_m, .005])
        else:
            frame.setShape(ry.ST.cylinder, [spec.dimensions_m[2], spec.dimensions_m[0] / 2])
        frame.setPosition(list(spec.position_m))
    return config


def test_gravity_preserves_launch_momentum_and_floor_supports_body():
    spec = generate_scenario("single")[0]
    config = _scene((spec,))
    physics = FallingObjects(config)
    try:
        physics.add_object(spec, [1.35, 0, .79, 1, 0, 0, 0], (.2, 0, 0))
        physics.step(.1)
        position = config.getFrame(spec.object_id).getPosition()
        assert position[0] == pytest.approx(1.37, abs=.001)
        assert position[2] < .75
        assert physics.get_velocity(spec.object_id)[0] == pytest.approx(.2, abs=.001)
        assert physics.get_velocity(spec.object_id)[2] == pytest.approx(-.981, abs=.02)
        physics.step(3.)
        position = config.getFrame(spec.object_id).getPosition()
        assert .018 <= position[2] <= .09
        assert physics.is_settled(spec.object_id)
        assert all(isfinite(x) for x in config.getFrame(spec.object_id).getPose())
    finally:
        physics.close()


@pytest.mark.parametrize("geometry_type", ["box", "cylinder"])
def test_edge_contact_generates_tipping_and_tumble_without_angular_kicks(geometry_type):
    spec = generate_scenario("single")[0]
    if geometry_type == "cylinder":
        spec = replace(spec, geometry_type="cylinder", dimensions_m=(.06, .06, .08))
    config = _scene((spec,))
    physics = FallingObjects(config)
    try:
        physics.add_object(spec, [1.204, 0, .79, 1, 0, 0, 0], (.12, 0, 0))
        assert physics.get_angular_velocity(spec.object_id) == (0., 0., 0.)
        physics.step(.15)
        angular = physics.get_angular_velocity(spec.object_id)
        assert sqrt(sum(x * x for x in angular)) > .1
        assert config.getFrame(spec.object_id).getPosition()[2] < .79
        physics.step(3.)
        pose = config.getFrame(spec.object_id).getPose()
        assert pose[0] > 1.25
        assert .018 < pose[2] < .09
        assert abs(pose[3]) < .99  # The body really rotated, not just translated.
        assert physics.is_settled(spec.object_id)
    finally:
        physics.close()


def test_adding_actor_preserves_existing_pose_and_linear_angular_velocity():
    first = generate_scenario("single")[0]
    second = replace(first, object_id="object_002")
    config = _scene((first, second))
    physics = FallingObjects(config)
    try:
        physics.add_object(first, [1.204, 0, .79, 1, 0, 0, 0], (.12, 0, 0))
        physics.step(.1)
        pose = config.getFrame(first.object_id).getPose().copy()
        velocity = physics.get_velocity(first.object_id)
        angular = physics.get_angular_velocity(first.object_id)
        physics.add_object(second, [1.35, .2, .9, 1, 0, 0, 0], (.2, 0, 0))
        assert config.getFrame(first.object_id).getPose() == pytest.approx(pose, abs=1e-6)
        assert physics.get_velocity(first.object_id) == pytest.approx(velocity, abs=1e-6)
        assert physics.get_angular_velocity(first.object_id) == pytest.approx(angular, abs=1e-6)
        assert physics.get_velocity(second.object_id) == pytest.approx((.2, 0, 0))
    finally:
        physics.close()


def test_native_objects_collide_with_each_other_and_lifecycle_releases_ownership():
    first = generate_scenario("single")[0]
    second = replace(first, object_id="object_002")
    config = _scene((first, second))
    physics = FallingObjects(config)
    try:
        physics.add_object(first, [1.35, 0, .04, 1, 0, 0, 0], (0, 0, 0))
        physics.add_object(second, [1.35, 0, .40, 1, 0, 0, 0], (0, 0, 0))
        physics.step(.3)
        bottom = config.getFrame(first.object_id).getPosition()
        top = config.getFrame(second.object_id).getPosition()
        assert bottom[2] > .018
        assert top[2] > .10  # It contacts the lower box before reaching the floor.
        physics.step(1.7)  # A real stack may topple rather than stay perfectly aligned.
        physics.remove_object(first.object_id)
        config.getFrame(first.object_id).setPosition([0, .4, 1.0])
        physics.step(.1)
        assert config.getFrame(first.object_id).getPosition() == pytest.approx((0, .4, 1.0))
        physics.reset()
        assert physics.object_ids == ()
        physics.close()
        with pytest.raises(RuntimeError):
            physics.step(.01)
        physics.reset()
        assert physics.object_ids == ()
    finally:
        physics.close()


def test_invalid_dynamics_inputs_fail_before_actor_mutation():
    spec = generate_scenario("single")[0]
    config = _scene((spec,))
    physics = FallingObjects(config)
    try:
        for dt in (0, -1, float("inf"), True):
            with pytest.raises(ValueError):
                physics.step(dt)
        with pytest.raises(ValueError):
            physics.add_object(spec, [1.3, 0, .79, 0, 0, 0, 0], (.2, 0, 0))
        assert physics.object_ids == ()
    finally:
        physics.close()
