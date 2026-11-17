"""Destination-bin geometry, open interiors, and native contact checks."""

import pytest

from conveyor.simulation import ConveyorSimulation
from conveyor.simulation.scene import build_dual_arm_scene

ry = pytest.importorskip("robotic")


def test_bin_reference_frames_and_metadata_are_stable():
    config = build_dual_arm_scene()
    for suffix, x, robot in (("a", -1.0, "robot_a"), ("b", 1.0, "robot_b")):
        name = f"target_bin_{suffix}"
        frame = config.getFrame(name)
        assert frame.getPosition() == pytest.approx((x, -.85, .75))
        assert frame.getAttributes()["inner_size_m"] == pytest.approx((.30, .24, .14))
        assert frame.getAttributes()["preferred_robot_id"] == robot
        assert config.getFrame(f"{name}_place").getPosition() == pytest.approx((x, -.85, .85))
        assert config.getFrame(f"{name}_approach").getPosition() == pytest.approx((x, -.85, 1.05))
    assert config.getJointDimension() == 14
    assert config.getCollisions(-1e-5) == []


@pytest.mark.parametrize("suffix", ["a", "b"])
def test_bin_has_an_open_interior_and_collidable_walls(suffix):
    simulation = ConveyorSimulation()
    try:
        config = simulation.config
        name = f"target_bin_{suffix}"
        centre = config.getFrame(name).getPosition()
        probe = config.addFrame("bin_probe")
        probe.setShape(ry.ST.ssBox, [.06, .045, .08, .005]).setContact(1)
        probe.setPosition([float(centre[0]), float(centre[1]), .79])
        config.computeCollisions()
        assert config.getCollisions(-1e-5) == []

        probe.setPosition([float(centre[0]) - .156, float(centre[1]), .82])
        # RAI 0.4.1 needs configuration invalidation after frame-only edits.
        config.setJointState(config.getJointState())
        config.computeCollisions()
        assert any(
            {first, second} == {"bin_probe", f"{name}_wall_left"}
            for first, second, _ in config.getCollisions(-1e-5)
        )
    finally:
        simulation.close()


def test_bin_bottom_supports_a_native_physics_drop():
    scene = build_dual_arm_scene()
    config = ry.Config()
    for name in scene.getFrameNames():
        if name.startswith("target_bin_a"):
            source = scene.getFrame(name)
            if source.getShapeType() != ry.ST.none:
                config.addFrame(name).setShape(source.getShapeType(), source.getSize()).setPose(source.getPose()).setContact(1)
    probe = config.addFrame("bin_probe")
    probe.setShape(ry.ST.ssBox, [.06, .045, .08, .005]).setPosition([-1.0, -.85, .98]).setMass(.12).setContact(1)
    engine = ry.Simulation(config, ry.SimulationEngine.physx, 0)
    for _ in range(200):
        engine.step([], .005, ry.ControlMode.none)
    position = probe.getPosition()
    assert position == pytest.approx((-1.0, -.85, .79), abs=.005)
