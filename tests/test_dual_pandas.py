"""Check synchronized collision validation and solver-free replay."""

import pytest

from conveyor.control import dual_pandas
from conveyor.control.dual_pandas import (
    ArmCommand, PlaybackFrame, PreparedPandaBatch, play_panda_batch, prepare_panda_batch,
)
from conveyor.planning.panda import PandaPlanningError, TcpPose
from conveyor.planning.static_panda import plan_static_panda
from conveyor.simulation import ConveyorSimulation


def test_prepared_playback_overlaps_arms_without_planning_during_motion(monkeypatch):
    pytest.importorskip("robotic")
    simulation = ConveyorSimulation(0.022, scenario="sparse", seed=21, object_count=2)
    try:
        original = simulation.get_ground_truth()
        batch = prepare_panda_batch(simulation)
        assert simulation.get_ground_truth() == original
        assert batch.overlap_s > 5.0
        assert {job.robot_id for job in batch.jobs} == {"robot_a", "robot_b"}

        def unexpected_planning(*args, **kwargs):
            pytest.fail("Planning ran during playback")

        for name in ("plan_moving_panda", "execute_panda_pickup", "execute_panda_placement", "_validate_clearance"):
            monkeypatch.setattr(dual_pandas, name, unexpected_planning)
        witnessed = []

        def observe(now):
            config = simulation.config
            if all(
                (parent := config.getFrame(job.object_id).getParent()) is not None
                and parent.name.endswith("_gripper") for job in batch.jobs
            ):
                witnessed.append(now)
            return True

        assert play_panda_batch(simulation, batch, after_step=observe)
        assert witnessed
        for job in batch.jobs:
            obj = simulation.get_ground_truth()[job.object_id]
            target = simulation.get_targets()[job.placement.target_id]
            assert obj.position_m[:2] == pytest.approx(target.position_m[:2], abs=0.001)
            assert simulation.config.getFrame(obj.object_id).getParent().name == "world"

        simulation.reset()
        assert play_panda_batch(simulation, batch, after_step=lambda now: now < 0.15) is False
        simulation.reset()
        object_id = simulation.object_ids[0]
        x, y, z = simulation.config.getFrame(object_id).getPosition()
        simulation.config.getFrame(object_id).setPosition((x + 0.03, y, z))
        with pytest.raises(ValueError, match="validated playback start"):
            play_panda_batch(simulation, batch)
    finally:
        simulation.close()


def test_shared_validation_rejects_individually_valid_arms_meeting_at_the_same_tcp():
    pytest.importorskip("robotic")
    simulation = ConveyorSimulation(0.0)
    try:
        robots = simulation.get_robot_states()
        shared_target = TcpPose((0.0, -0.10, 1.1), (1.0, 0.0, 0.0, 0.0))
        paths = tuple(plan_static_panda(simulation.config, robot, shared_target, duration_s=4.0) for robot in robots.values())
        start = tuple(ArmCommand(robot.arm_joint_positions_rad, robot.finger_joint_positions_m[0]) for robot in robots.values())
        end = tuple(ArmCommand(path.samples[-1].positions_rad, command.finger_m) for path, command in zip(paths, start))
        batch = PreparedPandaBatch(tuple(robots), (PlaybackFrame(0.0, start), PlaybackFrame(4.0, end)), (), (), 0.0)
        with pytest.raises(PandaPlanningError, match=r"Combined clearance failed.*([ab])_.* / [ab]_"):
            dual_pandas._play(simulation, batch, check_clearance=True)
    finally:
        simulation.close()
