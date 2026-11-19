"""Verify shared-scene, serialized Panda motion and live rescheduling."""

from math import dist

import pytest

from conveyor.control.moving_panda import execute_moving_panda
from conveyor.control.serial_pandas import execute_serial_pandas
from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)
from conveyor.simulation import ConveyorSimulation


@pytest.fixture
def scene():
    pytest.importorskip("robotic")
    simulation = ConveyorSimulation(0.12, scenario="clutter", seed=0, object_count=2)
    try:
        yield simulation
    finally:
        simulation.close()


def test_both_arms_place_in_one_scene_without_overlapping_motion(scene):
    robot_joints = {key: state.arm_joint_positions_rad for key, state in scene.get_robot_states().items()}
    original = scene.get_ground_truth()
    moved = set()
    idle_anchors = {}

    def observe(robot_id, phase, now):
        for key, state in scene.get_robot_states().items():
            if key != robot_id:
                anchor = idle_anchors.setdefault((robot_id, key), robot_joints[key])
                # Reattaching an object can recompute the same joint to floating-point precision.
                assert dist(state.arm_joint_positions_rad, anchor) < 1e-9
            elif dist(state.arm_joint_positions_rad, robot_joints[key]) > 1e-9:
                moved.add(key)
            robot_joints[key] = state.arm_joint_positions_rad
        return True

    result = execute_serial_pandas(scene, after_step=observe)
    assert result.unplaced == ()
    assert moved == {"robot_a", "robot_b"}
    first, second = result.placed
    assert first.robot_id == "robot_a"
    assert second.robot_id == "robot_b"
    assert first.planning_time_s == 0.0
    assert second.planning_time_s == pytest.approx(first.finished_at_s)
    assert second.intercepted_at_s > second.planning_time_s

    targets = scene.get_targets()
    for item in result.placed:
        target = targets[item.placement.target_id]
        assert target.preferred_robot_id == item.robot_id
        obj = scene.get_ground_truth()[item.object_id]
        assert obj.position_m[:2] == pytest.approx(target.position_m[:2], abs=0.001)
        assert obj.position_m[2] == pytest.approx(target.position_m[2] + obj.dimensions_m[2] / 2, abs=0.001)
        assert scene.config.getFrame(obj.object_id).getParent().name == "world"

    scene.reset()
    assert scene.get_ground_truth() == original


def test_execution_rejects_stale_collision_track_before_moving(scene):
    robots = scene.get_robot_states()
    objects = schedulable_objects_from_ground_truth(scene.get_ground_truth(active_only=True))
    scheduler = DeadlineAwareGreedyScheduler()
    result = scheduler.schedule(objects, robot_configs_from_snapshots(robots, scene.get_targets()), scene.simulation_time_s)
    assignment = min(result.assigned, key=lambda item: item.deadline_s)
    target = next(item for item in objects if item.object_id == assignment.object_id)
    neighbors = tuple(item.state for item in objects if item.object_id != assignment.object_id)
    plan = plan_moving_panda(
        scene.config, robots[assignment.robot_id], assignment, target.state, neighbors=neighbors,
    )

    other = scene.config.getFrame(neighbors[0].object_id)
    x, y, z = other.getPosition()
    other.setPosition((x + 0.03, y, z))
    with pytest.raises(ValueError, match="Conveyor collision track"):
        execute_moving_panda(scene, plan)
    assert scene.simulation_time_s == 0.0
    assert scene.get_robot_states() == robots


def test_batch_reports_neighbor_that_leaves_the_belt_during_the_first_job():
    pytest.importorskip("robotic")
    simulation = ConveyorSimulation(0.04, scenario="sparse", seed=26, object_count=2)
    try:
        result = execute_serial_pandas(simulation)
        assert [item.object_id for item in result.placed] == ["object_001"]
        assert result.unplaced == (("object_002", "exited_before_placement"),)
    finally:
        simulation.close()
