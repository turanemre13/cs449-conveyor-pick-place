"""Verify the bin handoff and retreat after a scheduled moving pickup."""

import pytest

from conveyor.control.panda_pickup import execute_panda_pickup
from conveyor.control.panda_placement import execute_panda_placement
from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.planning.panda import PandaPlanningError
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)
from conveyor.simulation import ConveyorSimulation


@pytest.fixture
def held_box(request):
    pytest.importorskip("robotic")
    simulation = ConveyorSimulation(belt_speed_m_s=getattr(request, "param", 0.12))
    try:
        robots = simulation.get_robot_states()
        objects = schedulable_objects_from_ground_truth(simulation.get_ground_truth(active_only=True))
        scheduler = DeadlineAwareGreedyScheduler()
        candidates = robot_configs_from_snapshots(robots, simulation.get_targets())
        assignment = scheduler.schedule(objects, candidates, simulation.simulation_time_s).assigned[0]
        pickup = plan_moving_panda(
            simulation.config, robots[assignment.robot_id], assignment, objects[0].state,
            close_duration_s=scheduler.config.timing.gripper_close_s,
        )
        execute_panda_pickup(simulation, pickup)
        yield simulation, pickup
    finally:
        simulation.close()


@pytest.mark.parametrize("held_box", [0.12, 0.20], indirect=True)
def test_box_stays_in_assigned_bin_after_opening_and_retreat(held_box):
    simulation, pickup = held_box
    parked = simulation.get_robot_states()["robot_b"].arm_joint_positions_rad
    target = simulation.get_targets()["target_bin_a"]
    result = execute_panda_placement(simulation, pickup)
    box = simulation.get_ground_truth()[pickup.object_id]
    assert result.target_id == target.target_id
    assert result.floor_gap_m == pytest.approx(0.0, abs=0.001)
    assert box.position_m[:2] == pytest.approx(target.position_m[:2], abs=0.001)
    assert simulation.config.getFrame(pickup.object_id).getParent().name == "world"
    assert result.release_time_s < simulation.simulation_time_s
    robots = simulation.get_robot_states()
    assert robots["robot_a"].finger_joint_positions_m == pytest.approx((0.04, 0.04))
    assert robots["robot_a"].gripper_position_m[2] == pytest.approx(target.approach_position_m[2], abs=0.002)
    assert robots["robot_b"].arm_joint_positions_rad == parked

    simulation.step(0.3)
    assert simulation.get_ground_truth()[pickup.object_id].position_m == box.position_m
    simulation.reset()
    assert simulation.get_ground_truth()[pickup.object_id].motion_state == "on_belt"
    assert simulation.config.getFrame(pickup.object_id).getParent() is None


def test_obstructed_bin_leaves_box_attached_without_starting_transfer(held_box):
    import robotic as ry

    simulation, pickup = held_box
    target = simulation.get_targets()["target_bin_a"]
    x, y, floor_z = target.position_m
    simulation.config.addFrame("bin_obstacle").setShape(ry.ST.sphere, [0.032]).setContact(1).setPosition([x, y, floor_z + 0.04])
    robot = simulation.get_robot_states()[pickup.robot_id]
    box = simulation.get_ground_truth()[pickup.object_id]
    with pytest.raises(PandaPlanningError):
        execute_panda_placement(simulation, pickup)
    assert simulation.get_robot_states()[pickup.robot_id] == robot
    assert simulation.get_ground_truth()[pickup.object_id] == box
    assert simulation.config.getFrame(pickup.object_id).getParent().name == robot.gripper_frame
