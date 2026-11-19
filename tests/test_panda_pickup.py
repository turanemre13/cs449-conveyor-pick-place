"""Verify timed jaw closure, belt ownership, and lift in the RAI scene."""

import pytest

from conveyor.control.panda_pickup import execute_panda_pickup
from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.planning.panda import PandaPlanningError
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)
from conveyor.simulation import ConveyorSimulation


@pytest.fixture
def pickup(request):
    pytest.importorskip("robotic")
    simulation = ConveyorSimulation(belt_speed_m_s=getattr(request, "param", 0.12))
    try:
        robots = simulation.get_robot_states()
        objects = schedulable_objects_from_ground_truth(simulation.get_ground_truth(active_only=True))
        scheduler = DeadlineAwareGreedyScheduler()
        candidates = robot_configs_from_snapshots(robots, simulation.get_targets())
        assignment = scheduler.schedule(objects, candidates, simulation.simulation_time_s).assigned[0]
        plan = plan_moving_panda(
            simulation.config, robots[assignment.robot_id], assignment, objects[0].state,
            close_duration_s=scheduler.config.timing.gripper_close_s,
        )
        yield simulation, plan
    finally:
        simulation.close()


@pytest.mark.parametrize("pickup", [0.12, 0.20], indirect=True)
def test_closing_keeps_belt_motion_then_the_attached_box_lifts(pickup):
    simulation, plan = pickup
    original = simulation.get_ground_truth()[plan.object_id]
    initial_robot = simulation.get_robot_states()[plan.robot_id]
    closing_samples = []

    def observe(now):
        if plan.intercept_time_s + 1e-6 < now <= plan.end_time_s + 1e-6:
            box = simulation.get_ground_truth()[plan.object_id]
            robot = simulation.get_robot_states()[plan.robot_id]
            assert box.motion_state == "on_belt"
            assert box.position_m == pytest.approx(plan.object_position_at(now))
            assert robot.finger_joint_positions_m[0] == pytest.approx(robot.finger_joint_positions_m[1])
            closing_samples.append(robot.finger_joint_positions_m[0])
        return True

    result = execute_panda_pickup(simulation, plan, after_step=observe)
    assert len(closing_samples) > 2
    assert closing_samples[0] > closing_samples[-1] == pytest.approx(plan.closure.closed_m)
    assert result.interception.success
    assert result.pickup_time_s == pytest.approx(plan.end_time_s)
    assert result.lift_height_m == pytest.approx(0.12, abs=0.002)
    box = simulation.get_ground_truth()[plan.object_id]
    assert box.motion_state == "released"
    assert box.position_m[2] == pytest.approx(original.position_m[2] + result.lift_height_m)
    assert simulation.config.getFrame(plan.object_id).getParent().name == initial_robot.gripper_frame

    simulation.reset()
    assert simulation.get_ground_truth()[plan.object_id] == original
    assert simulation.config.getFrame(plan.object_id).getParent() is None
    assert simulation.get_robot_states()[plan.robot_id].finger_joint_positions_m == initial_robot.finger_joint_positions_m


def test_failed_contact_keeps_box_on_belt_and_reopens_jaws(pickup):
    simulation, plan = pickup

    def move_box_at_closure(now):
        if now >= plan.end_time_s - 1e-6:
            box = simulation.config.getFrame(plan.object_id)
            x, y, z = box.getPosition()
            box.setPosition([x, y + 0.003, z])
        return True

    with pytest.raises(PandaPlanningError, match="Grasp contact not verified"):
        execute_panda_pickup(simulation, plan, after_step=move_box_at_closure)

    box = simulation.get_ground_truth()[plan.object_id]
    assert box.motion_state == "on_belt"
    assert simulation.config.getFrame(plan.object_id).getParent() is None
    assert simulation.get_robot_states()[plan.robot_id].finger_joint_positions_m == pytest.approx((0.04, 0.04))
    simulation.step(0.10)
    assert simulation.get_ground_truth()[plan.object_id].position_m[0] > box.position_m[0]


def test_interrupted_closure_leaves_box_on_belt(pickup):
    simulation, plan = pickup
    stop_at = plan.intercept_time_s + 0.12
    assert execute_panda_pickup(simulation, plan, after_step=lambda now: now < stop_at) is None
    assert simulation.get_ground_truth()[plan.object_id].motion_state == "on_belt"
    assert simulation.config.getFrame(plan.object_id).getParent() is None
    assert simulation.get_robot_states()[plan.robot_id].finger_joint_positions_m == pytest.approx((0.04, 0.04))
