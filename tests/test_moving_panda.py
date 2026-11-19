"""Schedule-driven Panda motion on the RAI conveyor and stale-plan rejection."""

from dataclasses import replace

import pytest

from conveyor.control.moving_panda import execute_moving_panda
from conveyor.planning.moving_panda import plan_moving_panda
from conveyor.planning.panda import PandaPlanningError
from conveyor.scheduling import (
    DeadlineAwareGreedyScheduler, robot_configs_from_snapshots, schedulable_objects_from_ground_truth,
)
from conveyor.simulation import ConveyorSimulation


@pytest.fixture(params=[(0.08, 0.0), (0.12, 0.0), (0.20, 0.37), (0.30, 0.0)])
def moving_scene(request):
    pytest.importorskip("robotic")
    speed, elapsed = request.param
    simulation = ConveyorSimulation(belt_speed_m_s=speed, exit_mode="remove")
    try:
        if elapsed:
            simulation.step(elapsed)
        robots = simulation.get_robot_states()
        objects = schedulable_objects_from_ground_truth(simulation.get_ground_truth(active_only=True))
        scheduler_robots = robot_configs_from_snapshots(robots, simulation.get_targets())
        schedule = DeadlineAwareGreedyScheduler().schedule(objects, scheduler_robots, simulation.simulation_time_s)
        assignment = schedule.assigned[0]
        yield simulation, robots[assignment.robot_id], assignment, objects[0].state
    finally:
        simulation.close()


def test_panda_tracks_the_moving_box_at_its_absolute_interception_time(moving_scene):
    simulation, robot, assignment, track = moving_scene
    original_robots = simulation.get_robot_states()
    original_truth = simulation.get_ground_truth()
    plan = plan_moving_panda(simulation.config, robot, assignment, track)
    assert plan.intercept_time_s == assignment.intercept_time_s
    assert plan.min_clearance_m >= 0.005
    assert simulation.get_ground_truth() == original_truth
    assert simulation.get_robot_states() == original_robots

    result = execute_moving_panda(simulation, plan)
    assert result.success
    assert result.alignment.position_error_m < 0.002
    assert result.alignment.velocity_error_m_s < 0.01
    assert simulation.simulation_time_s == pytest.approx(assignment.intercept_time_s)
    parked_arm = next(key for key in original_robots if key != plan.robot_id)
    assert simulation.get_robot_states()[parked_arm].arm_joint_positions_rad == original_robots[parked_arm].arm_joint_positions_rad
    truth = simulation.get_ground_truth()[plan.object_id]
    assert truth.motion_state == "on_belt"
    assert truth.position_m == pytest.approx(assignment.intercept_position_m)


def test_plan_refuses_late_assignment_and_stale_playback(moving_scene):
    simulation, robot, assignment, track = moving_scene
    late = replace(assignment, intercept_time_s=robot.simulation_time_s + 0.2)
    with pytest.raises(PandaPlanningError, match="insufficient time"):
        plan_moving_panda(simulation.config, robot, late, track)

    plan = plan_moving_panda(simulation.config, robot, assignment, track)
    simulation.step(0.02)
    with pytest.raises(ValueError, match="current simulation time"):
        execute_moving_panda(simulation, plan)
    assert simulation.get_robot_states()[robot.robot_id].arm_joint_positions_rad == robot.arm_joint_positions_rad
