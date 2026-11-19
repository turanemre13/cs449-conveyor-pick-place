"""Static Panda pose, path clearance and playback on the shared RAI scene."""

from itertools import pairwise
from math import sqrt

import pytest

from conveyor.control.static_panda import execute_static_panda
from conveyor.planning.static_panda import StaticPlanningError, TcpPose, plan_static_panda, tcp_pose_error
from conveyor.simulation import ConveyorSimulation


TARGET = TcpPose((-0.55, 0.0, 0.90), (sqrt(0.5), 0.0, 0.0, sqrt(0.5)))


@pytest.fixture
def simulation():
    pytest.importorskip("robotic")
    scene = ConveyorSimulation(belt_speed_m_s=0.0)
    try:
        yield scene
    finally:
        scene.close()


def test_pose_error_treats_opposite_quaternion_signs_as_same_orientation():
    actual = tuple(-value for value in TARGET.quaternion_wxyz)
    error = tcp_pose_error(TARGET, TARGET.position_m, actual)
    assert error.position_m == 0.0
    assert error.orientation_rad == pytest.approx(0.0, abs=1e-7)


def test_plan_and_execute_static_pose_without_moving_other_arm(simulation):
    initial = simulation.get_robot_states()
    robot = initial["robot_a"]
    trajectory = plan_static_panda(simulation.config, robot, TARGET)
    assert simulation.get_robot_states() == initial
    assert trajectory.joint_names == robot.arm_joint_names
    assert trajectory.samples[-1].elapsed_s == pytest.approx(3.0)
    assert trajectory.min_clearance_m >= 0.005
    assert trajectory.peak_joint_speed_rad_s <= 1.0
    assert trajectory.endpoint_error.position_m < 0.002
    assert trajectory.endpoint_error.orientation_rad < 0.02
    for start, end in pairwise(trajectory.samples):
        assert 0 < end.elapsed_s - start.elapsed_s <= 0.02
        assert all(
            lower <= q <= upper
            for q, (lower, upper) in zip(end.positions_rad, robot.arm_joint_limits_rad)
        )

    error = execute_static_panda(simulation, trajectory)
    assert error.position_m == pytest.approx(trajectory.endpoint_error.position_m, abs=1e-8)
    assert error.orientation_rad == pytest.approx(trajectory.endpoint_error.orientation_rad, abs=1e-7)
    assert simulation.simulation_time_s == pytest.approx(3.0)
    after = simulation.get_robot_states()
    assert after["robot_b"].arm_joint_positions_rad == initial["robot_b"].arm_joint_positions_rad
    assert after["robot_a"].finger_joint_positions_m == robot.finger_joint_positions_m
    with pytest.raises(ValueError, match="validated start state"):
        execute_static_panda(simulation, trajectory)


def test_plan_rejects_obstacle_against_arm_at_start(simulation):
    import robotic as ry

    config = simulation.config
    finger = config.getFrame("a_finger1").getPosition()
    config.addFrame("test_obstacle").setShape(ry.ST.sphere, [0.02]).setPosition(finger).setContact(1)
    robot = simulation.get_robot_states()["robot_a"]
    with pytest.raises(StaticPlanningError, match=r"Insufficient clearance.*test_obstacle.*0\.000 s"):
        plan_static_panda(config, robot, TARGET)
    assert simulation.get_robot_states()["robot_a"] == robot


def test_plan_rejects_joint_motion_exceeding_speed_limit(simulation):
    robot = simulation.get_robot_states()["robot_a"]
    with pytest.raises(StaticPlanningError, match="Joint speed"):
        plan_static_panda(simulation.config, robot, TARGET, max_joint_speed_rad_s=0.01)


def test_playback_rejects_new_obstacle_along_the_path(simulation):
    import robotic as ry

    config = simulation.config
    robot = simulation.get_robot_states()["robot_a"]
    trajectory = plan_static_panda(config, robot, TARGET)
    middle = trajectory.samples[len(trajectory.samples) // 2]
    config.setJointState(middle.positions_rad, list(robot.arm_joint_names))
    obstacle_position = config.getFrame("a_finger1").getPosition()
    config.setJointState(robot.arm_joint_positions_rad, list(robot.arm_joint_names))
    config.addFrame("mid_path_obstacle").setShape(ry.ST.sphere, [0.02]).setPosition(obstacle_position).setContact(1)
    with pytest.raises(StaticPlanningError, match=r"Insufficient clearance.*mid_path_obstacle at [1-9]\."):
        execute_static_panda(simulation, trajectory)
    assert simulation.simulation_time_s == 0.0
    assert simulation.get_robot_states()["robot_a"].arm_joint_positions_rad == pytest.approx(robot.arm_joint_positions_rad)
